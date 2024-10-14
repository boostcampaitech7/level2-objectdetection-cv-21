import mmcv
from mmcv import Config
import datetime
import uuid
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from config import create_config
import os
import wandb
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import argparse
import shutil
wandb.login()


def inference(cfg, epoch, model_config):
    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=8,
        dist=False,
        shuffle=False)

    # checkpoint path
    print(cfg.work_dir)
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    model = build_detector(cfg.model, test_cfg=cfg.get(f'{model_config}_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = cfg.data.test.classes
    model = MMDataParallel(model.cuda(), device_ids=[0])

    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                    o[2]) + ' ' + str(o[3]) + ' '
        
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
        
    os.makedirs('../level2-objectdetection-cv-21/mmdetection/output', exist_ok=True)
    save_dir = '/data/ephemeral/home/suhyun/level2-objectdetection-cv-21/mmdetection/output'

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(save_dir, f'submission_{epoch}.csv'), index=None)

    # checkpoint 삭제
    shutil.rmtree(cfg.work_dir)

def main():
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Inference')
    parser.add_argument('--model_config', type=str, default='test', help='config name without _config')
    parser.add_argument('--wandb_config', type=str, default=None, help='config file')
    parser.add_argument('--epoch', type=str, default='latest', help='epoch number')
    args = parser.parse_args()

    cfg, model_name, output_dir = create_config(args.model_config)
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None

    cfg.seed=2021
    cfg.gpu_ids = [1]
    # cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

    # loading wandb
    run = wandb.init()
    artifact = run.use_artifact(args.wandb_config, type='model')
    artifact_dir = artifact.download(path_prefix=f'{args.epoch}.pth')
    cfg.work_dir = artifact_dir

    inference(cfg, args.epoch, args.model_config)

if __name__ == "__main__":
    main()

import wandb