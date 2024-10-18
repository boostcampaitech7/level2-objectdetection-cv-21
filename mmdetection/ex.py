import os
import argparse
import datetime
import uuid
from typing import Tuple
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import shutil
import wandb
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from config import create_config


def inference(cfg, epoch_number: int, model_config: str) -> None:
    """Run inference on a given model and configuration."""
    # Build dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=8,
        dist=False,
        shuffle=False)

    # Load checkpoint
    checkpoint_path = os.path.join(cfg.work_dir, f'epoch_{epoch_number}.pth')
    model = build_detector(cfg.model, test_cfg=cfg.get(f'{model_config}_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Update model classes and parallelize
    model.CLASSES = cfg.data.test.classes
    model = MMDataParallel(model.cuda(), device_ids=[0])

    # Run single GPU test
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    # Post-processing for submission format
    pred_strings = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()
    class_num = 10

    for i, out in enumerate(output):
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                pred_strings.append(float(j))
                pred_strings.append(float(o[0]))
                pred_strings.append(float(o[1]))
                pred_strings.append(float(o[2]))
                pred_strings.append(float(o[3]))

        file_names.append(image_info['file_name'])

    # Create submission directory and save results
    save_dir = '../level2-objectdetection-cv-21/mmdetection/output'
    os.makedirs(save_dir, exist_ok=True)
    submission = pd.DataFrame()
    submission['PredictionString'] = pred_strings
    submission.to_csv(os.path.join(save_dir, f'submission_{model_config}_epoch_{epoch_number}.csv'), index=None)

    # Remove checkpoint directory
    shutil.rmtree(cfg.work_dir)


def main() -> None:
    """Parse arguments and run inference."""
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Inference')
    parser.add_argument('--model_config', type=str, default='test', help='config name without _config')
    parser.add_argument('--wandb_path', type=str, default=None, help='config file')
    parser.add_argument('--epoch_number', type=int, default=None, help='epoch number')
    args = parser.parse_args()

    cfg, model_name, output_dir = create_config(args.model_config)
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None

    cfg.seed = 2021
    cfg.gpu_ids = [1]

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

    # Initialize wandb and load artifact
    run = wandb.init()
    artifact = run.use_artifact(args.wandb_path, type='model')
    artifact_dir = artifact.download(path_prefix=f'epoch_{args.epoch_number}.pth')
    cfg.work_dir = artifact_dir

    inference(cfg, args.epoch_number, args.model_config)


if __name__ == "__main__":
    main()