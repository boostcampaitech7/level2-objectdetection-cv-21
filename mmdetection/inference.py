import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pycocotools.coco import COCO
import argparse

# 클래스 정의 (Pascal VOC 포맷에 맞춘 클래스명)
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

def inference(cfg, epoch):
    # 데이터셋 및 데이터로더 구축
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    # 체크포인트 경로
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    # 모델 구축 및 체크포인트 로드
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    # 추론 실행 (단일 GPU)
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)

    # COCO 형식의 주석 파일 로드
    coco = COCO(cfg.data.test.ann_file)

    # 추론 결과를 Pascal VOC 포맷에 맞춰서 후처리
    prediction_strings = []
    file_names = []
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        image_id = image_info['id']  # 파일 이름 대신 image_id 사용
        for j, class_output in enumerate(out):
            for bbox in class_output:
                score = bbox[4]
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                prediction_string += f'{j} {score} {xmin} {ymin} {xmax} {ymax} '
    
        prediction_strings.append(prediction_string.strip())
        file_names.append(image_id)  # image_id를 저장


    # 결과를 CSV 파일로 저장
    submission = pd.DataFrame({
        'image_id': file_names,
        'PredictionString': prediction_strings
    })
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=False)
    print(f'Submission file saved at {os.path.join(cfg.work_dir, f"submission_{epoch}.csv")}')

def main(): #/data/ephemeral/home/mmdetection/configs/atss/atss_r50_fpn_1x_coco.py
    parser = argparse.ArgumentParser(description='PyTorch Object Detection Inference')
    parser.add_argument('--config', type=str, default='/data/ephemeral/home/mmdetection/configs/atss/atss_r50_fpn_1x_coco.py', help='config file')
    parser.add_argument('--epoch', type=str, default='latest', help='epoch number')
    args = parser.parse_args()

    # config 파일 로드
    cfg = Config.fromfile(args.config)

    root = '/data/ephemeral/home/dataset/'
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # 이미지 크기 조정
    cfg.data.test.test_mode = True

    cfg.seed = 2021
    cfg.gpu_ids = [0]
    cfg.work_dir = '/data/ephemeral/home/output/mmdetection/2024-10-11_18-44-24_d215e'

    # num_classes 수정
    cfg.model.bbox_head.num_classes = len(classes)

    # Optimizer 설정 (필요 시)
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None

    # Inference 실행
    inference(cfg, args.epoch)

if __name__ == "__main__":
    main()
