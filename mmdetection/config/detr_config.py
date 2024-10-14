import os

from .base_config import BaseConfig
from mmcv import Config


class detr_config(BaseConfig):
    def __init__(self):
        super().__init__()
        # Cascade RCNN config 파일 경로 설정
        self.config_dir = '/data/ephemeral/home/mmdetection/configs/detr/detr_r50_8xb2-150e_coco.py'
        self.model_name = os.path.basename(self.config_dir).split('.')[0]
        try:
            # Config 파일 불러오기
            self.cfg = Config.fromfile(self.config_dir)
        except Exception as e:
            raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    def build_config(self):
        # 기본 설정을 구성하는 함수 호출
        self.cfg = self.setup_config(self.cfg)

        # dataset config 수정
        self.cfg.data.train.classes = self.classes
        self.cfg.data.train.img_prefix = self.data_dir
        self.cfg.data.train.ann_file = self.data_dir + 'train2.json'  # train json 정보
        self.cfg.data.train.pipeline[2]['img_scale'] = (512, 512)  # Resize

        self.cfg.data.val.classes = self.classes
        self.cfg.data.val.img_prefix = self.data_dir
        self.cfg.data.val.ann_file = self.data_dir + 'val2.json'  # val json 정보
        self.cfg.data.val.pipeline[1]['img_scale'] = (512, 512)  # Resize

        self.cfg.data.test.classes = self.classes
        self.cfg.data.test.img_prefix = self.data_dir
        self.cfg.data.test.ann_file = self.data_dir + 'test.json'  # test json 정보
        self.cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # Resize

        self.cfg.data.samples_per_gpu = 16  # Batch size
        # BoF 기법 적용
        # 1. Label Smoothing
        self.cfg.model.bbox_head.loss_cls = dict(
            type='LabelSmoothLoss',  # CrossEntropy 대신 Label Smoothing 적용
            reduction='mean',
            smoothing=0.1  # smoothing 정도 설정
        )

        # 2. IoU Loss
        self.cfg.model.bbox_head.loss_bbox = dict(
            type='IoULoss',  # IoU 기반의 손실 적용
            loss_weight=1.0
        )

        # 3. Data Augmentation
        # 다양한 이미지 변형 기법 적용 (회전, 색상 변형 등)
        self.cfg.data.train.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='RandomRotate', prob=0.5, degree=10),  # 회전 추가
            dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 변형 추가
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]

        # 학습 설정
        self.cfg.runner.max_epochs = 20  # smoke-test 시 1, 실험 시 더 큰 값으로 설정
        
        return self.cfg
