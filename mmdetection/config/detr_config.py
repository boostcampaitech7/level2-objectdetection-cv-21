import os
from .base_config import BaseConfig
from mmcv import Config


class detr_config(BaseConfig):
    def __init__(self, max_epochs=25):
        super().__init__(max_epochs=max_epochs)
        # DETR config 파일 경로 설정
        self.config_dir = '/data/ephemeral/home/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py'
        self.model_name = os.path.basename(self.config_dir).split('.')[0]
        self.img_scale = (512, 512)
        try:
            # Config 파일 불러오기
            self.cfg = Config.fromfile(self.config_dir)
        except Exception as e:
            raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    def build_config(self):
        # 클래스 수 설정
        self.num_classes = len(self.classes)  # self.classes의 길이로 설정
        self.cfg.model.bbox_head.num_classes = self.num_classes

        # 기본 설정을 구성하는 함수 호출
        self.cfg = self.setup_config(self.cfg)

        train_pipeline = [
            dict(type='Mosaic', img_scale=self.img_scale),
            dict(
                type='RandomAffine',
                scaling_ratio_range=(0.1, 2),
                border=(-self.img_scale[0] // 2, -self.img_scale[1] // 2)
            ),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **self.cfg.img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]

        # train 데이터셋 설정
        self.cfg.data.train = dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type='CocoDataset',
                ann_file=self.data_dir + 'train2.json',
                img_prefix=self.data_dir,
                classes=self.classes,  # 클래스 설정 추가
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True)
                ],
                filter_empty_gt=False,
            ),
            pipeline=train_pipeline
        )

        self.cfg.data.val.classes = self.classes
        self.cfg.data.val.img_prefix = self.data_dir
        self.cfg.data.val.ann_file = self.data_dir + 'val2.json'  # val json 정보
        self.cfg.data.val.pipeline[1]['img_scale'] = (512, 512)  # Resize

        self.cfg.data.test.classes = self.classes
        self.cfg.data.test.img_prefix = self.data_dir
        self.cfg.data.test.ann_file = self.data_dir + 'test.json'  # test json 정보
        self.cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # Resize

        self.cfg.data.samples_per_gpu = 16  # Batch size
        
        return self.cfg
