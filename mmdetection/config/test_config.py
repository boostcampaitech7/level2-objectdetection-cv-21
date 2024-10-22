# retinanet_config.py
import os

from .base_config import BaseConfig
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                                replace_ImageToTensor)


class test_config(BaseConfig):
    def __init__(self, max_epochs=25):
        super().__init__(max_epochs=max_epochs) # BaseConfig로 에폭 넘김
        self.config_dir = '/data/ephemeral/home/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        self.model_name = os.path.basename(self.config_dir).split('.')[0]
        try:
            self.cfg = Config.fromfile(self.config_dir)
        except Exception as e:
            raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    def build_config(self):
        self.cfg = self.setup_config(self.cfg)
        # # dataset config 수정
        # self.cfg.data.train.classes = self.classes
        # self.cfg.data.train.img_prefix = self.data_dir
        # self.cfg.data.train.ann_file = self.data_dir + 'train2.json' # train json 정보
        # # self.cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

        # MultiImageMixDataset으로 데이터셋 변경
        self.cfg.data.train = dict(
            type='MultiImageMixDataset',
            dataset=dict(
                type='CocoDataset',
                ann_file=self.data_dir + 'train2.json',
                img_prefix=self.data_dir,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                ],
                filter_empty_gt=False,
                classes=self.classes
            ),
            pipeline=[
                dict(type='Mosaic', img_scale=(512, 512), pad_val=114.0),
                dict(
                    type='RandomAffine',
                    scaling_ratio_range=(0.1, 2),
                    border=(-512 // 2, -512 // 2)
                ),                                                                          # Affine 변환
                dict(type='MixUp', img_scale=(512, 512), ratio_range=(0.8, 1.2)),           # MixUp 데이터 증강
                dict(type='PhotoMetricDistortion'),                                         # 색상 왜곡
                # dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5), min_crop_size=0.3), # MinIoURandomCrop 추가
                dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5, direction=['vertical']),            # direction이 없으면 horizontal만 함
                dict(type='Normalize', **self.cfg.img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
        )
        
        self.cfg.data.val.classes = self.classes
        self.cfg.data.val.img_prefix = self.data_dir
        self.cfg.data.val.ann_file = self.data_dir + 'val2.json' # val json 정보
        self.cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize
        
        self.cfg.data.samples_per_gpu = 16
                
        self.cfg.data.test.classes = self.classes
        self.cfg.data.test.img_prefix = self.data_dir
        self.cfg.data.test.ann_file = self.data_dir + 'test.json' # test json 정보
        self.cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

        self.cfg.model.roi_head.bbox_head.num_classes = 10
        
        return self.cfg
