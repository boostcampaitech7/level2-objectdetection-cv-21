# swin_config.py
import os

from .base_config import BaseConfig
from mmcv import Config


class swin_config(BaseConfig):
    def __init__(self, max_epochs=25):
        super().__init__(max_epochs=max_epochs)
        self.config_dir = '/data/ephemeral/home/mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
        self.model_name = os.path.basename(self.config_dir).split('.')[0]
        try:
            self.cfg = Config.fromfile(self.config_dir)
        except Exception as e:
            raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    def build_config(self):
        self.cfg = self.setup_config(self.cfg)
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
                # 기존 파이프라인에 Mosaic, RandomAffine, 추가적인 증강 기법을 추가
                dict(type='Mosaic', img_scale=(1024, 1024), pad_val=114.0),
                dict(type='RandomAffine', scaling_ratio_range=(0.1, 2), border=(-512 // 2, -512 // 2)), # Affine 변환
                dict(type='MixUp', img_scale=(1024, 1024), ratio_range=(0.8, 1.2)),                     # MixUp 데이터 증강
                # dict(type='PhotoMetricDistortion'),                                                   # 색상 왜곡
                dict(type='Resize', img_scale=[(640, 640), (768, 768)], keep_ratio=True),               # Resize 추가
                dict(type='RandomCrop', crop_size=(512, 512), allow_negative_crop=True),                # RandomCrop 추가
                # dict(type='Expand', mean=[123.675, 116.28, 103.53], ratio_range=(1, 4)),              # Expand는 memory가 너무 많이 터져서 삭제
                dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5), min_crop_size=0.3),             # MinIoURandomCrop 추가
                dict(type='CutOut', n_holes=5, cutout_shape=[(50, 50), (75, 75)]),                      # CutOut 추가
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', **self.cfg.img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
        )

        # 평가 메트릭에서 'segm' 제거
        self.cfg.evaluation = dict(
            interval=1,
            metric=['bbox'],  # 'segm' 제거하고 bbox만 평가
        )
        
        self.cfg.data.val.classes = self.classes
        self.cfg.data.val.img_prefix = self.data_dir
        self.cfg.data.val.ann_file = self.data_dir + 'val2.json' # val json 정보
        self.cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize

        self.cfg.data.test.classes = self.classes
        self.cfg.data.test.img_prefix = self.data_dir
        self.cfg.data.test.ann_file = self.data_dir + 'test.json' # test json 정보
        self.cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

        # print(self.cfg.data)
        # exit()

        self.cfg.data.samples_per_gpu = 8
        
        self.cfg.model.roi_head.bbox_head['num_classes'] = self.num_classes
        # self.cfg.model.roi_head.mask_head['num_classes'] = self.num_classes           # backbone 모델 faster rcnn으로 변경으로 mask_head 삭제
        
        return self.cfg

