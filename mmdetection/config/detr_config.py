import os
from .base_config import BaseConfig
from mmcv import Config


class detr_config(BaseConfig):
    def __init__(self, max_epochs=25):
        super().__init__(max_epochs=max_epochs)
        # DETR config 파일 경로 설정
        self.config_dir = '/data/ephemeral/home/mmdetection/configs/detr/detr_r50_8x2_150e_coco.py'
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
        # 파이프라인 수정 (img_scale 인자는 Resize에서만 사용)
        self.cfg.data.train.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),  # img_scale 인자 제거
            dict(
                type='AutoAugment',
                policies=[[
                    dict(
                        type='Resize',
                        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                   (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                   (736, 1333), (768, 1333), (800, 1333)],
                        multiscale_mode='value',
                        keep_ratio=True)
                ],
                          [
                              dict(
                                  type='Resize',
                                  img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                                  multiscale_mode='value',
                                  keep_ratio=True),
                              dict(
                                  type='RandomCrop',
                                  crop_type='absolute_range',
                                  crop_size=(384, 600),
                                  allow_negative_crop=True),
                              dict(
                                  type='Resize',
                                  img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                             (576, 1333), (608, 1333), (640, 1333),
                                             (672, 1333), (704, 1333), (736, 1333),
                                             (768, 1333), (800, 1333)],
                                  multiscale_mode='value',
                                  override=True,
                                  keep_ratio=True)
                          ]]),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]

        self.cfg.data.val.classes = self.classes
        self.cfg.data.val.img_prefix = self.data_dir
        self.cfg.data.val.ann_file = self.data_dir + 'val2.json'  # val json 정보
        self.cfg.data.val.pipeline[1]['img_scale'] = (512, 512)  # Resize

        self.cfg.data.test.classes = self.classes
        self.cfg.data.test.img_prefix = self.data_dir
        self.cfg.data.test.ann_file = self.data_dir + 'test.json'  # test json 정보
        self.cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # Resize

        self.cfg.data.samples_per_gpu = 16  # Batch size

        # bbox_head 설정 수정 (DETR 모델은 roi_head 대신 bbox_head만 사용)
        self.cfg.model.bbox_head.num_classes = self.num_classes
        
        return self.cfg
