# atss_config.py
import os

from .base_config import BaseConfig
from mmcv import Config


class atss_config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.config_dir = '/data/ephemeral/home/mmdetection/configs/atss/atss_r50_fpn_1x_coco.py'
        self.model_name = os.path.basename(self.config_dir).split('.')[0]
        try:
            self.cfg = Config.fromfile(self.config_dir)
        except Exception as e:
            raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    def build_config(self):
        self.cfg = self.setup_config(self.cfg)

        # 데이터셋 클래스 및 경로 설정
        self.cfg.data.train.classes = self.classes
        self.cfg.data.train.img_prefix = self.data_dir
        self.cfg.data.train.ann_file = os.path.join(self.data_dir, 'train2.json')  # train json 정보
        self.cfg.data.train.pipeline[2]['img_scale'] = (512, 512)  # 이미지 크기 조정

        self.cfg.data.val.classes = self.classes
        self.cfg.data.val.img_prefix = self.data_dir
        self.cfg.data.val.ann_file = os.path.join(self.data_dir, 'val2.json')  # val json 정보
        self.cfg.data.val.pipeline[1]['img_scale'] = (512, 512)  # 이미지 크기 조정

        self.cfg.data.test.classes = self.classes
        self.cfg.data.test.img_prefix = self.data_dir
        self.cfg.data.test.ann_file = os.path.join(self.data_dir, 'test.json')  # test json 정보
        self.cfg.data.test.pipeline[1]['img_scale'] = (512, 512)  # 이미지 크기 조정

        # 데이터 로드 및 배치 크기 설정
        self.cfg.data.samples_per_gpu = 16

        # ATSS는 roi_head를 사용하지 않음 -> bbox_head만 설정
        self.cfg.model.bbox_head.num_classes = self.num_classes

        # 학습 설정
        self.cfg.runner.max_epochs = 30  # 30 epochs로 설정, 필요시 변경 가능

        return self.cfg