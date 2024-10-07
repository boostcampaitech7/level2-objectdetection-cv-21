# retinanet_config.py
import os

from .base_config import BaseConfig
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                                replace_ImageToTensor)


class retinanet_config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.config_dir = '/data/ephemeral/home/mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_mstrain_640-800_3x_coco.py'
        self.model_name = os.path.basename(self.config_dir).split('.')[0]
        try:
            self.cfg = Config.fromfile(self.config_dir)
        except Exception as e:
            raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    def build_config(self):
        self.cfg = self.setup_config(self.cfg, self.model_name)
        # dataset config 수정
        self.cfg.data.train.classes = self.classes
        self.cfg.data.train.dataset.classes = self.classes
        self.cfg.data.train.dataset.img_prefix = self.data_dir
        self.cfg.data.train.dataset.ann_file = self.data_dir + 'train2.json' # train json 정보
        self.cfg.data.train.dataset.pipeline[2]['img_scale'] = (512,512) # Resize
        
        self.cfg.data.val.classes = self.classes
        self.cfg.data.val.img_prefix = self.data_dir
        self.cfg.data.val.ann_file = self.data_dir + 'val2.json' # val json 정보
        self.cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize
        
        self.cfg.data.samples_per_gpu = 16
                
        # cfg.data.test.classes = classes
        # cfg.data.test.img_prefix = DATA_DIR
        # cfg.data.test.ann_file = DATA_DIR + 'test.json' # test json 정보
        # cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize


        self.cfg.model.bbox_head.num_classes = self.num_classes
        del self.cfg.model.pretrained
        self.cfg.model.backbone.init_cfg.checkpoint = 'open-mmlab://resnext101_64x4d'
        
        # 학습 설정
        self.cfg.runner.max_epochs = 1 # 1 only when smoke-test, otherwise 12 or bigger
        
        return self.cfg
