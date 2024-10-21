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
        # dataset config 수정
        self.cfg.data.train.classes = self.classes
        self.cfg.data.train.img_prefix = self.data_dir + 'diffusion_data'
        self.cfg.data.train.ann_file = self.data_dir + 'diffusion_data/annotations/annotation.json' # train json 정보
        
        
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
        self.cfg.model.roi_head.mask_head['num_classes'] = self.num_classes
        
        return self.cfg

