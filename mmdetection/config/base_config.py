# base_config.py
import os
import datetime
import uuid

import wandb

from mmdet.utils import get_device

class BaseConfig:
    def __init__(self):
        self.cfg = None
        self.classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
        self.data_dir = '/data/ephemeral/home/dataset/'
        self.output_dir = '/data/ephemeral/home/output/mmdetection/'
        self.num_classes = 10

    def setup_config(self, cfg):
        '''
        아래 내용 중 주석 부분과 변경이 필요한 내용을 build_config에
        복사해서 쓰시되, 
        주석을 체크 해제하고 사용해주세요.
        또한, 모델 별로 설정이 다르니
        이를 조정해줘야 합니다.
        '''
        # cfg.data.train.classes = self.classes
        # cfg.data.train.img_prefix = self.data_dir
        # cfg.data.train.dataset.ann_file = self.data_dir + 'train2.json' # train json 정보
        # cfg.data.train.dataset.pipeline[2]['img_scale'] = (512,512) # Resize

        # cfg.data.val.classes = self.classes
        # cfg.data.val.img_prefix = self.data_dir
        # cfg.data.val.ann_file = self.data_dir + 'val2.json' # val json 정보
        # cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize

        cfg.data.samples_per_gpu = 4  # Batch size

        cfg.seed = 42
        cfg.gpu_ids = [0]

        # self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
        
        # 학습 설정
        cfg.runner.max_epochs = 1 # 1 only when smoke-test, otherwise 12 or bigger
        
        # 옵티마이저 설정
        cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
        cfg.checkpoint_config = None
        cfg.device = get_device()

        # Wandb 설정
        cfg.log_config.hooks = [
            dict(type='TextLoggerHook'),
            dict(
                type='MMDetWandbHook',
                interval=10,
                log_checkpoint=False,
                log_checkpoint_metadata=False,
                num_eval_images=4,
                bbox_score_thr=0.05,
                )
            ]
        # Custom Hook 설정 - CheckpointHook에 'bbox_mAP_50' 기준으로 best 모델 저장
        cfg.custom_hooks = [
            dict(type='CustomCheckpointHook', interval=1, max_keep_ckpts=1, save_best='bbox_mAP_50', save_last=True)
        ]
        
        # print(cfg)
        return cfg
