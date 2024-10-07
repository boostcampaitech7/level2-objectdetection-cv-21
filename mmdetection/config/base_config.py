# base_config.py
import os
import datetime
import uuid

from mmdet.utils import get_device

class BaseConfig:
    def __init__(self):
        self.cfg = None
        self.classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
        self.data_dir = '/data/ephemeral/home/dataset/'
        self.output_dir = '/data/ephemeral/home/output/mmdetection/'
        self.num_classes = 10

    def setup_config(self, cfg, model_name):
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


        # 실험 이름 생성
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        random_code = str(uuid.uuid4())[:5]
        experiment_dir = os.path.join(self.output_dir, f"{timestamp}_{random_code}")
        os.makedirs(experiment_dir, exist_ok=True)
        cfg.work_dir = experiment_dir

        # self.cfg.model.roi_head.bbox_head.num_classes = self.num_classes
        
        # 학습 설정
        cfg.runner.max_epochs = 1 # 1 only when smoke-test, otherwise 12 or bigger
        
        # 옵티마이저 설정
        cfg.optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
        cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
        cfg.checkpoint_config = dict(max_keep_ckpts=1, interval=1)
        cfg.device = get_device()

        # Wandb 설정
        cfg.log_config.hooks = [
            dict(type='TextLoggerHook'),
            dict(
                type='MMDetWandbHook',
                init_kwargs={'project': "Object Detection", 'name':f'{model_name}_{random_code}','config': cfg._cfg_dict.to_dict()},
                interval=1,
                log_checkpoint=True,
                log_checkpoint_metadata=True,
                num_eval_images=10,
                bbox_score_thr=0.05,
                )
            ]
        
        return cfg