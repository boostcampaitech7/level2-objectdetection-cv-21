import os
import uuid
import datetime

# MMDetection과 MMEngine 패키지 설치 확인
try:
    from mmcv import Config
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.apis import train_detector
    from mmdet.datasets import (build_dataloader, build_dataset,
                                replace_ImageToTensor)
    from mmdet.utils import get_device
except ImportError:
    raise ImportError(
        "MMDetection 관련 패키지가 설치되지 않았습니다. "
        "다음 명령어로 설치해주세요: "
        "pip install -U openmim&&mim install mmdet"
    )

# 상수 정의
CONFIG_DIR = '/data/ephemeral/home/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
DATA_DIR = '/data/ephemeral/home/dataset/'
OUTPUT_DIR = '/data/ephemeral/home/output/mmdetection/'
NUM_CLASSES = 10  # Number of classes in the dataset
MODEL_NAME = 'faster_rcnn'

def create_config() -> tuple[Config, str, str]:
    """MMDetection 설정 생성"""
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    try:
        # config file 들고오기
        cfg = Config.fromfile(CONFIG_DIR)    
    except Exception as e:
        raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    # 실험 이름 생성
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_code = str(uuid.uuid4())[:5]
    experiment_dir = os.path.join(OUTPUT_DIR, f"{timestamp}_{random_code}")
    os.makedirs(experiment_dir, exist_ok=True)

    # dataset config 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = DATA_DIR
    cfg.data.train.ann_file = DATA_DIR + 'train2.json' # train json 정보
    cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = DATA_DIR
    cfg.data.val.ann_file = DATA_DIR + 'val2.json' # val json 정보
    cfg.data.val.pipeline[1]['img_scale'] = (512,512) # Resize

    # cfg.data.test.classes = classes
    # cfg.data.test.img_prefix = DATA_DIR
    # cfg.data.test.ann_file = DATA_DIR + 'test.json' # test json 정보
    # cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = 42
    cfg.gpu_ids = [0]
    cfg.work_dir = experiment_dir

    cfg.model.roi_head.bbox_head.num_classes = NUM_CLASSES

    # 학습 설정
    cfg.runner.max_epochs = 1 # 1 only when smoke-test, otherwise 12 or bigger
    # 옵티마이저 설정
    cfg.optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={'project': MODEL_NAME, 'config': cfg._cfg_dict.to_dict()},
             interval=1,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=50,
             bbox_score_thr=0.05,
             )
    ]

    return cfg

def main():
    """메인 실행 함수"""
    # 설정 생성
    cfg = create_config()

    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

if __name__ == "__main__":
    main()
