import os
import uuid
import datetime
import wandb
from dataset.dataset import CocoDetectionDataset

# MMDetection과 MMEngine 패키지 설치 확인
try:
    from mmengine.config import Config
    from mmengine.runner import Runner
except ImportError:
    raise ImportError(
        "MMDetection 관련 패키지가 설치되지 않았습니다. "
        "다음 명령어로 설치해주세요: "
        "pip install -U openmim && mim install mmengine mmdet"
    )

# 상수 정의
DATA_DIR = '/data/ephemeral/home/dataset/'
OUTPUT_DIR = '/data/ephemeral/home/output/mmdetection/'
NUM_CLASSES = 10  # Number of classes in the dataset

def create_config() -> tuple[Config, str, str]:
    """MMDetection 설정 생성"""
    try:
        cfg = Config.fromfile('configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')
    except Exception as e:
        raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    # 실험 이름 생성
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_code = str(uuid.uuid4())[:5]
    experiment_dir = os.path.join(OUTPUT_DIR, f"{timestamp}_{random_code}")
    os.makedirs(experiment_dir, exist_ok=True)

    # 데이터셋 초기화
    train_dataset = CocoDetectionDataset(
        data_path=DATA_DIR,
        ann_file=os.path.join(DATA_DIR, 'train.json')
    )

    val_dataset = CocoDetectionDataset(
        data_path=DATA_DIR,
        ann_file=os.path.join(DATA_DIR, 'val.json')
    )

    # 데이터로더 설정
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        img_ids = [item[2] for item in batch]
        return images, targets, img_ids

    # 데이터셋 설정 적용
    cfg.dataset_type = 'CocoDetectionDataset'
    cfg.data_root = DATA_DIR

    cfg.train_dataloader = dict(
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=train_dataset,
        collate_fn=collate_fn
    )

    cfg.val_dataloader = dict(
        batch_size=16,
        num_workers=4,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=val_dataset,
        collate_fn=collate_fn
    )

    cfg.test_dataloader = cfg.val_dataloader

    # 검증 설정
    cfg.val_evaluator = dict(
        type='CocoMetric',
        ann_file=os.path.join(DATA_DIR, 'val.json'),
        metric='bbox',
        format_only=False
    )

    cfg.test_evaluator = cfg.val_evaluator

    # 모델 설정
    cfg.model.roi_head.bbox_head.num_classes = NUM_CLASSES

    # 학습 설정
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')

    # 옵티마이저 설정
    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    )

    # 체크포인트 설정
    cfg.default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',
            interval=1,
            max_keep_ckpts=3,
            save_best='auto',
            out_dir=experiment_dir
        ),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook')
    )

    return cfg, timestamp, random_code

def setup_wandb(cfg: Config, timestamp: str, random_code: str) -> str:
    """WandB 설정 및 초기화"""
    cfg_dict = cfg.to_dict()
    wandb.init(
        project="Object Detection",
        name=f"mmdet_{timestamp}_{random_code}",
        config=cfg_dict
    )
    return wandb.run.id

def main():
    """메인 실행 함수"""
    # 설정 생성
    cfg, timestamp, random_code = create_config()

    # WandB 설정
    wandb_id = setup_wandb(cfg, timestamp, random_code)
    print(f"Started training with WandB run ID: {wandb_id}")

    # 학습 실행
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == "__main__":
    main()
