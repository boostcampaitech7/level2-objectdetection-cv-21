import os
import uuid
import datetime
import wandb

# MMDetection과 MMEngine 패키지 설치 확인
try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.registry import DATASETS
    from mmdet.datasets import CocoDataset
except ImportError:
    raise ImportError(
        "MMDetection 관련 패키지가 설치되지 않았습니다. "
        "다음 명령어로 설치해주세요: "
        "pip install -U openmim && mim install mmengine mmdet"
        )

# 상수 정의
DATA_DIR = '/data/ephemeral/home/dataset/'
OUTPUT_DIR = '/data/ephemeral/home/output/mmdetection/'
TRAIN_JSON, VAL_JSON, TEST_JSON = 'train2.json', 'val2.json', 'test.json'

# 클래스 정의
CLASSES = ["General trash", "Paper", "Paper pack", "Metal",
           "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

@DATASETS.register_module()
class TrashDataset(CocoDataset):
    """
    쓰레기 분류를 위한 커스텀 COCO 데이터셋 클래스.

    MMDetection의 CocoDataset을 상속받아 쓰레기 분류에 특화된 데이터셋을 구현합니다.
    미리 정의된 클래스 레이블을 사용합니다.
    """
    CLASSES = CLASSES

    def __init__(self, **kwargs):
        """데이터셋 초기화"""
        super().__init__(**kwargs)

def create_config() -> Config:
    """
    Detectron2 설정과 호환되는 MMDetection 설정 생성

    Returns:
        Config: MMDetection 설정 객체

    Raises:
        FileNotFoundError: 기본 설정 파일을 찾을 수 없는 경우
        RuntimeError: 설정 생성 중 오류가 발생한 경우
    """
    # 기본 config 불러오기 - Faster R-CNN with R50
    try:
        cfg = Config.fromfile('configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')
    except Exception as e:
        raise RuntimeError(f"설정 파일을 불러오는 데 실패했습니다: {str(e)}")

    # 실험 이름 생성 (Detectron2 방식과 동일)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_code = str(uuid.uuid4())[:5]
    experiment_name = f"{timestamp}_{random_code}"
    experiment_dir = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # ... (나머지 설정 코드는 동일)

    return cfg, timestamp, random_code

def setup_wandb(cfg: Config, timestamp: str, random_code: str) -> str:
    """
    WandB 설정 및 초기화

    Args:
        cfg: MMDetection 설정 객체
        timestamp: 실험 시간 문자열
        random_code: 무작위 생성된 실험 코드

    Returns:
        str: WandB 실행 ID
    """
    cfg_dict = cfg.to_dict()
    wandb.init(
        project="Object Detection",
        name=f"mmdet_{timestamp}_{random_code}",
        config=cfg_dict
    )
    return wandb.run.id

def main():
    """
    메인 실행 함수

    MMDetection 학습 파이프라인을 설정하고 실행합니다.
    """
    # 설정 생성
    cfg, timestamp, random_code = create_config()

    # WandB 설정
    wandb_id = setup_wandb(cfg, timestamp, random_code)
    print(f"Started training with WandB run ID: {wandb_id}")  # 문자열 출력 효과 추가

    # 학습 실행
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == "__main__":
    main()