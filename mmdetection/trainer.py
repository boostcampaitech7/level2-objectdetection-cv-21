from config import create_config
# MMDetection과 MMEngine 패키지 설치 확인
try:
    from mmcv import Config
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.apis import train_detector
    from mmdet.utils import get_device
except ImportError:
    raise ImportError(
        "MMDetection 관련 패키지가 설치되지 않았습니다. "
        "다음 명령어로 설치해주세요: "
        "pip install -U openmim&&mim install mmdet"
    )

def main():
    """메인 실행 함수"""
    """메인 실행 함수"""
    # 설정 생성
    cfg = create_config('retinanet')

    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

if __name__ == "__main__":
    main()

