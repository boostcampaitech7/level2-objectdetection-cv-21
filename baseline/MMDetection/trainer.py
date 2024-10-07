import wandb
wandb.login()

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

def main(cfg):
    """메인 실행 함수"""
    """메인 실행 함수"""
    
    # 실험 이름 생성
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_code = str(uuid.uuid4())[:5]
    experiment_dir = os.path.join(self.output_dir, f"{timestamp}_{random_code}")
    os.makedirs(experiment_dir, exist_ok=True)

    wandb.init(project="Object Detection", name=f'{model_name}_{random_code}',config=cfg._cfg_dict.to_dict()},
                )
    
    # 설정 생성
    cfg = create_config('faster_rcnn')
    cfg.work_dir = experiment_dir
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "mAP50"},
        "parameters": {
            "lr": {"max": 0.003, "min": 0.0001},
            "weight_decay": {"max": 0.01, "min": 0.0001}
        },
        "early_terminate":{
            "type": "hyperband",
            "s": 3,
            "eta": 2, # half halving or one-third halving. 2 or 3 recommended
            "min_iter": 8,
        }
    }
    
        
   

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='Object Detection')




    main(cfg)

