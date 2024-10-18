import os
import datetime
import uuid
import wandb
import argparse
from typing import Tuple, Dict, Any, Optional

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.utils import get_device
from config import create_config

class TrainingException(Exception):
    """Training pipeline에서 발생하는 예외를 처리하기 위한 커스텀 예외"""
    pass

def load_wandb_artifact(artifact_path: str, artifact_type: str = 'model') -> Optional[wandb.Artifact]:
    """
    Weights & Biases artifact를 불러옵니다.

    Args:
        artifact_path (str): artifact 경로 (예: 'username/project/artifact:version')
        artifact_type (str): artifact 타입 (기본값: 'model')

    Returns:
        Optional[wandb.Artifact]: 불러온 artifact 객체
    """
    try:
        api = wandb.Api()
        run = wandb.init()
        artifact = run.use_artifact(artifact_path, type=artifact_type)
        artifact_dir = artifact.download()
        print(f"Successfully loaded artifact from {artifact_path}")
        return artifact
    except Exception as e:
        print(f"Failed to load artifact: {e}")
        return None

def setup_wandb(experiment_dir: str, model_name: str, random_code: str, cfg: Config) -> wandb.run:
    """
    Weights & Biases 설정을 초기화합니다.

    Args:
        experiment_dir (str): 실험 결과가 저장될 디렉토리
        model_name (str): 모델 이름
        random_code (str): 실험 구분을 위한 랜덤 코드
        cfg (Config): MMDetection 설정 객체

    Returns:
        wandb.Run: 초기화된 wandb run 객체
    """
    try:
        wandb.login()
        run = wandb.init(
            project="pseudo",
            dir=experiment_dir,
            name=f'{model_name}_{random_code}',
            config=cfg._cfg_dict.to_dict()
        )
        return run
    except Exception as e:
        raise TrainingException(f"Failed to initialize wandb: {e}")

def prepare_experiment_dir(output_dir: str) -> Tuple[str, str]:
    """
    실험 디렉토리를 준비합니다.

    Returns:
        Tuple[str, str]: (experiment_dir, random_code)
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_code = str(uuid.uuid4())[:5]
    experiment_dir = os.path.join(output_dir, f"{timestamp}_{random_code}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir, random_code

def save_model_as_artifact(run: wandb.run, model_path: str, model_name: str) -> None:
    """
    모델을 WandB artifact로 저장합니다.

    Args:
        run (wandb.Run): 현재 wandb run
        model_path (str): 모델 파일 경로
        model_name (str): 저장할 모델 이름
    """
    try:
        artifact = wandb.Artifact(
            name=f'{model_name}_model',
            type='model',
            description=f'Trained model: {model_name}'
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        print(f"Successfully saved model as artifact: {model_name}_model")
    except Exception as e:
        print(f"Failed to save model as artifact: {e}")

def main(input_model_name: str, max_epoch: int, pretrained_artifact_path: Optional[str] = None) -> None:
    """
    메인 실행 함수

    Args:
        input_model_name (str): 모델 이름
        max_epoch (int): 최대 에폭 수
        pretrained_artifact_path (Optional[str]): 사전 학습된 모델의 artifact 경로
    """
    try:
        # 설정 생성
        cfg, model_name, output_dir = create_config(input_model_name, max_epochs=max_epoch)
        if not all([cfg, model_name, output_dir]):
            raise ValueError("Invalid configuration values")

        # 실험 디렉토리 준비
        experiment_dir, random_code = prepare_experiment_dir(output_dir)
        cfg.work_dir = experiment_dir

        # Wandb 설정
        run = setup_wandb(experiment_dir, model_name, random_code, cfg)

        # pretrained 모델 artifact 불러오기
        if pretrained_artifact_path:
            artifact = load_wandb_artifact(pretrained_artifact_path)
            if artifact:
                pretrained_model_path = os.path.join(artifact.download(), 'model.pth')
                cfg.load_from = pretrained_model_path

        # 옵티마이저 설정
        cfg.optimizer = dict(
            type='AdamW',
            lr=wandb.config.lr,
            weight_decay=wandb.config.weight_decay
        )

        # 데이터셋 및 모델 준비
        datasets = [build_dataset(cfg.data.train)]
        model = build_detector(cfg.model)
        model.init_weights()

        # 학습 실행
        train_detector(
            model,
            datasets[0],
            cfg,
            distributed=False,
            validate=True,
            meta=dict()
        )

        # 학습된 모델을 artifact로 저장
        final_model_path = os.path.join(experiment_dir, 'latest.pth')
        if os.path.exists(final_model_path):
            save_model_as_artifact(run, final_model_path, model_name)

    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training pipeline for object detection')
    parser.add_argument('--max_epoch', type=int, default=1, help='Maximum number of epochs')
    parser.add_argument('--input_model_name', required=True, help='Input model name')
    parser.add_argument('--inf_path', type=str, help='Config file path')
    parser.add_argument('--pretrained_artifact', type=str, help='Path to pretrained model artifact (e.g., username/project/artifact:version)')
    args = parser.parse_args()

    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "val/bbox_mAP_50"},
        "parameters": {
            "lr": {"max": 0.00009, "min": 0.00002},
            "weight_decay": {"max": 0.001, "min": 0.0001}
        },
        "early_terminate": {
            "type": "hyperband",
            "s": 3,
            "eta": 2,
            "min_iter": 8,
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='pseudo')
    wandb.agent(
        sweep_id,
        function=lambda: main(
            args.input_model_name,
            args.max_epoch,
            args.pretrained_artifact
        ),
        count=10
    )
