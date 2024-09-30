# 필요한 라이브러리와 모듈을 불러옵니다
import argparse

import torch

from config.config_factory import get_config
from engine.tuner import RayTuner
from engine.test_runner import run_test
from utils import EnsemblePredictor


# 하이퍼파라미터 튜닝, 모델 학습 및 테스트를 수행하는 함수
def tune_train_and_test(config):
    """
    하이퍼파라미터 튜닝을 수행하고, 튜닝된 하이퍼파라미터로 모델을 학습 및 테스트 합니다.

    Args:
        config: 모델 및 실험 설정이 포함된 configuration 객체
    """
    # 수행할 하이퍼파라미터 튜닝을 진행하여 최적의 configuration을 얻습니다.
    with RayTuner(config) as ray_tuner:
        result_grid = ray_tuner.tune_and_train()
        # 최적의 configuration을 얻습니다.
        best_result = result_grid.get_best_result(metric="val_loss", mode="min")

    # 최적의 모델을 로드하고 테스트를 수행합니다.
    with best_result.checkpoint.as_directory() as ckpt_dir:
        run_test(config, ckpt_dir)


# 모델 학습 및 테스트를 위한 메인 함수
def main(config):
    """
    checkpoint 경로가 존재할 경우 테스트를 수행하고, 아닐 경우 모델 학습 후 테스트를 수행합니다.

    Args:
        config: 모델 및 실험 설정이 포함된 configuration 객체
    """
    # checkpoint 또는 앙상블 경로가 존재할 경우 테스트를 수행하고, 프로그램 종료
    if config.experiment.checkpoint_path:
        run_test(config, config.experiment.checkpoint_path)
        return  # Exit the program after test and saving the csv output
    elif config.experiment.ensemble_path:
        predictor = EnsemblePredictor(config)
        predictor.run()
        return # Exit the program after test and saving the csv output
    else:
        pass
    # CUDA 가 없을 경우에 대한 처리 (CPU 학습은 불필요해서 넣었습니다)
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA not available. This program requires a CUDA-enabled NVIDIA GPU."
        )

    # checkpoint 경로가 없을 경우 최적의 모델을 학습 후 테스트를 수행합니다
    tune_train_and_test(config)


# 메인 함수를 호출하기 위한 장치
if __name__ == "__main__":
    # 커맨드라인 인자를 파싱
    parser = argparse.ArgumentParser(
        description="Model training and hyperparameter tuning."
    )
    parser.add_argument("--model-name", type=str, help="Name of the model to use.")
    parser.add_argument("--num-gpus", type=int, help="Name of the model to use.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Perform a small trial to test the setup.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to use pretrained model or not",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Perform the distributed data parallel. Only use when you have multiple gpus.",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, help="Path to the checkpoint to load and test."
    )
    parser.add_argument(
        "--ensemble-path", type=str, help="Path to the ensemble checkpoints to load and test."
    )
    parser.add_argument(
        "--ensemble-method", type=str, help="Type of ensemble method to apply"
    )
    args = parser.parse_args()

    # 커맨드라인 인자를 파싱하고 configuration 객체를 인자에 맞게 업데이트
    ConfigClass = get_config(args.model_name)
    config = ConfigClass()
    config.update_from_args(args)

    # 소규모 테스트가 요청된 경우 적절한 테스트 설정으로 초기화합니다.
    if args.smoke_test:
        config.experiment.max_epochs = 1
        config.experiment.num_samples = 1
        config.experiment.num_workers = 1
        config.training.num_gpus = 1
        config.experiment.grace_period = 1

    main(config)
