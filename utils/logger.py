import wandb


class WandbLogger:
    def __init__(self, project_name, config=None):
        """
        WandbLogger를 초기화합니다.

        Args:
            project_name (str): wandb 프로젝트 이름
            config (dict, optional): 학습 하이퍼파라미터나 설정을 포함하는 딕셔너리
        """
        self.project_name = project_name
        self.config = config

        # wandb 실행을 초기화합니다
        self.run = wandb.init(project=self.project_name, config=self.config)

    def log_metrics(self, metrics, step=None):
        """
        메트릭을 wandb에 기록합니다.

        Args:
            metrics (dict): 메트릭 딕셔너리 (예: {"accuracy": 0.9, "loss": 0.1})
            step (int, optional): 현재 스텝 또는 에포크
        """
        if step:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_image(self, image, caption=""):
        """
        이미지를 wandb에 기록합니다.

        Args:
            image (numpy array): 기록할 이미지 데이터
            caption (str, optional): 이미지 설명
        """
        wandb.log({"image": wandb.Image(image, caption=caption)})

    def finish(self):
        """wandb 실행을 종료합니다."""
        self.run.finish()


"""
# 테스트 예시 코드 (실제 사용할 때는 아래 부분을 다른 파일에서 실행)
if __name__ == "__main__":
    # 하이퍼파라미터 설정 예시
    config = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
    }

    # WandbLogger 인스턴스 생성
    logger = WandbLogger(project_name="my-awesome-project", config=config)

    # 에포크마다 메트릭을 기록하는 예시
    for epoch in range(1, config['epochs'] + 1):
        train_loss = 0.05 * epoch  # 임의 값
        val_loss = 0.04 * epoch  # 임의 값
        accuracy = 0.1 * epoch  # 임의 값

        # 메트릭 기록
        logger.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy
        }, step=epoch)

    # wandb 실행 종료
    logger.finish()
"""
