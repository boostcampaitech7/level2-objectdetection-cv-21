# 필요한 import 수행
import lightning as pl
import torch

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler_v2
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .model_factory import create_model, create_stable_diffusion_model
from config import ModelConfig



# 라이트닝 모듈 정의
class DetectionModule(pl.LightningModule):
    def __init__(self, hparams, config: ModelConfig = None):
        """
        라이트닝 모듈 초기화.

        Args:
            hparams (dict): 모델 하이퍼 파라미터.
        """
        super().__init__()

        model_hparams = vars(config) if config else {}
        hparams = {**hparams, **model_hparams}
        self.save_hyperparameters(hparams)
        # 객체 탐지 모델 생성 (Stable Diffusion으로 데이터 증강)
        self.model = create_model(**model_hparams)

        # Stable Diffusion을 사용한 증강 모델도 추가적으로 생성
        self.stable_diffusion = create_stable_diffusion_model(device="cuda", fp16=True, sd_version="1.5")

        self.map = MeanAveragePrecision(
            box_format="xyxy",
            iou_thresholds=[0.5],
            class_metrics=True
        )

        # self.mixup_fn = Mixup(
        #     mixup_alpha=self.hparams.mixup, cutmix_alpha=self.hparams.cutmix,
        #     prob=self.hparams.mixup_prob, switch_prob=self.hparams.mixup_switch_prob,
        #     label_smoothing=self.hparams.smoothing, num_classes=500)

    def forward(self, images, targets=None):
        """
        객체 탐지 모델의 forward 함수. 
        """
        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)

    def training_step(self, train_batch, batch_idx):
        """
        훈련 스텝 정의. Stable Diffusion으로 생성된 이미지 포함.
        """
        images, targets, _ = train_batch
        
        # 객체 탐지 모델 예측 및 손실 계산
        outputs = self.forward(images, targets)
        losses = sum(loss for loss in outputs.values())
        
        # 학습 손실 기록
        self.log("train_loss", losses, sync_dist=True)
        return {"loss": losses}

    def validation_step(self, val_batch, batch_idx):
        """
        검증 스텝 정의. mAP로 성능 평가.
        """
        images, targets, _ = val_batch
        
        # 객체 탐지 모델 예측
        outputs = self.forward(images, targets)
        losses = sum(loss for loss in outputs.values())
        
        # mAP 업데이트
        self.map.update(outputs, targets)

        # 검증 손실 기록
        self.log("val_loss", losses, sync_dist=True)
        return {"loss": losses}

    def on_validation_epoch_end(self, outputs):
        """
        각 에폭이 끝날 때 mAP 계산.
        """
        map_value = self.map.compute()
        self.log("map50", map_value["map_50"].item(), sync_dist=True)
        self.map.reset()

    def configure_optimizers(self):
        """
        최적화 함수 정의 (AdamW + 스케줄러).
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # 스케줄러 설정
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=8, gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def generate_synthetic_data(self, prompts, resolutions=[(512, 512)]):
        """
        Stable Diffusion을 사용해 데이터 증강.
        """
        # Stable Diffusion을 사용해 이미지를 생성
        synthetic_images = self.stable_diffusion.generate_synthetic_data(prompts=prompts, resolutions=resolutions)
        return synthetic_images