# 필요한 import 수행
import lightning as pl
import torch

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler_v2
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .model_factory import create_model
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
        self.model = create_model(**model_hparams)

        self.map = MeanAveragePrecision(
            box_format="xyxy",
            iou_thresholds=[0.5],
            class_metrics=True
        )

        # self.mixup_fn = Mixup(
        #     mixup_alpha=self.hparams.mixup, cutmix_alpha=self.hparams.cutmix,
        #     prob=self.hparams.mixup_prob, switch_prob=self.hparams.mixup_switch_prob,
        #     label_smoothing=self.hparams.smoothing, num_classes=500)

    def forward(self, prompts):
        """
        Forward method for text-to-image generation using Stable Diffusion.
        """
        # Generate images from text prompts
        images = self.model.prompt_to_img(prompts)
        return images

    def training_step(self, train_batch, batch_idx):
        """
        Training step for fine-tuning Stable Diffusion.
        """
        prompts, target_images, _ = train_batch  # Assuming dataset returns (prompts, images, labels)
        
        # Get text embeddings
        text_embeddings = self.model.get_text_embeds(prompts)
        
        # Perform training step (you can add noise, latent manipulation, etc.)
        pred_rgb = target_images.to(self.device)
        loss = self.model.train_step(text_embeddings, pred_rgb)
        
        self.log("train_loss", loss, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Validation step for fine-tuned Stable Diffusion.
        """
        prompts, target_images, _ = val_batch
        text_embeddings = self.model.get_text_embeds(prompts)
        pred_rgb = target_images.to(self.device)

        # Calculate validation loss
        loss = self.model.train_step(text_embeddings, pred_rgb)
        self.log("val_loss", loss, sync_dist=True)

        # Optionally, update Mean Average Precision (MAP) for comparison
        # Since diffusion-based models don’t natively support bounding boxes, you might need to adapt MAP for image comparison
        self.map.update(pred_rgb, target_images)
        return {"loss": loss}


    def on_validation_epoch_end(self, val_step_outputs):
        """
        Called at the end of the validation epoch.
        """
        map_value = self.map.compute()
        self.log("map50", map_value["map_50"].item(), sync_dist=True)
        self.map.reset()
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler for Stable Diffusion fine-tuning.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Use a cosine or step scheduler
        lr_scheduler, _ = create_scheduler_v2(
            optimizer,
            sched=self.hparams.sched,
            num_epochs=self.trainer.max_epochs,
            warmup_epochs=self.hparams.warmup_epochs,
            warmup_lr=self.hparams.warmup_lr,
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, lr_scheduler, metric):
        """
        Update learning rate scheduler after each epoch.
        """
        lr_scheduler.step(epoch=self.current_epoch)  # timm's scheduler needs the epoch value

    def generate_images(self, prompts, resolutions=[(128, 128), (512, 512)]):
        """
        Use Stable Diffusion to generate synthetic data.
        """
        # Generate synthetic images with Stable Diffusion
        self.model.generate_synthetic_data(prompts=prompts, resolutions=resolutions)