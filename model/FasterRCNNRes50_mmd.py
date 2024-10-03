import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from mmcv import Config
from mmdet.models import build_detector

class FasterRCNNRes50_mmd(nn.Module):
    """
    A custom object detection model based on Faster R-CNN with ResNet-50 backbone.

    This model uses a pre-trained ResNet-50 model with Feature Pyramid Network (FPN)
    and modifies the classifier head for custom number of classes.

    Attributes:
        device (str): Device to run the model on ('cuda' or 'cpu')
        model (torchvision.models): The underlying Faster R-CNN model
    """

    def __init__(self, num_classes=11, device='cuda', **kwargs):
        super(FasterRCNNRes50_mmd, self).__init__()
        self.device = device
        # 모델 build 및 pretrained network 불러오기
        cfg = Config.fromfile('/data/ephemeral/home/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
        root='/data/ephemeral/home/dataset/'
        classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
        # dataset config 수정
        cfg.data.train.classes = classes
        cfg.data.train.img_prefix = root
        cfg.data.train.ann_file = root + 'train.json' # train json 정보
        cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

        cfg.data.test.classes = classes
        cfg.data.test.img_prefix = root
        cfg.data.test.ann_file = root + 'test.json' # test json 정보
        cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

        cfg.data.samples_per_gpu = 4

        cfg.seed = 2022
        cfg.gpu_ids = [0]
        cfg.work_dir = '/data/ephemeral/home/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_trash'

        cfg.model.roi_head.bbox_head.num_classes = 10

        cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
        cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
        



        model = build_detector(cfg.model)
        model.init_weights()

        print(model)
        exit()

    def forward(self, images, targets=None):
        """
        Forward pass of the model.

        Args:
            images (List[torch.Tensor]): List of input images
            targets (List[Dict[str, torch.Tensor]], optional): List of target dictionaries
                containing boxes and labels for each image

        Returns:
            Model predictions or losses depending on whether targets are provided
        """
        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)
