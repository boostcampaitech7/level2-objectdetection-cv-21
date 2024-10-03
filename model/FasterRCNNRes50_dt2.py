import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNNRes50_dt2(nn.Module):
    """
    A custom object detection model based on Faster R-CNN with ResNet-50 backbone.

    This model uses a pre-trained ResNet-50 model with Feature Pyramid Network (FPN)
    and modifies the classifier head for custom number of classes.

    Attributes:
        device (str): Device to run the model on ('cuda' or 'cpu')
        model (torchvision.models): The underlying Faster R-CNN model
    """

    def __init__(self, num_classes=11, device='cuda', **kwargs):
        super(FasterRCNNRes50_dt2, self).__init__()
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=kwargs["pretrained"])

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print(self.model)
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
