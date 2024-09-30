# 필요한 패키지 임포트
from typing import Optional

import torch
import torch.nn as nn
import timm

# from torchvision import models


class ResNet18(nn.Module):
    """
    추가 레이어나 사용자 정의 구현이 가능한 ResNet18 모델.

    인자:
        num_classes (int, optional): 출력 클래스 수. 기본값은 None이다.
        pretrained (bool, optional): 사전 학습된 가중치를 사용할지 여부. 기본값은 False이다.
    """

    def __init__(
        self, num_classes: Optional[int] = 500, pretrained: bool = False, **kwargs
    ):
        super(ResNet18, self).__init__()

        # 사전 학습된 ResNet18 모델 로드
        self.model = timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=num_classes, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ResNet18 모델을 통한 포워드 패스.

        인자:
            x (torch.Tensor): 입력 텐서.

        반환:
            torch.Tensor: ResNet18 모델의 출력.
        """
        return self.model(x)
