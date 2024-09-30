# 필요한 패키지 임포트
from typing import Optional

import torch
import torch.nn as nn
import timm

# from torchvision import models


class CoAtNet(nn.Module):
    """
    추가 레이어나 사용자 정의 구현이 가능한 CoAtNet 모델.

    인자:
        num_classes (int, optional): 출력 클래스 수. 기본값은 None이다.
        pretrained (bool, optional): 사전 학습된 가중치를 사용할지 여부. 기본값은 False이다.
    """

    def __init__(
        self, num_classes: Optional[int] = 500, pretrained: bool = False, **kwargs
    ):
        super(CoAtNet, self).__init__()

        # 사전 학습된 CoAtNet 모델 로드
        self.model = timm.create_model(
            "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
            **kwargs
        )
        self.set_attn_only_finetune()

        # 설정 확인
        # for name, param in self.model.named_parameters():
        #    print(f"{name}: requires_grad = {param.requires_grad}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CoAtNet 모델을 통한 포워드 패스.

        인자:
            x (torch.Tensor): 입력 텐서.

        반환:
            torch.Tensor: CoAtNet 모델의 출력.
        """
        return self.model(x)

    def set_attn_only_finetune(self):
        """
        모델의 attention 파라미터만 학습하도록 설정.

        인자:
            model (nn.Module): 입력 모델
        """
        # 전체 파라미터 학습 비활성화
        for name_p, p in self.model.named_parameters():
            p.requires_grad = False

        # stem, stages0과 stages1의 파라미터 학습 활성화
        for name, param in self.model.named_parameters():
            if 'stem' in name or 'stages.0.' in name or 'stages.1.' in name:
                param.requires_grad = True

        # Attention 파라미터와 head.fc 파라미터만 학습 활성화
        for name, param in self.model.named_parameters():
            if '.attn.' in name or 'head.fc' in name:
                param.requires_grad = True