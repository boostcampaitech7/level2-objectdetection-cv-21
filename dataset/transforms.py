import numpy as np
import torch
from torchvision import transforms

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import create_transform
from torchvision.transforms import TrivialAugmentWide


class TransformSelector:
    """
    이미지 변환 라이브러리를 선택하기 위한 클래스.
    """

    def __init__(self, transform_type: str, input_size: int = 224, **kwargs):

        # 지원하는 변환 라이브러리인지 확인
        if transform_type in ["torchvision", "albumentations", "autoaugment", "trivial"]:
            self.transform_type = transform_type
        else:
            raise ValueError("Unknown transformation library specified.")
        self.input_size = input_size
        if 'aa' in kwargs and kwargs['aa']:
            self.aa = kwargs['aa']
    def get_transforms(self, is_train: bool):

        # 선택된 라이브러리에 따라 적절한 변환 객체를 생성
        if self.transform_type == "torchvision":
            transform = TorchvisionTransform(is_train=is_train, input_size=self.input_size)

        elif self.transform_type == "albumentations":
            transform = AlbumentationsTransform(is_train=is_train, input_size=self.input_size)

        elif self.transform_type == "autoaugment":
            transform = AutoAugmentTransform(is_train=is_train, input_size=self.input_size, aa=self.aa)
            
        elif self.transform_type == "trivial":
            transform = TrivialTransform(is_train=is_train, input_size=self.input_size)
            
        else:
            raise ValueError("Transform is not properly selected")
        return transform


class TorchvisionTransform:
    def __init__(self, is_train: bool = True, input_size: int = 224):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 정규화
        common_transforms = [
            transforms.Resize((input_size, input_size)),  # 이미지를 224x224 크기로 리사이즈
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 정규화
        ]

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 색상 조정 추가
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(
                        p=0.5
                    ),  # 50% 확률로 이미지를 수평 뒤집기
                    transforms.RandomRotation(15),  # 최대 15도 회전
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2
                    ),  # 밝기 및 대비 조정
                ]
                + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환

        transformed = self.transform(image)  # 설정된 변환을 적용

        return transformed  # 변환된 이미지 반환


class AlbumentationsTransform:
    def __init__(self, is_train: bool = True, input_size: int = 224):
        # 공통 변환 설정: 이미지 리사이즈, 정규화, 텐서 변환
        common_transforms = [
            A.Resize(input_size, input_size),  # 이미지를 224x224 크기로 리사이즈
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 정규화
            ToTensorV2(),  # albumentations에서 제공하는 PyTorch 텐서 변환
        ]

        if is_train:
            # 훈련용 변환: 랜덤 수평 뒤집기, 랜덤 회전, 랜덤 밝기 및 대비 조정 추가
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),  # 50% 확률로 이미지를 수평 뒤집기
                    A.Rotate(limit=15),  # 최대 15도 회전
                    A.RandomBrightnessContrast(p=0.2),  # 밝기 및 대비 무작위 조정
                ]
                + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = A.Compose(common_transforms)

    def __call__(self, image) -> torch.Tensor:
        # 이미지가 NumPy 배열인지 확인
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")

        # 이미지에 변환 적용 및 결과 반환
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용

        return transformed["image"]  # 변환된 이미지의 텐서를 반환


class AutoAugmentTransform:
    def __init__(self, is_train: bool = True, input_size: int = 224, aa: str = None):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 정규화
        common_transforms = [
            transforms.Resize((input_size, input_size)),  # 이미지를 224x224 크기로 리사이즈
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 정규화
        ]

        if is_train:
            # 훈련용 변환: AutoAugment 적용
            self.transform = create_transform(
                input_size=input_size,
                is_training=True,
                auto_augment=aa,
                interpolation='bicubic'
                )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환

        transformed = self.transform(image)  # 설정된 변환을 적용

        return transformed  # 변환된 이미지 반환


class TrivialTransform:
    def __init__(self, is_train: bool = True, input_size: int = 224):
        # 공통 변환 설정: 이미지 리사이즈, 텐서 변환, 정규화
        common_transforms = [
            transforms.Resize((input_size, input_size)),  # 이미지를 224x224 크기로 리사이즈
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # 정규화
        ]

        if is_train:
            # 훈련 데이터에 적용할 추가적인 변환 (TrivialAugmentWide 포함)
            self.transform = transforms.Compose([
                TrivialAugmentWide(num_magnitude_bins=31),
                ]
                + common_transforms
            )
        else:
            # 검증/테스트용 변환: 공통 변환만 적용
            self.transform = transforms.Compose(common_transforms)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(image)  # numpy 배열을 PIL 이미지로 변환

        transformed = self.transform(image)  # 설정된 변환을 적용

        return transformed  # 변환된 이미지 반환
