import os

import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, data_path, info_df, transform=None, is_inference: bool = False):
        """
        Initializes the CustomDataset class.

        Args:
        - data_path (str): Path to the dataset.
        - info_df (str): The index information dataframe.
        - transform (transforms.Compose): The transforms to be applied to the images (default: None).
        """

        self.data_path = data_path
        self.info_df = info_df
        self.is_inference = is_inference
        self.transform = transform
        self.image_paths = self.info_df["image_path"].tolist()
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.RandomRotation(15),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.CenterCrop(224),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        ]

        if not self.is_inference:
            self.labels = self.info_df[
                "target"
            ].tolist()  # Read image paths from test.csv

    def _load_image(self, image_path):
        image = cv2.imread(
            image_path, cv2.IMREAD_COLOR
        )  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # BGR 포맷을 RGB 포맷으로 변환합니다.
        return image

    def _apply_transform(self, image):
        if self.transform:
            image = self.transform(image)
        return image
    
    def _apply_transform_tta(self, image, tta_transform):
        image = Image.fromarray(image)
        if self.transform:
            image = tta_transform(image)
        return image

    def __getitem__(self, index):
        """
        Returns the image and label at the specified index.

        Args:
        - index (int): The index of the image and label.

        Returns:
        - image: The image at the specified index.
        - label: The label at the specified index (Not returned if in 'test' mode).
        """
        image_path = os.path.join(self.data_path, self.image_paths[index])
        image = self._load_image(image_path)

        if self.is_inference:
            tta_images = [self._apply_transform_tta(image, tta_transform) for tta_transform in self.tta_transforms]
            return tta_images
        else:
            image = self._apply_transform(image)
            label = self.labels[index]
            return image, label

    def __len__(self):
        return len(self.image_paths)
