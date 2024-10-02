import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


class CocoDetectionDataset(Dataset):
    """
    Dataset class for loading COCO format object detection datasets
    """
    # data_path ex : './home/dataset' 
    def __init__(self, data_path, ann_file, image_ids=None, is_inference=False):
        """
        Args:
            data_path (str): Root directory of the dataset
            ann_file (str): Path to annotation file (train.json or test.json)
            image_ids (list): Optional list of image ids to use
            is_inference (bool): Whether the dataset is for inference
        """
        self.data_path = data_path
        self.is_inference = is_inference
        
        # Initialize COCO api
        self.coco = COCO(ann_file)
        
        # Set image ids
        self.image_ids = image_ids if image_ids is not None else list(self.coco.imgs.keys())
        
        # Determine if we're using train or test images
        self.image_dir = 'train' if 'train.json' in ann_file else 'test'
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image id and load image
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image from correct directory
        img_path = os.path.join(self.data_path, self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Convert PIL image to tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Verify image size
        assert image.shape[1:] == (1024, 1024), f"Image {img_path} size is not 1024x1024"
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Prepare target
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            bbox = ann['bbox']
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            bbox = [
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ]
            boxes.append(bbox)
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])
        
        # Convert to tensor
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        
        if self.is_inference:
            return image, target, img_id
        
        return image, target