import os
from pycocotools.coco import COCO
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import CocoDetectionDataset


def get_dataloaders(config):
    """
    Returns train and validation data loaders for COCO format detection dataset.

    Args:
        config: Configuration object containing dataset parameters

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders.
    """
    # Load COCO format annotations
    ann_file = os.path.join(config.dataset.data_path, 'train.json')
    coco = COCO(ann_file)

    # Get all image ids and split into train/val
    image_ids = list(coco.imgs.keys())

    # Stratified split based on the categories present in each image
    image_categories = []
    for img_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        categories = set([ann['category_id'] for ann in anns])
        image_categories.append(list(categories)[0] if categories else 0)

    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=0.2,
        stratify=image_categories,
        random_state=42
    )

    # Create datasets
    train_dataset = CocoDetectionDataset(
        data_path=config.dataset.data_path,
        ann_file=ann_file,
        image_ids=train_ids,
        augment=True
    )

    val_dataset = CocoDetectionDataset(
        data_path=config.dataset.data_path,
        ann_file=ann_file,
        image_ids=val_ids,
        augment=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )

    return train_loader, val_loader


def get_test_loader(config):
    """
    Returns a test data loader for COCO format detection dataset.

    Args:
        config: Configuration object containing dataset parameters

    Returns:
        DataLoader: Test data loader.
    """
    ann_file = os.path.join(config.dataset.data_path, 'test.json')

    test_dataset = CocoDetectionDataset(
        data_path=config.dataset.data_path,
        ann_file=ann_file,
        is_inference=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        num_workers=3,
        pin_memory=True,
        collate_fn=detection_collate_fn
    )

    return test_loader


def detection_collate_fn(batch):
    """
    Custom collate function for detection data.
    Handles variable number of objects in images.
    """
    images = []
    targets = []
    img_ids = []

    for sample in batch:
        images.append(sample[0])
        targets.append(sample[1])
        if len(sample) > 2:  # For inference mode
            img_ids.append(sample[2])

    images = torch.stack(images, 0)

    if img_ids:
        return images, targets, img_ids

    return images, targets
