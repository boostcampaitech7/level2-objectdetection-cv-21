# dataset/dataloader.py
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

from .dataset import CustomDataset
from .transforms import TransformSelector


def get_dataloaders(config, batch_size=32):
    """
    Returns train and validation data loaders.

    Args:
        data_path (str): Path to the dataset. Defaults to '/home/data/'.
        batch_size (int): Batch size for each data loader. Defaults to 32.
        num_workers (int): Number of worker threads for each data loader. Defaults to 1.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders.
    """
    info_df = pd.read_csv(os.path.join(config.dataset.data_path, 'train.csv'))
    data_path = os.path.join(config.dataset.data_path, 'train')

    train_df, val_df = train_test_split(
        info_df, test_size=0.2, stratify=info_df["target"]
    )

    transform_selector = TransformSelector(
        input_size=config.dataset.input_size, 
        transform_type=config.dataset.transform_type,
        aa=(config.dataset.aa if config.dataset.transform_type=="autoaugment" else None)
    )

    train_transform = transform_selector.get_transforms(is_train=True)
    train_dataset = CustomDataset(data_path, train_df, transform=train_transform)

    # Create val dataset with validation transforms
    val_transform = transform_selector.get_transforms(is_train=False)
    val_dataset = CustomDataset(data_path, val_df, transform=val_transform)

    # Create samplers for the train and validation sets
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)

    # Use DataCollator to create batches for both datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # sampler=train_sampler,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        #   sampler=val_sampler,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def get_test_loader(config):
    """
    Returns a test data loader.

    Args:
        data_path (str): Path to the dataset. Defaults to '/home/data/'.
        batch_size (int): Batch size for the data loader. Defaults to 32.
        num_workers (int): Number of worker threads for the data loader. Defaults to 1.

    Returns:
        DataLoader: Test data loader.
    """
    transform_selector = TransformSelector(
        input_size=config.dataset.input_size, 
        transform_type=config.dataset.transform_type,
        aa=(config.dataset.aa if config.dataset.transform_type=="autoaugment" else None)
        )
    test_df = pd.read_csv(os.path.join(config.dataset.data_path, "test.csv"))
    data_path = os.path.join(config.dataset.data_path, "test")
    transform = transform_selector.get_transforms(is_train=False)
    test_dataset = CustomDataset(
        data_path, test_df, transform=transform, is_inference=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, num_workers=6, pin_memory=True
    )
    return test_loader
