from dataclasses import dataclass

@dataclass
class DatasetConfig:
    # Data paths
    data_path: str = "/path/to/data_path"
    
    # Dataset parameters
    num_classes: int = 10
    
    # DataLoader parameters
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()