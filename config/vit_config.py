from ray import tune

from .config import Config

class ViTConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = "ViT"  # Override the baseline model name

        self.model.drop_path_rate = 0.2

        self.training.batch_size = 64
        self.training.lr = tune.loguniform(1e-5, 2e-4)
        self.training.weight_decay = tune.loguniform(0.0001, 0.1)

        self.dataset.transform_type = 'autoaugment'

        self.experiment.num_workers = 3
        self.experiment.num_samples = 20
        self.experiment.max_epochs = 50