from ray import tune

from .config import Config


class DeiT3LargeConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = "DeiT3Large"  # Override the baseline model name
        self.training.batch_size = 32
        self.training.lr = tune.loguniform(1e-5, 2e-4)
        self.training.weight_decay = tune.loguniform(0.0001, 0.1)

        self.dataset.transform_type = 'autoaugment'

        self.experiment.num_workers = 3
        self.experiment.num_samples = 10
        self.experiment.max_epochs = 30