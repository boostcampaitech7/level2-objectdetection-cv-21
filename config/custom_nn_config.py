from ray import tune

from .config import Config


class CustomNNConfig(Config):
    def __init__(self):
        super().__init__()
        self.model_name = "custom_nn"  # Override the baseline model name
        self.n_layers = tune.randint(
            2, 5
        )  # Add a new hyperparameter specific to CustomNN
        self.search_space = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "n_workers": self.n_workers,
            "lr_schedule": self.lr_schedule,
            "n_estimators": self.n_estimators,
            "n_layers": self.n_layers,  # Add the new hyperparameter to the search space
        }
