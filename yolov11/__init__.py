# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict_ori import DetectionPredictor
from .train_ori import DetectionTrainer
from .val_ori import DetectionValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator"
