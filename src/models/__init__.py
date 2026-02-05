"""Model training modules for YOLO pill detection project."""

from .base_trainer import BaseTrainer
from .yolo_trainer import YOLOTrainer

__all__ = [
    'BaseTrainer',
    'YOLOTrainer',
]
