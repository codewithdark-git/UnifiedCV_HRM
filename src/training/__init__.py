"""
Training Module

Complete training infrastructure: Trainer, losses, metrics, callbacks, optimizers, schedulers.
"""

from .trainer import Trainer, TrainerConfig
from .losses import (
    MultiTaskLoss,
    ClassificationLoss,
    SegmentationLoss,
    DetectionLoss,
    HaltingLoss,
    get_loss_fn,
)
from .metrics import (
    Accuracy,
    TopKAccuracy,
    IoU,
    DiceScore,
    COCOmAP,
    get_metrics,
)
from .callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    WandbCallback,
    TensorBoardCallback,
)
from .optim import create_optimizer, create_scheduler

__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    # Losses
    "MultiTaskLoss",
    "ClassificationLoss",
    "SegmentationLoss",
    "DetectionLoss",
    "HaltingLoss",
    "get_loss_fn",
    # Metrics
    "Accuracy",
    "TopKAccuracy",
    "IoU",
    "DiceScore",
    "COCOmAP",
    "get_metrics",
    # Callbacks
    "Callback",
    "ModelCheckpoint",
    "EarlyStopping",
    "LearningRateMonitor",
    "WandbCallback",
    "TensorBoardCallback",
    # Optim
    "create_optimizer",
    "create_scheduler",
]