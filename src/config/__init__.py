"""
Configuration Package

Centralized configuration system for model, data, and training.
All configs use YAML + dataclass validation with registry support.
"""

from .model_config import HRMConfig, AttentionType, FFNType, TaskType
from .data_config import DataConfig, DataSource, DatasetRegistry
from .train_config import TrainConfig, OptimizerType, SchedulerType
from .loader import load_config, merge_configs, save_config, ConfigManager

__all__ = [
    "HRMConfig",
    "AttentionType",
    "FFNType",
    "TaskType",
    "DataConfig",
    "DataSource",
    "DatasetRegistry",
    "TrainConfig",
    "OptimizerType",
    "SchedulerType",
    "load_config",
    "merge_configs",
    "save_config",
    "ConfigManager",
]