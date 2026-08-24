"""Configuration Loader Utilities

Handles loading, merging, and managing YAML configs with CLI override support.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import yaml
import copy

from .model_config import HRMConfig, PRESET_CONFIGS as MODEL_PRESETS
from .data_config import DataConfig
from .train_config import TrainConfig, PRESET_CONFIGS as TRAIN_PRESETS


class ConfigManager:
    """Central config manager for loading and merging configurations."""

    def __init__(self, config_root: Optional[str] = None):
        self.config_root = Path(config_root) if config_root else Path(__file__).parent.parent / "configs"
        self.model_dir = self.config_root / "model"
        self.data_dir = self.config_root / "data"
        self.train_dir = self.config_root / "train"

    def load_model_config(self, name_or_path: str) -> HRMConfig:
        """Load model config from preset name or YAML path."""
        if name_or_path in MODEL_PRESETS:
            return MODEL_PRESETS[name_or_path]()

        path = Path(name_or_path)
        if not path.is_absolute():
            path = self.model_dir / path
        if path.suffix not in ['.yaml', '.yml']:
            path = path.with_suffix('.yaml')
        return HRMConfig.from_yaml(str(path))

    def load_data_config(self, name_or_path: str) -> DataConfig:
        """Load data config from preset name or YAML path."""
        # Check registry first
        from .data_config import DatasetRegistry
        if (cfg := DatasetRegistry.get(name_or_path)) is not None:
            return cfg

        path = Path(name_or_path)
        if not path.is_absolute():
            path = self.data_dir / path
        if path.suffix not in ['.yaml', '.yml']:
            path = path.with_suffix('.yaml')
        return DataConfig.from_yaml(str(path))

    def load_train_config(self, name_or_path: str) -> TrainConfig:
        """Load train config from preset name or YAML path."""
        if name_or_path in TRAIN_PRESETS:
            cfg = TRAIN_PRESETS[name_or_path]()
            # Handle callable presets like multi_gpu
            if callable(cfg):
                cfg = cfg()
            return cfg

        path = Path(name_or_path)
        if not path.is_absolute():
            path = self.train_dir / path
        if path.suffix not in ['.yaml', '.yml']:
            path = path.with_suffix('.yaml')
        return TrainConfig.from_yaml(str(path))

    def list_available(self) -> Dict[str, List[str]]:
        """List all available config files."""
        return {
            "model": [f.stem for f in self.model_dir.glob("*.yaml")],
            "data": [f.stem for f in self.data_dir.glob("*.yaml")],
            "train": [f.stem for f in self.train_dir.glob("*.yaml")],
            "model_presets": list(MODEL_PRESETS.keys()),
            "train_presets": list(TRAIN_PRESETS.keys()),
        }


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: Optional[str] = None,
    model_config: Optional[str] = None,
    data_config: Optional[str] = None,
    train_config: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load and merge configurations with priority:
    1. Low: Default presets
    2. Mid: Combined YAML file (config_path)
    3. High: Individual YAML files (model_config, data_config, train_config)
    4. Highest: CLI overrides (cli_overrides)
    """
    manager = ConfigManager()

    # Start with empty
    merged = {"model": {}, "data": {}, "train": {}}

    # 1. Load combined config if provided (mid priority)
    if config_path:
        with open(config_path, 'r') as f:
            combined = yaml.safe_load(f)
        if combined:
            merged = deep_merge(merged, combined)

    # 2. Load individual configs (high priority)
    if model_config:
        merged["model"] = deep_merge(merged.get("model", {}),
                                     manager.load_model_config(model_config).to_dict())
    if data_config:
        merged["data"] = deep_merge(merged.get("data", {}),
                                    manager.load_data_config(data_config).to_dict())
    if train_config:
        merged["train"] = deep_merge(merged.get("train", {}),
                                     manager.load_train_config(train_config).to_dict())

    # 3. Apply CLI overrides (highest priority)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if key in ["model", "data", "train"]:
                merged[key] = deep_merge(merged.get(key, {}), value)
            else:
                # Flat override - try to infer section
                for section in ["model", "data", "train"]:
                    if key in merged.get(section, {}):
                        merged[section][key] = value
                        break

    return merged


def build_configs(
    config_path: Optional[str] = None,
    model_config: Optional[str] = None,
    data_config: Optional[str] = None,
    train_config: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[HRMConfig, DataConfig, TrainConfig]:
    """
    Build fully validated config objects from merged configuration.

    Returns:
        Tuple of (model_config_obj, data_config_obj, train_config_obj)
    """
    merged = load_config(config_path, model_config, data_config, train_config, cli_overrides)

    model_cfg = HRMConfig.from_dict(merged.get("model", {}))
    data_cfg = DataConfig.from_dict(merged.get("data", {}))
    train_cfg = TrainConfig.from_dict(merged.get("train", {}))

    return model_cfg, data_cfg, train_cfg


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple config dictionaries."""
    if not configs:
        return {}
    result = configs[0]
    for cfg in configs[1:]:
        result = deep_merge(result, cfg)
    return result


def save_config(config: Dict[str, Any], path: str):
    """Save merged config to YAML."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def config_to_argparse_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert config dict to flat argparse-compatible dict."""
    flat = {}
    for section, values in config.items():
        for key, value in values.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{section}_{key}_{sub_key}"] = sub_value
            else:
                flat[f"{section}_{key}"] = value
    return flat


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides (prefix: UCV_HRM_)."""
    import os
    result = copy.deepcopy(config)

    for key, value in os.environ.items():
        if key.startswith("UCV_HRM_"):
            config_key = key[8:].lower()  # Remove prefix
            # Parse type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string

            # Set in appropriate section
            for section in result:
                if config_key in result[section]:
                    result[section][config_key] = value
                    break

    return result