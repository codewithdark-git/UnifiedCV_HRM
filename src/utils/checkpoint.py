"""
Checkpoint Utilities

Save/load training checkpoints, model weights, and HF conversion.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Union, Optional, Dict, Any
from pathlib import Path


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    epoch: int = 0,
    step: int = 0,
    best_metric: float = float("inf"),
    config: Optional[Dict] = None,
    model_config: Optional[Dict] = None,
    data_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save training checkpoint.

    Args:
        path: Save path
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        scaler: Gradient scaler state
        epoch: Current epoch
        step: Current global step
        best_metric: Best validation metric
        config: Training config
        model_config: Model config
        data_config: Data config
        metadata: Additional metadata
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": config,
        "model_config": model_config,
        "data_config": data_config,
        "metadata": metadata or {},
    }

    torch.save(state, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load state into
        optimizer: Optimizer to restore
        scheduler: Scheduler to restore
        scaler: GradScaler to restore
        device: Device to map tensors to
        strict: Whether to enforce matching keys

    Returns:
        Checkpoint dict with epoch, step, best_metric, etc.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)

    # Load model
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load scaler
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_metric": checkpoint.get("best_metric", float("inf")),
        "config": checkpoint.get("config"),
        "model_config": checkpoint.get("model_config"),
        "data_config": checkpoint.get("data_config"),
        "metadata": checkpoint.get("metadata", {}),
    }


def save_model(
    path: Union[str, Path],
    model: nn.Module,
    config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """Save model only (no optimizer/scheduler)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metadata": metadata or {},
    }
    torch.save(state, path)


def load_model(
    path: Union[str, Path],
    model: nn.Module,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model only."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)

    return {
        "config": checkpoint.get("config"),
        "metadata": checkpoint.get("metadata", {}),
    }


def strip_ddp(state_dict: OrderedDict) -> OrderedDict:
    """Remove 'module.' prefix from DDP state dict."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_partial(
    model: nn.Module,
    path: Union[str, Path],
    prefix_to_strip: str = "",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Load partial checkpoint (e.g., only backbone from HF checkpoint).

    Args:
        model: Target model
        path: Checkpoint path
        prefix_to_strip: Strip this prefix from keys (e.g., "backbone.")
        device: Device
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if prefix_to_strip:
        state_dict = {k[len(prefix_to_strip):]: v for k, v in state_dict.items() if k.startswith(prefix_to_strip)}

    # Filter out keys not in model or shape mismatch
    model_dict = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)

    print(f"Loaded {len(filtered)}/{len(state_dict)} keys from {path}")
    return model


def convert_to_hf_checkpoint(
    path: Union[str, Path],
    output_dir: Union[str, Path],
    model_class,
    config,
    tokenizer=None,
) -> None:
    """
    Convert local checkpoint to HF format.

    Args:
        path: Local checkpoint path
        output_dir: Output directory for HF model
        model_class: HF model class (e.g., UnifiedCVHRM)
        config: HF config object
        tokenizer: Optional tokenizer
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load local checkpoint
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Create HF model and load weights
    model = model_class(config)
    model.load_state_dict(state_dict, strict=False)

    # Save in HF format
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    if tokenizer:
        tokenizer.save_pretrained(output_dir)

    print(f"HF model saved to {output_dir}")


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """Find latest checkpoint in directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest