"""
Logging Utilities

Centralized logging setup with Rich and TensorBoard support.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

from rich.logging import RichHandler
from rich.console import Console


# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logging(
    level: str = "INFO",
    verbose: bool = False,
    log_file: Optional[str] = None,
    rich: bool = True,
) -> logging.Logger:
    """
    Configure logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        verbose: Enable verbose/debug output
        log_file: Optional log file path
        rich: Use Rich formatting (colors, tracebacks)

    Returns:
        Root logger
    """
    global _logger

    # Set level
    if verbose:
        level = "DEBUG"

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    # Console handler
    if rich:
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=verbose,
        )
    else:
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(log_level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    root_logger.addHandler(handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)

    # Reduce noise from libraries
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _logger = logging.getLogger("ihrm")
    return _logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    if _logger is None:
        setup_logging()
    return logging.getLogger(name)


class TensorBoardLogger:
    """Simple TensorBoard wrapper."""

    def __init__(self, log_dir: str):
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to {log_dir}")
        except ImportError:
            print("tensorboard not installed, skipping TensorBoard logging")

    def add_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, tag: str, values: dict, step: int):
        if self.writer:
            self.writer.add_scalars(tag, values, step)

    def add_histogram(self, tag: str, values, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def add_text(self, tag: str, text: str, step: int):
        if self.writer:
            self.writer.add_text(tag, text, step)

    def close(self):
        if self.writer:
            self.writer.close()


class WandbLogger:
    """Weights & Biases logger wrapper."""

    def __init__(
        self,
        project: str = "ihrm",
        name: Optional[str] = None,
        config: Optional[dict] = None,
        entity: Optional[str] = None,
    ):
        self.run = None
        try:
            import wandb
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                entity=entity,
            )
            print(f"W&B run: {self.run.name} ({self.run.url})")
        except ImportError:
            print("wandb not installed, skipping W&B logging")

    def log(self, metrics: dict, step: Optional[int] = None):
        if self.run:
            import wandb
            wandb.log(metrics, step=step)

    def watch(self, model, log: str = "all", log_freq: int = 100):
        if self.run:
            import wandb
            wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self):
        if self.run:
            import wandb
            wandb.finish()


def create_experiment_dir(base: str = "outputs", name: Optional[str] = None) -> Path:
    """Create timestamped experiment directory."""
    from datetime import datetime
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = name or f"exp_{timestamp}"
    exp_dir = base_path / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


class ProgressLogger:
    """Progress logging for training loops."""

    def __init__(self, logger: logging.Logger, log_freq: int = 10):
        self.logger = logger
        self.log_freq = log_freq
        self.step = 0
        self.epoch = 0

    def log_step(self, metrics: dict, step: int = None):
        self.step = step if step is not None else self.step + 1
        if self.step % self.log_freq == 0:
            msg = f"Step {self.step} | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            self.logger.info(msg)

    def log_epoch(self, metrics: dict, epoch: int = None):
        self.epoch = epoch if epoch is not None else self.epoch + 1
        msg = f"Epoch {self.epoch} | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self.logger.info(msg)


# ─────────────────────────────────────────────────────────────────
# Distributed Utilities (also exported from logging for convenience)
# ─────────────────────────────────────────────────────────────────

import os
import torch
import torch.distributed as dist
from typing import Optional, Any, List, Dict


def is_main_process() -> bool:
    """Check if current process is rank 0."""
    return get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get local rank (GPU index)."""
    if dist.is_available() and dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_distributed() -> bool:
    """Check if distributed training is active."""
    return dist.is_available() and dist.is_initialized()


def reduce_dict(data: Dict[str, torch.Tensor], op: str = "mean") -> Dict[str, torch.Tensor]:
    """
    Reduce dictionary of tensors across all processes.

    Args:
        data: Dict of tensors to reduce
        op: "mean" | "sum" | "min" | "max"

    Returns:
        Reduced dict (on all ranks if distributed)
    """
    if not is_distributed():
        return data

    world_size = get_world_size()
    if world_size == 1:
        return data

    with torch.no_grad():
        keys = sorted(data.keys())
        tensors = [data[k] for k in keys]
        stacked = torch.stack(tensors, dim=0)

        dist_op = getattr(dist.ReduceOp, op.upper())
        dist.all_reduce(stacked, op=dist_op)

        if op == "mean":
            stacked /= world_size

        return {k: v for k, v in zip(keys, stacked.unbind(0))}


def all_gather(data: Any) -> List[Any]:
    """Gather data from all processes."""
    if not is_distributed():
        return [data]

    world_size = get_world_size()
    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, data)
    return gather_list