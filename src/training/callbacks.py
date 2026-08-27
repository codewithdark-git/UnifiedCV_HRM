"""
Callbacks Module

Training callbacks for checkpointing, early stopping, logging, etc.
"""

import os
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path
from abc import ABC, abstractmethod


class Callback(ABC):
    """Base callback class."""

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, train_metrics: Dict, val_metrics: Dict):
        pass

    def on_batch_begin(self, trainer, batch_idx: int, batch: Any):
        pass

    def on_batch_end(self, trainer, batch_idx: int, batch: Any, outputs: Any, loss: float):
        pass


class ModelCheckpoint(Callback):
    """
    Save model checkpoints.
    """

    def __init__(
        self,
        dirpath: str = "./checkpoints",
        filename: str = "epoch_{epoch}",
        monitor: str = "val_loss",
        mode: str = "min",  # "min" | "max"
        save_best_only: bool = True,
        save_last: bool = True,
        max_keep: int = 5,
        verbose: bool = True,
    ):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.max_keep = max_keep
        self.verbose = verbose

        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.saved_checkpoints: List[Path] = []

        self.dirpath.mkdir(parents=True, exist_ok=True)

    def is_better(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def on_epoch_end(self, trainer, train_metrics: Dict, val_metrics: Dict):
        if not trainer.is_main_process():
            return

        current_score = val_metrics.get(self.monitor)
        if current_score is None:
            return

        is_best = False
        if self.is_better(current_score):
            is_best = True
            self.best_score = current_score

        should_save = is_best if self.save_best_only else True

        if should_save or (not self.save_best_only and self.save_last):
            filename = self.filename.format(epoch=trainer.epoch, step=trainer.step)
            if is_best:
                filename += "_best"
            self._save_checkpoint(trainer, filename, is_best=is_best)

        if self.save_last:
            self._save_checkpoint(trainer, "last")

        self._cleanup()

    def _save_checkpoint(self, trainer, filename: str, is_best: bool = False):
        path = self.dirpath / f"{filename}.pt"
        state = {
            "epoch": trainer.epoch,
            "step": trainer.step,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "scaler_state_dict": trainer.scaler.state_dict(),
            "best_metric": self.best_score,
        }
        torch.save(state, path)
        self.saved_checkpoints.append(path)

        if self.verbose:
            print(f"Checkpoint saved: {path} {'(best)' if is_best else ''}")

    def _cleanup(self):
        if len(self.saved_checkpoints) > self.max_keep:
            to_remove = self.saved_checkpoints[:-self.max_keep]
            for path in to_remove:
                if path.exists() and path.name != "last.pt":
                    path.unlink()
            self.saved_checkpoints = self.saved_checkpoints[-self.max_keep:]


class EarlyStopping(Callback):
    """
    Stop training when metric stops improving.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def is_better(self, score: float) -> bool:
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        return score > (self.best_score + self.min_delta)

    def on_train_begin(self, trainer):
        self.best_weights = trainer.model.state_dict().copy()

    def on_epoch_end(self, trainer, train_metrics: Dict, val_metrics: Dict):
        if not trainer.is_main_process():
            return

        current_score = val_metrics.get(self.monitor)
        if current_score is None:
            return

        if self.is_better(current_score):
            self.best_score = current_score
            self.counter = 0
            self.best_weights = trainer.model.state_dict().copy()
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} - {self.monitor}={current_score:.4f}")

            if self.counter >= self.patience:
                trainer.should_stop = True
                self.stopped_epoch = trainer.epoch
                if self.verbose:
                    print(f"Early stopping at epoch {self.stopped_epoch}")

                if self.restore_best_weights and self.best_weights:
                    trainer.model.load_state_dict(self.best_weights)


class LearningRateMonitor(Callback):
    """Log learning rate."""

    def __init__(self, logging_interval: str = "step"):  # "step" | "epoch"
        self.logging_interval = logging_interval

    def on_batch_end(self, trainer, batch_idx: int, batch: Any, outputs: Any, loss: float):
        if self.logging_interval == "step" and trainer.is_main_process():
            lr = trainer.optimizer.param_groups[0]["lr"]
            if hasattr(trainer, "tb_writer") and trainer.tb_writer:
                trainer.tb_writer.add_scalar("lr", lr, trainer.step)
            if hasattr(trainer, "wandb_run") and trainer.wandb_run:
                import wandb
                wandb.log({"lr": lr}, step=trainer.step)

    def on_epoch_end(self, trainer, train_metrics: Dict, val_metrics: Dict):
        if self.logging_interval == "epoch" and trainer.is_main_process():
            lr = trainer.optimizer.param_groups[0]["lr"]
            if hasattr(trainer, "tb_writer") and trainer.tb_writer:
                trainer.tb_writer.add_scalar("lr", lr, trainer.epoch)
            if hasattr(trainer, "wandb_run") and trainer.wandb_run:
                import wandb
                wandb.log({"lr": lr}, step=trainer.epoch)


class WandbCallback(Callback):
    """Weights & Biases logging callback."""

    def __init__(
        self,
        project: str = "ihrm",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        log_model: bool = False,
        log_freq: int = 50,
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.log_model = log_model
        self.log_freq = log_freq
        self.wandb = None

    def on_train_begin(self, trainer):
        if not trainer.is_main_process():
            return

        try:
            import wandb
            self.wandb = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                config={
                    "model": trainer.model_config.__dict__ if trainer.model_config else {},
                    "train": trainer.config.__dict__,
                    "data": trainer.data_config.__dict__ if trainer.data_config else {},
                },
            )
            if self.log_model:
                self.wandb.watch(trainer.model, log="all", log_freq=self.log_freq)
        except ImportError:
            pass

    def on_batch_end(self, trainer, batch_idx: int, batch: Any, outputs: Any, loss: float):
        if self.wandb and trainer.is_main_process() and trainer.step % self.log_freq == 0:
            self.wandb.log({"train/loss": loss, "step": trainer.step})

    def on_epoch_end(self, trainer, train_metrics: Dict, val_metrics: Dict):
        if self.wandb and trainer.is_main_process():
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            log_dict["epoch"] = trainer.epoch
            self.wandb.log(log_dict)

    def on_train_end(self, trainer):
        if self.wandb:
            if self.log_model and trainer.is_main_process():
                self.wandb.save_model(str(trainer.ckpt_dir / "best.pt"))
            self.wandb.finish()


class TensorBoardCallback(Callback):
    """TensorBoard logging callback."""

    def __init__(self, log_dir: str = "./logs/tensorboard", log_freq: int = 50):
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.writer = None

    def on_train_begin(self, trainer):
        if not trainer.is_main_process():
            return
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)

    def on_batch_end(self, trainer, batch_idx: int, batch: Any, outputs: Any, loss: float):
        if self.writer and trainer.step % self.log_freq == 0:
            self.writer.add_scalar("train/loss_batch", loss, trainer.step)
            self.writer.add_scalar("lr", trainer.optimizer.param_groups[0]["lr"], trainer.step)

    def on_epoch_end(self, trainer, train_metrics: Dict, val_metrics: Dict):
        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, trainer.epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, trainer.epoch)
            self.writer.add_scalar("lr", trainer.optimizer.param_groups[0]["lr"], trainer.epoch)

    def on_train_end(self, trainer):
        if self.writer:
            self.writer.close()


class GradientNormCallback(Callback):
    """Log gradient norms for monitoring training stability."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval

    def on_batch_end(self, trainer, batch_idx: int, batch: Any, outputs: Any, loss: float):
        if trainer.step % self.log_interval == 0 and trainer.is_main_process():
            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if hasattr(trainer, "tb_writer") and trainer.tb_writer:
                trainer.tb_writer.add_scalar("gradients/norm", total_norm, trainer.step)


class ProgressBarCallback(Callback):
    """Display training progress bar."""

    def __init__(self, total_steps: Optional[int] = None):
        self.total_steps = total_steps
        self.pbar = None

    def on_train_begin(self, trainer):
        try:
            from tqdm import tqdm
            self.pbar = tqdm(
                total=self.total_steps or len(trainer.train_loader) * trainer.config.epochs,
                desc="Training",
                disable=not trainer.is_main_process(),
            )
        except ImportError:
            pass

    def on_batch_end(self, trainer, batch_idx: int, batch: Any, outputs: Any, loss: float):
        if self.pbar:
            self.pbar.set_postfix({"loss": f"{loss:.4f}"})
            self.pbar.update(1)

    def on_train_end(self, trainer):
        if self.pbar:
            self.pbar.close()


# ─────────────────────────────────────────────────────────────────
# Callback Registry
# ─────────────────────────────────────────────────────────────────

CALLBACKS = {
    "model_checkpoint": ModelCheckpoint,
    "early_stopping": EarlyStopping,
    "lr_monitor": LearningRateMonitor,
    "wandb": WandbCallback,
    "tensorboard": TensorBoardCallback,
    "grad_norm": GradientNormCallback,
    "progress_bar": ProgressBarCallback,
}


def get_callbacks(config: Dict) -> List[Callback]:
    """Create callbacks from config dict."""
    callbacks = []
    for name, params in config.items():
        if name in CALLBACKS:
            callbacks.append(CALLBACKS[name](**params))
    return callbacks