"""
Trainer Module

Complete training loop with:
- Mixed precision (FP16/BF16)
- Gradient accumulation
- Distributed training (DDP/FSDP/DeepSpeed)
- Checkpointing
- Logging (TensorBoard, W&B)
- Early stopping
- Gradient clipping
"""

import os
import time
import math
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────
# Trainer Config
# ─────────────────────────────────────────────────────────────────

class TrainerConfig:
    """Configuration for Trainer (merged from TrainConfig)."""

    def __init__(self, **kwargs):
        # Training loop
        self.epochs: int = kwargs.get("epochs", 100)
        self.max_steps: Optional[int] = kwargs.get("max_steps", None)
        self.batch_size: int = kwargs.get("batch_size", 32)
        self.gradient_accumulation_steps: int = kwargs.get("gradient_accumulation_steps", 1)

        # Optimization
        # Support both 'optimizer' (TrainConfig) and 'optimizer_type' (TrainerConfig)
        self.optimizer_type: str = kwargs.get("optimizer_type", kwargs.get("optimizer", "adamw"))
        self.lr: float = kwargs.get("lr", 3e-4)
        self.weight_decay: float = kwargs.get("weight_decay", 0.05)
        self.beta1: float = kwargs.get("beta1", 0.9)
        self.beta2: float = kwargs.get("beta2", 0.999)
        self.eps: float = kwargs.get("eps", 1e-8)

        # Scheduler
        self.scheduler_type: str = kwargs.get("scheduler", "cosine_warmup")
        self.warmup_epochs: int = kwargs.get("warmup_epochs", 20)
        self.warmup_start_lr: float = kwargs.get("warmup_start_lr", 1e-6)
        self.min_lr: float = kwargs.get("min_lr", 1e-6)
        self.step_size: int = kwargs.get("step_size", 30)
        self.gamma: float = kwargs.get("gamma", 0.1)

        # Regularization
        self.grad_clip: float = kwargs.get("grad_clip", 1.0)
        self.label_smoothing: float = kwargs.get("label_smoothing", 0.1)

        # Precision
        self.precision: str = kwargs.get("precision", "fp16")
        self.bf16: bool = kwargs.get("bf16", False)

        # Distributed
        self.distributed_backend: str = kwargs.get("distributed_backend", "none")
        self.ddp_find_unused_parameters: bool = kwargs.get("ddp_find_unused_parameters", False)

        # Logging
        self.log_interval: int = kwargs.get("log_interval", 50)
        self.log_level: str = kwargs.get("log_level", "INFO")
        self.use_wandb: bool = kwargs.get("use_wandb", False)
        self.wandb_project: str = kwargs.get("wandb_project", "ihrm")
        self.wandb_run: Optional[str] = kwargs.get("wandb_run", None)
        self.use_tensorboard: bool = kwargs.get("use_tensorboard", True)
        self.tensorboard_dir: str = kwargs.get("tensorboard_dir", "./logs/tensorboard")

        # Evaluation
        self.eval_interval: int = kwargs.get("eval_interval", 1)
        self.eval_batch_size: int = kwargs.get("eval_batch_size", None)

        # Checkpointing
        self.save_interval: int = kwargs.get("save_interval", 10)
        self.save_best_only: bool = kwargs.get("save_best_only", True)
        self.save_last: bool = kwargs.get("save_last", True)
        self.max_keep_ckpts: int = kwargs.get("max_keep_ckpts", 5)
        self.checkpoint_dir: str = kwargs.get("checkpoint_dir", "./checkpoints")
        self.resume_from: Optional[str] = kwargs.get("resume_from", None)

        # HF Hub
        self.push_to_hub: bool = kwargs.get("push_to_hub", False)
        self.hub_repo: Optional[str] = kwargs.get("hub_repo", None)
        self.hub_token: Optional[str] = kwargs.get("hub_token", None)
        self.hub_private: bool = kwargs.get("hub_private", False)

        # Reproducibility
        self.seed: int = kwargs.get("seed", 42)
        self.deterministic: bool = kwargs.get("deterministic", False)
        self.benchmark: bool = kwargs.get("benchmark", True)

        # Early stopping
        self.early_stopping: bool = kwargs.get("early_stopping", False)
        self.early_stopping_patience: int = kwargs.get("early_stopping_patience", 10)
        self.early_stopping_metric: str = kwargs.get("early_stopping_metric", "val_loss")
        self.early_stopping_mode: str = kwargs.get("early_stopping_mode", "min")

        # Halting loss weight
        self.halt_loss_weight: float = kwargs.get("halt_loss_weight", 0.1)

        # Compile
        self.compile: bool = kwargs.get("compile", False)


# ─────────────────────────────────────────────────────────────────
# Trainer Class
# ─────────────────────────────────────────────────────────────────

class Trainer:
    """
    Main training class for I-HRM models.

    Supports:
    - Single GPU / Multi-GPU (DDP)
    - Mixed precision (FP16, BF16)
    - Gradient accumulation
    - Distributed training (DDP, FSDP, DeepSpeed)
    - Checkpointing & resume
    - TensorBoard & W&B logging
    - Early stopping
    - Custom callbacks
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
        model_config: Optional[Any] = None,
        data_config: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None,
        loss_fn: Optional[Callable] = None,
        metrics: Optional[Dict[str, Callable]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # Convert TrainConfig to TrainerConfig if needed
        if config is not None and not isinstance(config, TrainerConfig):
            # Convert TrainConfig (from src.config.train_config) to TrainerConfig
            cfg_dict = config.__dict__.copy()
            # Map TrainConfig attributes to TrainerConfig attributes
            cfg_dict['optimizer_type'] = cfg_dict.get('optimizer', 'adamw')
            cfg_dict['scheduler_type'] = cfg_dict.get('scheduler', 'cosine_warmup')
            cfg_dict['precision'] = cfg_dict.get('precision', 'fp16')
            cfg_dict['distributed_backend'] = cfg_dict.get('distributed', 'none')
            self.config = TrainerConfig(**cfg_dict)
        else:
            self.config = config or TrainerConfig()
        self.model_config = model_config
        self.data_config = data_config
        self.callbacks = callbacks or []
        self.loss_fn = loss_fn
        self.metrics = metrics or {}

        # Device setup
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        # Distributed setup
        self._setup_distributed()

        # Precision
        self.scaler = GradScaler(enabled=(self.config.precision == "fp16"))
        self.autocast_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        # Compile
        if self.config.compile and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Optimizer & Scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # State
        self.epoch = 0
        self.step = 0
        self.best_metric = float("inf") if self.config.early_stopping_mode == "min" else -float("inf")
        self.early_stop_counter = 0
        self.train_losses = []
        self.val_metrics = {}

        # Logging
        self._setup_logging()

        # Checkpoint directory
        self.ckpt_dir = Path(self.config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Resume if requested
        if self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            if self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = self.config.benchmark
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        return device

    def _setup_distributed(self):
        """Setup distributed training."""
        self.is_distributed = False
        self.world_size = 1
        self.local_rank = 0
        self.rank = 0

        if self.config.distributed_backend != "none" and torch.cuda.device_count() > 1:
            if self.config.distributed_backend == "ddp":
                self._setup_ddp()
            elif self.config.distributed_backend == "fsdp":
                self._setup_fsdp()
            elif self.config.distributed_backend == "deepspeed":
                self._setup_deepspeed()

    def _setup_ddp(self):
        """Setup DistributedDataParallel."""
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        self.model = DistributedDataParallel(
            self.model.to(self.device),
            device_ids=[self.local_rank],
            find_unused_parameters=self.config.ddp_find_unused_parameters,
        )
        self.is_distributed = True
        logger.info(f"DDP initialized: rank={self.rank}, world_size={self.world_size}")

    def _setup_fsdp(self):
        """Setup Fully Sharded Data Parallel."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Implementation would go here
        logger.warning("FSDP not fully implemented yet, falling back to DDP")
        self._setup_ddp()

    def _setup_deepspeed(self):
        """Setup DeepSpeed."""
        logger.warning("DeepSpeed not fully implemented yet, falling back to DDP")
        self._setup_ddp()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        from src.training.optim import create_optimizer

        opt = create_optimizer(
            self.model,
            opt_type=self.config.optimizer_type,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )
        return opt

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        from src.training.optim import create_scheduler

        total_steps = len(self.train_loader) // self.config.gradient_accumulation_steps * self.config.epochs
        if self.config.max_steps:
            total_steps = min(total_steps, self.config.max_steps)

        sched = create_scheduler(
            self.optimizer,
            sched_type=self.config.scheduler_type,
            num_warmup_steps=self.config.warmup_epochs * len(self.train_loader),
            num_training_steps=total_steps,
            min_lr=self.config.min_lr,
        )
        return sched

    def _setup_logging(self):
        """Setup TensorBoard and W&B logging."""
        self.tb_writer = None
        self.wandb_run = None

        if self.is_main_process() and self.config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.config.tensorboard_dir)
            logger.info(f"TensorBoard logging to {self.config.tensorboard_dir}")

        if self.is_main_process() and self.config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run,
                    config={
                        "model": self.model_config.__dict__ if self.model_config else {},
                        "train": self.config.__dict__,
                        "data": self.data_config.__dict__ if self.data_config else {},
                    },
                )
                logger.info(f"W&B run: {self.wandb_run.name}")
            except ImportError:
                logger.warning("wandb not installed, skipping")

    def is_main_process(self) -> bool:
        """Check if current process is main (rank 0)."""
        return self.rank == 0

    # ─────────────────────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────────────────────

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.config.epochs}, Steps per epoch: {len(self.train_loader)}")

        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch

            # Train epoch
            train_metrics = self._train_epoch()

            # Validation
            val_metrics = {}
            if self.val_loader and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate(self.val_loader)

            # Logging
            self._log_epoch(train_metrics, val_metrics)

            # Callbacks
            for cb in self.callbacks:
                cb.on_epoch_end(self, train_metrics, val_metrics)

            # Checkpointing
            self._maybe_save_checkpoint(val_metrics)

            # Early stopping
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            # Step scheduler (per epoch for some schedulers)
            if self.scheduler and self.config.scheduler_type in ["step", "exponential"]:
                self.scheduler.step()

        # Final evaluation
        if self.test_loader:
            test_metrics = self.evaluate(self.test_loader, prefix="test")
            logger.info(f"Final test metrics: {test_metrics}")

        # Push to hub
        if self.config.push_to_hub and self.is_main_process():
            self._push_to_hub()

        # Close loggers
        if self.tb_writer:
            self.tb_writer.close()

        logger.info("Training complete!")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0
        metric_sums = defaultdict(float)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._to_device(batch)

            # Forward + backward
            loss, metrics = self._train_step(batch)

            epoch_loss += loss
            num_batches += 1
            for k, v in metrics.items():
                metric_sums[k] += v

            # Logging
            if batch_idx % self.config.log_interval == 0 and self.is_main_process():
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss:.4f} LR: {lr:.2e}"
                )

            # Step scheduler (per step for cosine)
            if self.scheduler and self.config.scheduler_type in ["cosine", "cosine_warmup", "one_cycle"]:
                self.scheduler.step()

            self.step += 1

            # Max steps check
            if self.config.max_steps and self.step >= self.config.max_steps:
                break

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_metrics = {k: v / max(num_batches, 1) for k, v in metric_sums.items()}
        avg_metrics["loss"] = avg_loss

        return avg_metrics

    def _train_step(self, batch) -> tuple:
        """Single training step."""
        # Handle different batch formats
        if isinstance(batch, dict):
            pixel_values = batch["image"]
            labels = batch.get("label") or batch.get("labels")
        else:
            pixel_values, labels = batch

        pixel_values = pixel_values.to(self.device, non_blocking=True)
        if labels is not None:
            labels = labels.to(self.device, non_blocking=True)

        # Mixed precision forward
        # Disable autocast on CPU (device_type not supported for CPU)
        use_amp = (self.config.precision != "fp32") and (self.device.type == "cuda")
        with autocast(dtype=self.autocast_dtype, enabled=use_amp):
            outputs = self.model(pixel_values, return_steps=True)
            features, steps = outputs if isinstance(outputs, tuple) else (outputs, None)

            # Compute loss
            if self.loss_fn:
                loss = self.loss_fn(features, labels, steps)
            else:
                # Default: classification
                logits = self.model.heads.classification_head(features)
                loss = F.cross_entropy(logits, labels, label_smoothing=self.config.label_smoothing)

            # Halting loss for adaptive halting
            if steps and self.model.config.max_steps > 1:
                halt_loss = self._compute_halt_loss(steps)
                loss = loss + self.config.halt_loss_weight * halt_loss

        # Backward
        self.scaler.scale(loss).backward()

        # Gradient accumulation
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        # Compute metrics
        with torch.no_grad():
            metrics = self._compute_metrics(features, labels)

        return loss.item(), metrics

    def _compute_halt_loss(self, steps: Dict) -> torch.Tensor:
        """Compute adaptive halting loss (Algorithm 1)."""
        # steps contains halting probabilities and steps taken
        halt_probs = steps.get("halt_probs")  # [B, K]
        halt_steps = steps.get("halt_steps")  # [B]

        if halt_probs is None:
            return torch.tensor(0.0, device=self.device)

        # Expected halting step
        expected_steps = torch.sum(halt_probs * torch.arange(1, halt_probs.size(1) + 1, device=self.device), dim=1)
        halt_loss = F.mse_loss(expected_steps.float(), halt_steps.float())
        return halt_loss

    def _compute_metrics(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Compute metrics for current batch."""
        metrics = {}

        if labels is not None and "accuracy" in self.metrics:
            # Classification accuracy
            logits = self.model.heads.classification_head(features)
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
            metrics["accuracy"] = acc

        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        """Evaluate model on dataloader."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        metric_sums = defaultdict(float)

        for batch in loader:
            batch = self._to_device(batch)

            if isinstance(batch, dict):
                pixel_values = batch["image"]
                labels = batch.get("label") or batch.get("labels")
            else:
                pixel_values, labels = batch

            pixel_values = pixel_values.to(self.device, non_blocking=True)
            if labels is not None:
                labels = labels.to(self.device, non_blocking=True)

            use_amp = (self.config.precision != "fp32") and (self.device.type == "cuda")
            with autocast(dtype=self.autocast_dtype, enabled=use_amp):
                outputs = self.model(pixel_values, return_steps=True)
                features, steps = outputs if isinstance(outputs, tuple) else (outputs, None)

                if self.loss_fn:
                    loss = self.loss_fn(features, labels, steps)
                else:
                    logits = self.model.heads.classification_head(features)
                    loss = F.cross_entropy(logits, labels, label_smoothing=self.config.label_smoothing)

            total_loss += loss.item()
            num_batches = 1  # simplified

            # Collect predictions
            if labels is not None:
                logits = self.model.heads.classification_head(features)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # Compute metrics
        avg_loss = total_loss / max(num_batches, 1)
        metrics = {f"{prefix}_loss": avg_loss}

        if all_preds:
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            # Accuracy
            acc = (all_preds == all_labels).float().mean().item()
            metrics[f"{prefix}_accuracy"] = acc

            # Top-5
            # Would need logits, simplified here
            if hasattr(self.metrics, "top5"):
                metrics[f"{prefix}_top5"] = self.metrics["top5"](all_preds, all_labels).item()

        self.model.train()
        return metrics

    def _to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return [self._to_device(b) for b in batch]
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch

    # ─────────────────────────────────────────────────────────────
    # Logging & Checkpointing
    # ─────────────────────────────────────────────────────────────

    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch metrics."""
        log_str = f"Epoch {self.epoch} | "
        log_str += f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
        log_str += f"Train Acc: {train_metrics.get('accuracy', 0):.4f}"

        if val_metrics:
            log_str += f" | Val Loss: {val_metrics.get('val_loss', 0):.4f} | "
            log_str += f"Val Acc: {val_metrics.get('val_accuracy', 0):.4f}"

        if self.is_main_process():
            logger.info(log_str)

            # TensorBoard
            if self.tb_writer:
                for k, v in train_metrics.items():
                    self.tb_writer.add_scalar(f"train/{k}", v, self.epoch)
                for k, v in val_metrics.items():
                    self.tb_writer.add_scalar(f"val/{k}", v, self.epoch)
                self.tb_writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.epoch)

            # W&B
            if self.wandb_run:
                import wandb
                wandb.log({**{f"train/{k}": v for k, v in train_metrics.items()},
                          **{f"val/{k}": v for k, v in val_metrics.items()},
                          "epoch": self.epoch})

    def _maybe_save_checkpoint(self, val_metrics: Dict):
        """Save checkpoint if needed."""
        if not self.is_main_process():
            return

        should_save = False
        is_best = False

        if self.config.save_best_only and val_metrics:
            metric_val = val_metrics.get(self.config.early_stopping_metric, val_metrics.get("val_loss", float("inf")))
            if self.config.early_stopping_mode == "min":
                is_best = metric_val < self.best_metric
                if is_best:
                    self.best_metric = metric_val
            else:
                is_best = metric_val > self.best_metric
                if is_best:
                    self.best_metric = metric_val

            if is_best:
                should_save = True
        elif (self.epoch + 1) % self.config.save_interval == 0:
            should_save = True

        if should_save:
            filename = f"epoch_{self.epoch}"
            if is_best:
                filename += "_best"
            self.save_checkpoint(self.ckpt_dir / f"{filename}.pt", is_best=is_best)

        # Save last
        if self.config.save_last:
            self.save_checkpoint(self.ckpt_dir / "last.pt")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only max_keep_ckpts checkpoints."""
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > self.config.max_keep_ckpts:
            ckpts[0].unlink()
            ckpts = ckpts[1:]

    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """Check early stopping condition."""
        if not self.config.early_stopping or not val_metrics:
            return False

        metric_val = val_metrics.get(self.config.early_stopping_metric)
        if metric_val is None:
            return False

        if self.config.early_stopping_mode == "min":
            improved = metric_val < self.best_metric
        else:
            improved = metric_val > self.best_metric

        if improved:
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        return self.early_stop_counter >= self.config.early_stopping_patience

    # ─────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ─────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save training checkpoint."""
        state = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config.__dict__,
            "model_config": self.model_config.__dict__ if self.model_config else None,
            "data_config": self.data_config.__dict__ if self.data_config else None,
        }
        torch.save(state, path)
        if self.is_main_process():
            logger.info(f"Checkpoint saved: {path} {'(best)' if is_best else ''}")

    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        logger.info(f"Loading checkpoint: {path}")
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if self.scheduler and state["scheduler_state_dict"]:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])

        self.epoch = state["epoch"] + 1
        self.step = state["step"]
        self.best_metric = state["best_metric"]

        logger.info(f"Resumed from epoch {self.epoch}, step {self.step}")

    def _push_to_hub(self):
        """Push model to Hugging Face Hub."""
        if not self.config.hub_repo:
            return

        try:
            from transformers import AutoModel
            from src.models.unified_hrm import UnifiedCVHRM, UnifiedCVHRMConfig

            # Convert to HF format
            hf_config = UnifiedCVHRMConfig.from_local_config(self.model_config)
            hf_model = UnifiedCVHRM(hf_config)
            hf_model.backbone.load_state_dict(self.model.state_dict())

            hf_model.push_to_hub(
                self.config.hub_repo,
                token=self.config.hub_token,
                private=self.config.hub_private,
            )
            logger.info(f"Model pushed to {self.config.hub_repo}")
        except Exception as e:
            logger.error(f"Failed to push to hub: {e}")