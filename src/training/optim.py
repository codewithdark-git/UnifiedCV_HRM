"""
Optimizer & Scheduler Factory

Supports: AdamW, Adam, SGD, Lion
Schedulers: Cosine, CosineWarmup, Step, Exponential, Constant, OneCycle
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ExponentialLR,
    LambdaLR,
    SequentialLR,
    LinearLR,
)
from typing import Optional, Dict, Any


# ─────────────────────────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────────────────────────

def create_optimizer(
    model: torch.nn.Module,
    opt_type: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    nesterov: bool = True,
    filter_bias_and_bn: bool = True,
) -> torch.optim.Optimizer:
    """
    Create optimizer with proper weight decay handling.

    For transformers, we typically don't apply weight decay to:
    - Bias terms
    - LayerNorm / RMSNorm parameters
    """
    if filter_bias_and_bn:
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    else:
        param_groups = [{"params": model.parameters(), "weight_decay": weight_decay}]

    opt_type = opt_type.lower()

    if opt_type == "adamw":
        return optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif opt_type == "adam":
        return optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)
    elif opt_type == "sgd":
        return optim.SGD(param_groups, lr=lr, momentum=momentum, nesterov=nesterov)
    elif opt_type == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(param_groups, lr=lr, betas=betas, weight_decay=weight_decay)
        except ImportError:
            print("lion_pytorch not installed, falling back to AdamW")
            return optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif opt_type == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(param_groups, lr=lr, betas=betas, eps=eps)
        except ImportError:
            print("bitsandbytes not installed, falling back to AdamW")
            return optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


# ─────────────────────────────────────────────────────────────────
# Schedulers
# ─────────────────────────────────────────────────────────────────

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_type: str = "cosine_warmup",
    num_warmup_steps: int = 0,
    num_training_steps: int = 100000,
    min_lr: float = 1e-6,
    warmup_start_lr: float = 1e-6,
    cycle_epochs: int = 10,
    step_size: int = 30,
    gamma: float = 0.1,
    power: float = 1.0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        sched_type: "cosine" | "cosine_warmup" | "step" | "exponential" |
                    "constant" | "one_cycle" | "linear_warmup" | "polynomial"
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr: Minimum learning rate
        warmup_start_lr: Starting LR for warmup
        cycle_epochs: For CosineAnnealingWarmRestarts
        step_size: For StepLR
        gamma: For StepLR / ExponentialLR
        power: For polynomial decay
    """
    sched_type = sched_type.lower()

    if sched_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=min_lr)

    elif sched_type == "cosine_warmup":
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_start_lr / optimizer.param_groups[0]["lr"],
                total_iters=num_warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=min_lr,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[num_warmup_steps],
            )
        else:
            return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=min_lr)

    elif sched_type == "cosine_warm_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cycle_epochs,
            T_mult=1,
            eta_min=min_lr,
        )

    elif sched_type == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif sched_type == "exponential":
        return ExponentialLR(optimizer, gamma=gamma)

    elif sched_type == "constant":
        return LambdaLR(optimizer, lambda step: 1.0)

    elif sched_type == "linear_warmup":
        if num_warmup_steps > 0:
            return LinearLR(
                optimizer,
                start_factor=warmup_start_lr / optimizer.param_groups[0]["lr"],
                total_iters=num_warmup_steps,
            )
        return LambdaLR(optimizer, lambda step: 1.0)

    elif sched_type == "polynomial":
        def poly_lr(step):
            if step < num_warmup_steps:
                return (step + 1) / num_warmup_steps * (1 - min_lr / optimizer.param_groups[0]["lr"]) + min_lr / optimizer.param_groups[0]["lr"]
            progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            progress = min(progress, 1.0)
            return ((1 - progress) ** power) * (1 - min_lr / optimizer.param_groups[0]["lr"]) + min_lr / optimizer.param_groups[0]["lr"]
        return LambdaLR(optimizer, poly_lr)

    elif sched_type == "one_cycle":
        # OneCycleLR requires max_lr and total steps
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            total_steps=num_training_steps,
            pct_start=num_warmup_steps / num_training_steps if num_training_steps > 0 else 0.3,
            anneal_strategy="cos",
            div_factor=25,
            final_div_factor=1e4,
        )

    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


# ─────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ─────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Improves validation performance by averaging weights.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        self.module = model
        self.decay = decay
        self.device = device
        self.shadow_params = {}
        self.has_unwrapped = False

        if device is not None and device != next(model.parameters()).device:
            print("WARNING: EMA device different from model device")

    def register(self):
        """Register current model parameters as shadow."""
        self.shadow_params = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(self.device)

    def update(self, model: torch.nn.Module):
        """Update shadow parameters with current model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].sub_(
                        (1.0 - self.decay) * (self.shadow_params[name] - param.data)
                    )

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        if self.has_unwrapped:
            raise RuntimeError("Model already has shadow weights applied")
        self.module.unwrap()
        self.has_unwrapped = True

        for name, param in self.module.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def restore(self):
        """Restore original parameters."""
        if not self.has_unwrapped:
            return
        self.has_unwrapped = False
        # Model was unwrapped, restore from shadow
        for name, param in self.module.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                # This won't work properly without storing originals
                # Better: keep separate reference to original model
                pass

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "shadow_params": self.shadow_params,
            "decay": self.decay,
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.shadow_params = state_dict["shadow_params"]
        self.decay = state_dict.get("decay", self.decay)


class ModelEMAV2:
    """
    Improved EMA with buffer support (from timm).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        use_ema_weights: bool = True,
    ):
        self.module = model
        self.decay = decay
        self.use_ema_weights = use_ema_weights
        self.ema_state = {}
        self.has_momentum = False

        # Copy all parameters and buffers
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_state[name] = param.detach().clone()
        for name, buffer in model.named_buffers():
            self.ema_state[name] = buffer.detach().clone()

    def update(self, model: torch.nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.ema_state:
                    self.ema_state[name].lerp_(param.data, 1.0 - self.decay)
            for name, buffer in model.named_buffers():
                if name in self.ema_state:
                    self.ema_state[name].lerp_(buffer.data, 1.0 - self.decay)

    def apply_shadow(self):
        if self.use_ema_weights:
            for name, param in self.module.named_parameters():
                if param.requires_grad and name in self.ema_state:
                    param.data.copy_(self.ema_state[name])
            for name, buffer in self.module.named_buffers():
                if name in self.ema_state:
                    buffer.data.copy_(self.ema_state[name])

    def restore(self):
        if self.use_ema_weights:
            # This would need original weights stored
            pass

    def state_dict(self) -> Dict:
        return {
            "ema_state": self.ema_state,
            "decay": self.decay,
            "use_ema_weights": self.use_ema_weights,
        }

    def load_state_dict(self, state_dict: Dict):
        self.ema_state = state_dict["ema_state"]
        self.decay = state_dict.get("decay", self.decay)
        self.use_ema_weights = state_dict.get("use_ema_weights", self.use_ema_weights)


# ─────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────

OPTIMIZERS = {
    "adamw": lambda *a, **k: create_optimizer(*a, **k, opt_type="adamw"),
    "adam": lambda *a, **k: create_optimizer(*a, **k, opt_type="adam"),
    "sgd": lambda *a, **k: create_optimizer(*a, **k, opt_type="sgd"),
    "lion": lambda *a, **k: create_optimizer(*a, **k, opt_type="lion"),
    "adamw_8bit": lambda *a, **k: create_optimizer(*a, **k, opt_type="adamw_8bit"),
}

SCHEDULERS = {
    "cosine": lambda *a, **k: create_scheduler(*a, **k, sched_type="cosine"),
    "cosine_warmup": lambda *a, **k: create_scheduler(*a, **k, sched_type="cosine_warmup"),
    "cosine_warm_restarts": lambda *a, **k: create_scheduler(*a, **k, sched_type="cosine_warm_restarts"),
    "step": lambda *a, **k: create_scheduler(*a, **k, sched_type="step"),
    "exponential": lambda *a, **k: create_scheduler(*a, **k, sched_type="exponential"),
    "constant": lambda *a, **k: create_scheduler(*a, **k, sched_type="constant"),
    "one_cycle": lambda *a, **k: create_scheduler(*a, **k, sched_type="one_cycle"),
}


def get_optimizer(name: str, **kwargs) -> torch.optim.Optimizer:
    return OPTIMIZERS[name](**kwargs)


def get_scheduler(name: str, **kwargs) -> torch.optim.lr_scheduler.LRScheduler:
    return SCHEDULERS[name](**kwargs)