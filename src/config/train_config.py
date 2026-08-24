"""Training Configuration

Complete training setup: optimizer, scheduler, precision, logging, checkpointing.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import yaml


class OptimizerType(str, Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    LION = "lion"


class SchedulerType(str, Enum):
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    STEP = "step"
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"
    ONE_CYCLE = "one_cycle"


class PrecisionType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class DistributedBackend(str, Enum):
    DDP = "ddp"
    FSDP = "fsdp"
    DEEPSPEED = "deepspeed"
    NONE = "none"


@dataclass
class TrainConfig:
    """
    Complete training configuration.

    Paper training specs:
    - CIFAR-100: 300 epochs, batch 128, lr 3e-4, cos + 20ep warmup
    - ImageNet-1K: 100 epochs, batch 256 (or 32×8 accum), lr 3e-4, cos + 20ep warmup
    - Medical: 100 epochs, batch 32, lr 1e-4, cos + 10ep warmup
    """

    # Training loop
    epochs: int = 100
    batch_size: int = 32
    max_steps: Optional[int] = None           # Override epochs if set
    gradient_accumulation_steps: int = 1

    # Optimization
    optimizer: OptimizerType = OptimizerType.ADAMW
    lr: float = 3e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Scheduler
    scheduler: SchedulerType = SchedulerType.COSINE_WARMUP
    warmup_epochs: int = 20
    warmup_start_lr: float = 1e-6
    min_lr: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1

    # Regularization
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

    # Precision
    precision: PrecisionType = PrecisionType.FP16
    bf16: bool = False                        # Use BF16 if available (Ampere+)

    # Distributed
    distributed_backend: DistributedBackend = DistributedBackend.NONE
    ddp_find_unused_parameters: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"

    # Logging
    log_interval: int = 50
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "unifiedcv-hrm"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    use_tensorboard: bool = True
    tensorboard_dir: str = "./logs/tensorboard"

    # Evaluation
    eval_interval: int = 1
    eval_batch_size: Optional[int] = None     # Defaults to batch_size

    # Checkpointing
    save_interval: int = 10                   # Save every N epochs
    save_best_only: bool = True
    save_last: bool = True
    max_keep_ckpts: int = 5
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None

    # HF Hub
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None
    hub_token: Optional[str] = None
    hub_private: bool = False
    hub_commit_every: int = 1                  # Push every N epochs

    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True

    # Output/Logging (CLI convenience fields)
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    wandb: bool = False
    wandb_project: str = "unifiedcv-hrm"
    wandb_run: Optional[str] = None

    # Compile/Distributed (CLI convenience)
    compile: bool = False
    distributed: Optional[str] = None
    num_workers: int = 8
    pin_memory: bool = True

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"          # "min" | "max"

    # Halting loss (for adaptive halting)
    halt_loss_weight: float = 0.1

    def __post_init__(self):
        if isinstance(self.optimizer, str):
            self.optimizer = OptimizerType(self.optimizer)
        if isinstance(self.scheduler, str):
            self.scheduler = SchedulerType(self.scheduler)
        if isinstance(self.precision, str):
            self.precision = PrecisionType(self.precision)
        if isinstance(self.distributed_backend, str):
            self.distributed_backend = DistributedBackend(self.distributed_backend)

        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size if hasattr(self, 'batch_size') else 256

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["optimizer"] = self.optimizer.value
        data["scheduler"] = self.scheduler.value
        data["precision"] = self.precision.value
        data["distributed_backend"] = self.distributed_backend.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        if "optimizer" in data and isinstance(data["optimizer"], str):
            data["optimizer"] = OptimizerType(data["optimizer"])
        if "scheduler" in data and isinstance(data["scheduler"], str):
            data["scheduler"] = SchedulerType(data["scheduler"])
        if "precision" in data and isinstance(data["precision"], str):
            data["precision"] = PrecisionType(data["precision"])
        if "distributed_backend" in data and isinstance(data["distributed_backend"], str):
            data["distributed_backend"] = DistributedBackend(data["distributed_backend"])
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# ─────────────────────────────────────────────────────────────────
# Paper Training Presets
# ─────────────────────────────────────────────────────────────────

def quick_debug() -> TrainConfig:
    """Fast debug run: 1 epoch, small batch, synthetic data"""
    return TrainConfig(
        epochs=1,
        batch_size=4,
        lr=1e-4,
        log_interval=10,
        eval_interval=1,
        save_interval=1,
        use_wandb=False,
        use_tensorboard=False,
        precision=PrecisionType.FP32,
        seed=42,
    )


def cifar100_paper() -> TrainConfig:
    """CIFAR-100 paper config: 300 epochs, batch 128, cos+warmup"""
    return TrainConfig(
        epochs=300,
        batch_size=128,
        lr=3e-4,
        weight_decay=0.05,
        optimizer=OptimizerType.ADAMW,
        scheduler=SchedulerType.COSINE_WARMUP,
        warmup_epochs=20,
        grad_clip=1.0,
        label_smoothing=0.1,
        precision=PrecisionType.FP16,
        log_interval=50,
        eval_interval=1,
        save_interval=10,
        use_wandb=True,
        seed=42,
    )


def imagenet1k_paper() -> TrainConfig:
    """ImageNet-1K paper config: 100 epochs, batch 256 (or 32×8 accum)"""
    return TrainConfig(
        epochs=100,
        batch_size=32,           # Per GPU; use gradient_accumulation_steps=8 for 256 global
        gradient_accumulation_steps=8,
        lr=3e-4,
        weight_decay=0.05,
        optimizer=OptimizerType.ADAMW,
        scheduler=SchedulerType.COSINE_WARMUP,
        warmup_epochs=20,
        grad_clip=1.0,
        label_smoothing=0.1,
        precision=PrecisionType.FP16,
        distributed_backend=DistributedBackend.DDP,
        log_interval=100,
        eval_interval=1,
        save_interval=5,
        use_wandb=True,
        seed=42,
    )


def medical_paper() -> TrainConfig:
    """Medical datasets: 100 epochs, batch 32, lr 1e-4"""
    return TrainConfig(
        epochs=100,
        batch_size=32,
        lr=1e-4,
        weight_decay=0.01,
        optimizer=OptimizerType.ADAMW,
        scheduler=SchedulerType.COSINE_WARMUP,
        warmup_epochs=10,
        grad_clip=1.0,
        precision=PrecisionType.FP16,
        log_interval=20,
        eval_interval=1,
        save_interval=10,
        use_wandb=True,
        seed=42,
    )


def single_gpu_default() -> TrainConfig:
    """Single GPU friendly defaults (e.g., T4 16GB)"""
    return TrainConfig(
        epochs=100,
        batch_size=32,
        lr=3e-4,
        gradient_accumulation_steps=4,  # Effective batch 128
        distributed_backend=DistributedBackend.NONE,
        precision=PrecisionType.FP16,
        seed=42,
    )


def multi_gpu_default(num_gpus: int = 8) -> TrainConfig:
    """Multi-GPU DDP defaults"""
    return TrainConfig(
        epochs=100,
        batch_size=32,  # Per GPU
        lr=3e-4 * num_gpus,  # Linear scaling
        gradient_accumulation_steps=1,
        distributed_backend=DistributedBackend.DDP,
        precision=PrecisionType.BF16 if num_gpus >= 1 else PrecisionType.FP16,
        seed=42,
    )


# Auto-export presets
PRESET_CONFIGS = {
    "quick_debug": quick_debug,
    "cifar100_paper": cifar100_paper,
    "imagenet1k_paper": imagenet1k_paper,
    "medical_paper": medical_paper,
    "single_gpu": single_gpu_default,
    "multi_gpu": multi_gpu_default,
}