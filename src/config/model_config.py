"""Model Configuration

Defines the complete architecture configuration for HRM models.
All values align with the I-HRM paper specifications (Tables 1-2).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import yaml


class AttentionType(str, Enum):
    """Attention mechanism variants (Table 2)"""
    MHA = "mha"                    # Multi-Head Attention (Standard)
    DSA = "dsa"                    # Dynamic Sparse Attention (w=7)
    DELTA = "delta"                # L2-Normalized Delta Attention
    FLASH = "flash"                # FlashAttention-2 (optimized MHA)


class FFNType(str, Enum):
    """Feed-Forward Network variants (Table 2)"""
    SWIGLU = "swiglu"              # SwiGLU (Standard)
    MOE = "moe"                    # Dense-Routed Mixture-of-Experts
    DELTA_NET = "delta_net"        # Convolutional DeltaNet


class TaskType(str, Enum):
    """Task types for model heads"""
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    MULTI_TASK = "multi_task"


@dataclass
class HRMConfig:
    """
    Complete model configuration for Hierarchical Reasoning Model.

    Paper Reference: "I-HRM: Image-Hierarchical Reasoning Model"

    Key architectural decisions (Table 2):
    - Standard: MHA + SwiGLU (3.4M params)
    - DSA: DSA(w=7) + SwiGLU (3.4M params)
    - MoE: MHA + MoE(8 experts) dense (8.1M params)
    - DSA+MoE: DSA(w=7) + MoE(8 experts) dense (8.1M params) ← Best

    Adaptive Halting (Algorithm 1):
    - max_steps: K_max (default 6)
    - halt_threshold: τ (default 0.5)
    - halt_loss_weight: λ (default 0.1)
    """

    # Architecture - Core (matching paper)
    hidden_size: int = 384              # D in paper (was 256 in code, paper uses 384)
    num_heads: int = 8                  # H in paper
    num_layers_per_stream: int = 2      # Transformer blocks per H/L stream
    max_steps: int = 6                  # K_max in Algorithm 1
    halt_threshold: float = 0.5         # τ in Algorithm 1
    halt_loss_weight: float = 0.1       # λ for halting loss

    # Attention variant
    attention_type: AttentionType = AttentionType.DSA
    dsa_window: int = 7                 # w=7 for DSA (Table 2)
    use_rotary: bool = False            # No rotary for vision (non-causal)
    causal: bool = False

    # FFN variant
    ffn_type: FFNType = FFNType.MOE
    ffn_expansion: float = 4.0          # Expansion factor (SwiGLU: 4x, paper uses 2x)
    moe_num_experts: int = 8            # Number of experts (Table 2)
    moe_routing: str = "dense"          # "dense" | "top2" (paper: dense prevents collapse)
    moe_top_k: int = 2                  # For top-k routing

    # Tokenizer / Input
    image_size: int = 224
    patch_size: int = 16                # p=16 → L=196 patches (paper Fig 1)
    in_channels: int = 3
    use_cls_token: bool = True
    use_pos_embed: bool = True

    # Video (optional)
    num_frames: int = 8
    video_patch_size: tuple = (2, 4, 4)  # (T, H, W)

    # Task heads
    task: TaskType = TaskType.CLASSIFICATION
    num_classes: int = 1000

    # Segmentation head
    seg_num_classes: int = 80

    # Detection head
    det_num_classes: int = 80
    det_num_proposals: int = 100

    # Normalization
    norm_eps: float = 1e-5
    use_rmsnorm: bool = True            # Paper uses RMSNorm

    # Initialization
    init_std: float = 0.02

    # HF compatibility
    model_type: str = "unified_cv_hrm"
    architectures: List[str] = field(default_factory=lambda: ["UnifiedCVHRM"])

    def __post_init__(self):
        """Validate configuration"""
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")

        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})")

        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

        if not 0 < self.halt_threshold <= 1:
            raise ValueError("halt_threshold must be in (0, 1]")

        if self.moe_routing not in ["dense", "top2", "topk"]:
            raise ValueError(f"Invalid moe_routing: {self.moe_routing}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def seq_len(self) -> int:
        return self.num_patches + (1 if self.use_cls_token else 0)

    @property
    def video_tokens(self) -> tuple:
        """(T', H', W') for video tokenizer output"""
        T_p, H_p, W_p = self.video_patch_size
        return (
            self.num_frames // T_p,
            self.image_size // H_p,
            self.image_size // W_p,
        )

    # Preset configurations matching paper Table 2
    @classmethod
    def standard(cls) -> "HRMConfig":
        """Standard: MHA + SwiGLU (3.4M params)"""
        return cls(
            attention_type=AttentionType.MHA,
            ffn_type=FFNType.SWIGLU,
            hidden_size=256,
            num_heads=8,
        )

    @classmethod
    def dsa(cls) -> "HRMConfig":
        """DSA: DSA(w=7) + SwiGLU (3.4M params)"""
        return cls(
            attention_type=AttentionType.DSA,
            ffn_type=FFNType.SWIGLU,
            dsa_window=7,
            hidden_size=256,
            num_heads=8,
        )

    @classmethod
    def moe(cls) -> "HRMConfig":
        """MoE: MHA + MoE(8 dense) (8.1M params)"""
        return cls(
            attention_type=AttentionType.MHA,
            ffn_type=FFNType.MOE,
            moe_num_experts=8,
            moe_routing="dense",
            hidden_size=256,
            num_heads=8,
        )

    @classmethod
    def dsa_moe(cls) -> "HRMConfig":
        """DSA+MoE: DSA(w=7) + MoE(8 dense) (8.1M params) - Paper Best"""
        return cls(
            attention_type=AttentionType.DSA,
            ffn_type=FFNType.MOE,
            dsa_window=7,
            moe_num_experts=8,
            moe_routing="dense",
            hidden_size=256,
            num_heads=8,
        )

    @classmethod
    def paper_base(cls, num_classes: int = 1000) -> "HRMConfig":
        """Paper base config with D=384, p=16 (Fig 1)"""
        return cls(
            attention_type=AttentionType.DSA,
            ffn_type=FFNType.MOE,
            dsa_window=7,
            moe_num_experts=8,
            moe_routing="dense",
            hidden_size=384,
            num_heads=8,
            image_size=224,
            patch_size=16,
            num_classes=num_classes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        data = asdict(self)
        # Convert enums to strings
        data["attention_type"] = self.attention_type.value
        data["ffn_type"] = self.ffn_type.value
        data["task"] = self.task.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HRMConfig":
        """Create from dictionary (e.g., loaded from YAML)"""
        # Convert string enums back
        if "attention_type" in data and isinstance(data["attention_type"], str):
            data["attention_type"] = AttentionType(data["attention_type"])
        if "ffn_type" in data and isinstance(data["ffn_type"], str):
            data["ffn_type"] = FFNType(data["ffn_type"])
        if "task" in data and isinstance(data["task"], str):
            data["task"] = TaskType(data["task"])
        return cls(**data)

    def to_yaml(self, path: str):
        """Save to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "HRMConfig":
        """Load from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# Convenience presets dict for CLI/config references
PRESET_CONFIGS = {
    "standard": HRMConfig.standard,
    "dsa": HRMConfig.dsa,
    "moe": HRMConfig.moe,
    "dsa_moe": HRMConfig.dsa_moe,
    "paper_base": HRMConfig.paper_base,
}


def get_preset_config(name: str) -> HRMConfig:
    """Get a preset configuration by name."""
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[name]()