"""
Models Package - Core I-HRM Architecture

Paper: "I-HRM: Image-Hierarchical Reasoning Model"
"""

from .hrm import (
    HRMStream,
    AdaptiveHaltingModule,
    HierarchicalVisionTransformer,
    TransformerBlock,
    RMSNorm,
)

from .attention import (
    MultiHeadAttention,
    DynamicSparseAttention,
    DeltaAttention,
    create_attention,
)

from .ffn import (
    SwiGLU,
    DenseRoutedMoE,
    DeltaNetFFN,
    create_ffn,
)

from .tokenizer import (
    ImagePatchEmbed,
    VideoTokenizer3D,
    VideoPatchEmbed,
    create_tokenizer,
)

from .heads import (
    ClassificationHead,
    SegmentationHead,
    DetectionHead,
    MultiTaskHead,
)

from .unified_hrm import (
    UnifiedCVHRM,
    UnifiedCVHRMConfig,
)

# Enums from model_config
from src.config.model_config import (
    AttentionType,
    FFNType,
    TaskType,
)

__all__ = [
    # Core HRM
    "HRMStream",
    "AdaptiveHaltingModule",
    "HierarchicalVisionTransformer",
    "TransformerBlock",
    "RMSNorm",
    # Attention variants
    "MultiHeadAttention",
    "DynamicSparseAttention",
    "DeltaAttention",
    "create_attention",
    # FFN variants
    "SwiGLU",
    "DenseRoutedMoE",
    "DeltaNetFFN",
    "create_ffn",
    # Tokenizers
    "ImagePatchEmbed",
    "VideoTokenizer3D",
    "VideoPatchEmbed",
    "create_tokenizer",
    # Heads
    "ClassificationHead",
    "SegmentationHead",
    "DetectionHead",
    "MultiTaskHead",
    # Unified model
    "UnifiedCVHRM",
    "UnifiedCVHRMConfig",
    # Enums
    "AttentionType",
    "FFNType",
    "TaskType",
]