"""
Unified CV-HRM: HF-Compatible Model Wrapper

HF-compatible PreTrainedModel for I-HRM with:
- Multi-task support (classification, segmentation, detection)
- push_to_hub / from_pretrained integration
- Trainer compatibility
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    ImageClassifierOutput,
    BaseModelOutput,
)

from .hrm import (
    HierarchicalVisionTransformer,
)
from .heads import (
    MultiTaskHead,
)
from src.config.model_config import (
    HRMConfig,
    AttentionType,
    FFNType,
    TaskType,
)


# ─────────────────────────────────────────────────────────────────
# HF Config
# ─────────────────────────────────────────────────────────────────

class UnifiedCVHRMConfig(PretrainedConfig):
    """HF Configuration for UnifiedCVHRM"""

    model_type = "unified_cv_hrm"
    is_composition = False

    def __init__(
        self,
        # Architecture
        hidden_size: int = 384,
        num_heads: int = 8,
        num_layers_per_stream: int = 2,
        max_steps: int = 6,
        halt_threshold: float = 0.5,
        halt_loss_weight: float = 0.1,
        attention_type: str = "dsa",
        dsa_window: int = 7,
        ffn_type: str = "moe",
        ffn_expansion: float = 4.0,
        moe_num_experts: int = 8,
        moe_routing: str = "dense",
        # Input
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        use_cls_token: bool = True,
        use_pos_embed: bool = True,
        # Video
        num_frames: int = 8,
        video_patch_size: tuple = (2, 4, 4),
        # Tasks
        task: str = "classification",
        num_classes: int = 1000,
        seg_num_classes: int = 80,
        det_num_classes: int = 80,
        det_num_proposals: int = 100,
        # Init
        init_std: float = 0.02,
        norm_eps: float = 1e-5,
        use_rmsnorm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Architecture
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers_per_stream = num_layers_per_stream
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        self.halt_loss_weight = halt_loss_weight
        self.attention_type = attention_type
        self.dsa_window = dsa_window
        self.ffn_type = ffn_type
        self.ffn_expansion = ffn_expansion
        self.moe_num_experts = moe_num_experts
        self.moe_routing = moe_routing

        # Input
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.use_cls_token = use_cls_token
        self.use_pos_embed = use_pos_embed

        # Video
        self.num_frames = num_frames
        self.video_patch_size = video_patch_size

        # Tasks
        self.task = task
        self.num_classes = num_classes
        self.seg_num_classes = seg_num_classes
        self.det_num_classes = det_num_classes
        self.det_num_proposals = det_num_proposals

        # Init
        self.init_std = init_std
        self.norm_eps = norm_eps
        self.use_rmsnorm = use_rmsnorm

    def to_local_config(self) -> HRMConfig:
        """Convert to local HRMConfig."""
        return HRMConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers_per_stream=self.num_layers_per_stream,
            max_steps=self.max_steps,
            halt_threshold=self.halt_threshold,
            halt_loss_weight=self.halt_loss_weight,
            attention_type=AttentionType(self.attention_type),
            dsa_window=self.dsa_window,
            ffn_type=FFNType(self.ffn_type),
            ffn_expansion=self.ffn_expansion,
            moe_num_experts=self.moe_num_experts,
            moe_routing=self.moe_routing,
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            use_cls_token=self.use_cls_token,
            use_pos_embed=self.use_pos_embed,
            num_frames=self.num_frames,
            video_patch_size=self.video_patch_size,
            task=TaskType(self.task),
            num_classes=self.num_classes,
            seg_num_classes=self.seg_num_classes,
            det_num_classes=self.det_num_classes,
            det_num_proposals=self.det_num_proposals,
            init_std=self.init_std,
            norm_eps=self.norm_eps,
            use_rmsnorm=self.use_rmsnorm,
        )

    @classmethod
    def from_local_config(cls, local_config: HRMConfig) -> "UnifiedCVHRMConfig":
        """Create HF config from local config."""
        return cls(
            hidden_size=local_config.hidden_size,
            num_heads=local_config.num_heads,
            num_layers_per_stream=local_config.num_layers_per_stream,
            max_steps=local_config.max_steps,
            halt_threshold=local_config.halt_threshold,
            halt_loss_weight=local_config.halt_loss_weight,
            attention_type=local_config.attention_type.value,
            dsa_window=local_config.dsa_window,
            ffn_type=local_config.ffn_type.value,
            ffn_expansion=local_config.ffn_expansion,
            moe_num_experts=local_config.moe_num_experts,
            moe_routing=local_config.moe_routing,
            image_size=local_config.image_size,
            patch_size=local_config.patch_size,
            in_channels=local_config.in_channels,
            use_cls_token=local_config.use_cls_token,
            use_pos_embed=local_config.use_pos_embed,
            num_frames=local_config.num_frames,
            video_patch_size=local_config.video_patch_size,
            task=local_config.task.value,
            num_classes=local_config.num_classes,
            seg_num_classes=local_config.seg_num_classes,
            det_num_classes=local_config.det_num_classes,
            det_num_proposals=local_config.det_num_proposals,
            init_std=local_config.init_std,
            norm_eps=local_config.norm_eps,
            use_rmsnorm=local_config.use_rmsnorm,
        )


# ─────────────────────────────────────────────────────────────────
# HF Model
# ─────────────────────────────────────────────────────────────────

class UnifiedCVHRM(PreTrainedModel):
    """
    Unified CV-HRM Model (HF Compatible)

    Wraps I-HRM backbone with multi-task heads.
    Supports push_to_hub, from_pretrained, Trainer.
    """

    config_class = UnifiedCVHRMConfig
    base_model_prefix = "ihrm"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HierarchicalVisionTransformer", "HRMLayer", "TransformerBlock"]

    def __init__(self, config: UnifiedCVHRMConfig):
        super().__init__(config)
        self.config = config

        # Build backbone using local config
        local_config = config.to_local_config()
        self.backbone = HierarchicalVisionTransformer(local_config)

        # Task heads
        self.heads = MultiTaskHead(
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
            image_size=config.image_size,
            patch_size=config.patch_size,
            det_num_proposals=config.det_num_proposals,
        )

        # Initialize weights
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        task: str = "classification",
        labels: Optional[torch.Tensor] = None,
        return_steps: bool = False,
        **kwargs,
    ):
        """
        Args:
            pixel_values: (B, C, H, W) or (B, C, T, H, W)
            task: "classification" | "segmentation" | "detection" | "all"
            labels: Ground truth labels
            return_steps: Whether to return reasoning steps

        Returns:
            Task-specific output matching HF format
        """
        # Backbone forward
        features, steps = self.backbone(pixel_values, return_steps=True)

        if task == "classification":
            logits = self.heads.classification_head(features)

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)

            return ImageClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=(features,) if return_steps else None,
            )

        elif task == "segmentation":
            logits = self.heads.segmentation_head(features)

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels, ignore_index=255)

            return BaseModelOutput(
                loss=loss,
                logits=logits,
                hidden_states=(features,) if return_steps else None,
            )

        elif task == "detection":
            cls_logits, bbox_pred = self.heads.detection_head(features)

            loss = None
            if labels is not None:
                cls_t, bbox_t = labels
                cls_loss = nn.functional.cross_entropy(cls_logits, cls_t)
                bbox_loss = nn.functional.smooth_l1_loss(bbox_pred, bbox_t)
                loss = cls_loss + bbox_loss

            return {
                "loss": loss,
                "cls_logits": cls_logits,
                "bbox_pred": bbox_pred,
                "hidden_states": features if return_steps else None,
            }

        elif task == "all":
            outputs = self.heads(features, task="all")
            if return_steps:
                outputs["reasoning_steps"] = steps
            return outputs

        else:
            raise ValueError(f"Unknown task: {task}")

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        task: str = "classification",
        top_k: int = 5,
    ):
        """Inference generation."""
        self.eval()
        outputs = self.forward(pixel_values, task=task)

        if task == "classification":
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k, dim=-1)
            return {"labels": top_indices.cpu().tolist(), "scores": top_probs.cpu().tolist()}
        elif task == "segmentation":
            logits = outputs.logits
            pred = logits.argmax(dim=1)
            return {"segmentation": pred.cpu().numpy()}
        elif task == "detection":
            cls_logits, bbox_pred = outputs["cls_logits"], outputs["bbox_pred"]
            probs = torch.softmax(cls_logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k, dim=-1)
            return {
                "labels": top_indices.cpu().tolist(),
                "scores": top_probs.cpu().tolist(),
                "boxes": bbox_pred.cpu().tolist(),
            }
        return {}

    def get_backbone(self):
        """Access backbone for feature extraction."""
        return self.backbone

    def get_heads(self):
        """Access task heads."""
        return self.heads


# ─────────────────────────────────────────────────────────────────
# Register with HF Auto classes
# ─────────────────────────────────────────────────────────────────

from transformers import AutoConfig, AutoModel

AutoConfig.register("unified_cv_hrm", UnifiedCVHRMConfig)
AutoModel.register(UnifiedCVHRMConfig, UnifiedCVHRM)

# Also register for custom model loading
from transformers import AutoModelForImageClassification
AutoModelForImageClassification.register(UnifiedCVHRMConfig, UnifiedCVHRM)