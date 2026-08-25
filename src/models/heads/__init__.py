"""
Multi-Task Heads for I-HRM

Combines Classification, Segmentation, and Detection heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .classification_head import ClassificationHead
from .segmentation_head import SegmentationHead
from .detection_head import DetectionHead


class MultiTaskHead(nn.Module):
    """
    Combined multi-task head for I-HRM

    Produces outputs for:
    - Classification (CLS token)
    - Segmentation (spatial upsampling)
    - Detection (CLS token → proposals)
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int = 80,
        image_size: int = 224,
        patch_size: int = 16,
        det_num_proposals: int = 100,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Token shape for segmentation
        self.tokens_per_side = image_size // patch_size
        self.token_shape = (self.tokens_per_side, self.tokens_per_side)

        # Individual heads
        self.classification_head = ClassificationHead(hidden_size, num_classes)
        self.segmentation_head = SegmentationHead(
            hidden_size,
            num_classes,
            token_shape=self.token_shape,
        )
        self.detection_head = DetectionHead(
            hidden_size,
            num_classes,
            num_proposals=det_num_proposals,
        )

    def forward(
        self,
        x: torch.Tensor,
        task: str = "all",
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, L, D) - backbone output
            task: "classification" | "segmentation" | "detection" | "all"

        Returns:
            Dict with task outputs
        """
        outputs = {}

        if task in ["classification", "all"]:
            outputs["classification"] = self.classification_head(x)

        if task in ["segmentation", "all"]:
            outputs["segmentation"] = self.segmentation_head(x)

        if task in ["detection", "all"]:
            outputs["detection"] = self.detection_head(x)

        return outputs