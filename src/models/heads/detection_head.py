"""
Detection Head

Predicts class logits and bounding boxes from CLS token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    """
    Simple detection head.

    Uses CLS token to predict:
    - Class logits for fixed proposals
    - Bbox coordinates (normalized cx, cy, w, h)
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int = 80,
        num_proposals: int = 100,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals

        # Class embeddings
        self.class_embed = nn.Linear(hidden_size, num_proposals * num_classes)

        # Bbox regression
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_proposals * 4),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.class_embed.weight, std=0.02)
        nn.init.zeros_(self.class_embed.bias)
        for m in self.bbox_embed:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, L, D) - backbone output

        Returns:
            cls_logits: (B, num_proposals, num_classes)
            bbox_pred: (B, num_proposals, 4) - sigmoid normalized [0, 1]
        """
        cls_token = x[:, 0]  # (B, D)

        # Class logits
        cls_logits = self.class_embed(cls_token)
        cls_logits = cls_logits.view(cls_logits.size(0), self.num_proposals, self.num_classes)

        # Bbox
        bbox_pred = self.bbox_embed(cls_token)
        bbox_pred = bbox_pred.view(bbox_pred.size(0), self.num_proposals, 4)
        bbox_pred = torch.sigmoid(bbox_pred)  # Normalized to [0, 1]

        return cls_logits, bbox_pred