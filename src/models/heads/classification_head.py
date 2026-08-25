"""
Classification Head

Uses CLS token for classification.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """
    Classification head using CLS token.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int = 1000,
        dropout: float = 0.0,
        hidden_layers: int = 0,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        if hidden_layers > 0:
            layers = []
            in_dim = hidden_size
            for i in range(hidden_layers):
                out_dim = hidden_dim if hidden_dim else in_dim
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, num_classes))
            self.head = nn.Sequential(*layers)
        else:
            self.head = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) - backbone output with CLS at index 0

        Returns:
            logits: (B, num_classes)
        """
        cls_token = x[:, 0]  # (B, D)
        return self.head(cls_token)