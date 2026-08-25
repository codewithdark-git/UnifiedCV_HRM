"""
Segmentation Head

Upsamples patch embeddings to full resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    """
    Semantic segmentation head.

    Upsamples from patch-level to pixel-level predictions.
    """

    def __init__(
        self,
        hidden_size: int,
        num_classes: int,
        token_shape: tuple,  # (H, W) or (T, H, W)
        is_video: bool = False,
        img_size: int = 224,
        patch_size: int = 16,
        upsample_mode: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.token_shape = token_shape
        self.is_video = is_video
        self.img_size = img_size
        self.patch_size = patch_size
        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

        if is_video:
            # token_shape = (T, H, W)
            self.T, self.H, self.W = token_shape
            self.conv = nn.Conv3d(hidden_size, num_classes, kernel_size=1)
            self.target_size = (img_size, img_size, img_size)  # Example
        else:
            # token_shape = (H, W)
            self.H, self.W = token_shape
            self.conv = nn.Conv2d(hidden_size, num_classes, kernel_size=1)
            self.target_size = (img_size, img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) - includes CLS token at index 0

        Returns:
            logits: (B, C, H, W) or (B, C, T, H, W)
        """
        B, L, D = x.shape

        # Remove CLS token
        if L > (self.H * self.W * (self.T if self.is_video else 1)):
            x = x[:, 1:]  # Remove CLS

        # Reshape to spatial
        if self.is_video:
            x = x.reshape(B, self.T, self.H, self.W, D)
            x = x.permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
        else:
            x = x.reshape(B, self.H, self.W, D)
            x = x.permute(0, 3, 1, 2)  # (B, D, H, W)

        # 1x1 conv to classes
        logits = self.conv(x)

        # Upsample
        if self.is_video:
            mode = 'trilinear' if self.upsample_mode == 'bilinear' else self.upsample_mode
            logits = F.interpolate(
                logits,
                size=self.target_size,
                mode=mode,
                align_corners=self.align_corners,
            )
        else:
            logits = F.interpolate(
                logits,
                size=self.target_size,
                mode=self.upsample_mode,
                align_corners=self.align_corners,
            )

        return logits