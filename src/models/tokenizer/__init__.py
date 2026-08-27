"""
Tokenizers for I-HRM

1. ImagePatchEmbed - 2D patch embedding (ViT-style)
2. VideoTokenizer3D - 3D CNN tokenizer for video
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImagePatchEmbed(nn.Module):
    """
    2D Patch Embedding (Standard ViT tokenizer)

    Conv2d with kernel=patch_size, stride=patch_size
    Input: (B, C, H, W) → Output: (B, D, H/p, W/p)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        bias: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})"

        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, L, D) where L = H/p * W/p
        return x


class VideoTokenizer3D(nn.Module):
    """
    3D CNN Video Tokenizer (from original code)

    Input: (B, C, T, H, W)
    Output: (B, D, T', H', W') where T'=T/p_t, H'=H/p_h, W'=W/p_w
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        patch_size: tuple = (2, 4, 4),  # (T, H, W)
        frames: int = 8,
        height: int = 64,
        width: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.frames = frames
        self.height = height
        self.width = width

        self.output_shape = (
            frames // patch_size[0],
            height // patch_size[1],
            width // patch_size[2],
        )

        t_p, h_p, w_p = patch_size

        self.conv1 = nn.Conv3d(in_channels, hidden_dim // 4, 3, 1, 1)
        self.conv2 = nn.Conv3d(hidden_dim // 4, hidden_dim // 2, patch_size, stride=patch_size)
        self.conv3 = nn.Conv3d(hidden_dim // 2, hidden_dim, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))  # (B, D/2, T', H', W')
        x = self.conv3(x)  # (B, D, T', H', W')
        return x


class VideoPatchEmbed(nn.Module):
    """
    Alternative: 3D Patch Embedding (like ViViT)
    Uses single 3D conv for tokenization
    """

    def __init__(
        self,
        num_frames: int = 8,
        img_size: int = 224,
        patch_size: tuple = (2, 16, 16),  # (T, H, W)
        in_chans: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        t_p, h_p, w_p = patch_size
        self.grid_size = (
            num_frames // t_p,
            img_size // h_p,
            img_size // w_p,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.proj(x)  # (B, D, T', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)
        return x


def create_tokenizer(
    tokenizer_type: str,
    config,
) -> nn.Module:
    """Factory for creating tokenizers."""
    if tokenizer_type == "image":
        return ImagePatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            embed_dim=config.hidden_size,
        )
    elif tokenizer_type == "video_3d_cnn":
        return VideoTokenizer3D(
            in_channels=config.in_channels,
            hidden_dim=config.hidden_size,
            patch_size=config.video_patch_size,
            frames=config.num_frames,
            height=config.image_size,
            width=config.image_size,
        )
    elif tokenizer_type == "video_patch":
        return VideoPatchEmbed(
            num_frames=config.num_frames,
            img_size=config.image_size,
            patch_size=config.video_patch_size,
            in_chans=config.in_channels,
            embed_dim=config.hidden_size,
        )
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")