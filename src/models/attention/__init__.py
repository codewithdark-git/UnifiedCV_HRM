"""
Attention Variants for I-HRM

1. MultiHeadAttention - Standard MHA
2. DynamicSparseAttention - DSA with window size w=7
3. DeltaAttention - L2-Normalized Delta Attention
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA)"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, float('-inf'))

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class DynamicSparseAttention(nn.Module):
    """
    Dynamic Sparse Attention (DSA) - Paper Section 3.3

    Windowed attention with window size w=7.
    Each token attends to local w×w window.
    Theoretical FLOPs: O(L × w²) vs O(L²) for full attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.causal = causal
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D) where L = H × W (spatial tokens, no CLS)
        spatial_shape: (H, W) for window computation
        """
        B, L, D = x.shape

        if spatial_shape is None:
            H = W = int(math.sqrt(L))
        else:
            H, W = spatial_shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Reshape to spatial: (B, H, H_grid, W_grid, D)
        q = q.reshape(B, self.num_heads, H, W, self.head_dim)
        k = k.reshape(B, self.num_heads, H, W, self.head_dim)
        v = v.reshape(B, self.num_heads, H, W, self.head_dim)

        # Extract local windows
        w = self.window_size
        pad = w // 2

        # Pad spatial dims
        k_pad = F.pad(k, (0, 0, pad, pad, pad, pad), value=0)
        v_pad = F.pad(v, (0, 0, pad, pad, pad, pad), value=0)

        # Unfold: (B, H, H, W, w, w, D)
        k_windows = k_pad.unfold(2, w, 1).unfold(3, w, 1)
        v_windows = v_pad.unfold(2, w, 1).unfold(3, w, 1)

        # Flatten windows: (B, H, H*W, w*w, D)
        k_windows = k_windows.reshape(B, self.num_heads, H * W, w * w, self.head_dim)
        v_windows = v_windows.reshape(B, self.num_heads, H * W, w * w, self.head_dim)

        # Query: (B, H, H*W, 1, D)
        q_flat = q.reshape(B, self.num_heads, H * W, 1, self.head_dim)

        # Windowed attention
        attn = (q_flat @ k_windows.transpose(-2, -1)) * self.scale

        if self.causal:
            # Not typically used for vision
            pass

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output: (B, H, H*W, D)
        out = (attn @ v_windows).squeeze(-2)

        # Reshape back: (B, H, H, W, D) → (B, L, D)
        out = out.reshape(B, self.num_heads, H, W, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, H * W, D)

        return self.proj(out)


class DeltaAttention(nn.Module):
    """
    L2-Normalized Delta Attention (Optional variant)

    Uses L2-normalized Q/K for stability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # L2 Normalize
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


def create_attention(
    attention_type: str,
    dim: int,
    num_heads: int,
    **kwargs,
) -> nn.Module:
    """Factory function for attention modules."""
    if attention_type == "mha":
        return MultiHeadAttention(dim, num_heads, **kwargs)
    elif attention_type == "dsa":
        return DynamicSparseAttention(dim, num_heads, **kwargs)
    elif attention_type == "delta":
        return DeltaAttention(dim, num_heads, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")