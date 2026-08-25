"""
Core HRM Architecture

Implements the Hierarchical Reasoning Module with:
- Dual-stream: H-layer (high-level semantic) ↔ L-layer (low-level perceptual)
- Adaptive Halting (Algorithm 1, corrected)
- Input injection at each step
"""

import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.model_config import HRMConfig
from src.models.tokenizer import ImagePatchEmbed


# ─────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (paper uses this)"""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D)
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return x / (norm + self.eps) * self.weight


# ─────────────────────────────────────────────────────────────────
# Feed-Forward Networks
# ─────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (used in Standard and DSA variants)"""

    def __init__(self, dim: int, expansion: float = 4.0, bias: bool = False):
        super().__init__()
        hidden = int(dim * expansion)
        self.w1 = nn.Linear(dim, hidden, bias=bias)
        self.w2 = nn.Linear(dim, hidden, bias=bias)
        self.w3 = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class DenseRoutedMoE(nn.Module):
    """
    Dense-Routed Mixture-of-Experts (Paper: prevents expert collapse)

    All experts receive gradients for every token (dense routing),
    but weights determine contribution. Prevents collapse seen in Top-K routing.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        expansion: float = 4.0,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        hidden = int(dim * expansion)

        # Router: produces dense weights for all experts per token
        self.router = nn.Linear(dim, num_experts, bias=bias)

        # Experts (shared across tokens)
        self.experts = nn.ModuleList([
            SwiGLU(dim, expansion) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape

        # Router logits: (B, L, E)
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)  # Dense weights, sum to 1

        # Expert outputs: (B, L, D, E)
        expert_outs = torch.stack([exp(x) for exp in self.experts], dim=-1)

        # Weighted sum: (B, L, D) = sum_E (B, L, D, E) * (B, L, 1, E)
        out = (expert_outs * weights.unsqueeze(-2)).sum(dim=-1)

        # Return output + aux info for monitoring
        return out, {
            "router_logits": logits,
            "router_weights": weights,
            "entropy": -(weights * weights.log()).sum(-1).mean(),
        }


class DeltaNetFFN(nn.Module):
    """
    Convolutional DeltaNet Feed-Forward (Optional variant)

    Uses depthwise separable convolutions for local mixing + global gating.
    """

    def __init__(self, dim: int, expansion: float = 2.0, kernel_size: int = 3):
        super().__init__()
        hidden = int(dim * expansion)

        self.proj_up = nn.Linear(dim, hidden * 2)
        self.dwconv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden)
        self.proj_down = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        x_up = self.proj_up(x)  # (B, L, 2*hidden)
        gate, value = x_up.chunk(2, dim=-1)  # (B, L, hidden)

        # Depthwise conv on value (needs (B, hidden, L))
        value = value.transpose(1, 2)  # (B, hidden, L)
        value = self.dwconv(value)
        value = value.transpose(1, 2)  # (B, L, hidden)

        # Gated output
        out = F.silu(gate) * value
        return self.proj_down(out)


# ─────────────────────────────────────────────────────────────────
# Attention Variants
# ─────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA)"""

    def __init__(self, dim: int, num_heads: int, causal: bool = False, dropout: float = 0.0):
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    Windowed attention with dynamic window size w=7.
    Each token attends to local neighborhood + learns to attend globally.
    Theoretical FLOPs: O(L * w^2) vs O(L^2) for full attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,           # Paper: w=7
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

        # Learned global token indices (optional, paper doesn't specify)
        self.register_buffer("global_indices", torch.arange(0, dim, dim // num_heads))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        x: (B, L, D) where L = H * W (spatial tokens, no CLS)
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

        # Reshape to spatial: (B, H, L, D) → (B, H, H_grid, W_grid, D)
        q = q.reshape(B, self.num_heads, H, W, self.head_dim)
        k = k.reshape(B, self.num_heads, H, W, self.head_dim)
        v = v.reshape(B, self.num_heads, H, W, self.head_dim)

        # Extract local windows around each position
        w = self.window_size
        pad = w // 2

        # Pad spatial dims
        k_pad = F.pad(k, (0, 0, pad, pad, pad, pad), value=0)  # (B, H, H+2p, W+2p, D)
        v_pad = F.pad(v, (0, 0, pad, pad, pad, pad), value=0)

        # Unfold windows: (B, H, H, W, w, w, D)
        k_windows = k_pad.unfold(2, w, 1).unfold(3, w, 1)  # (B, H, H, W, w, w, D)
        v_windows = v_pad.unfold(2, w, 1).unfold(3, w, 1)

        # Compute attention per window
        # q: (B, H, H, W, D) → (B, H, H*W, 1, D)
        q_flat = q.reshape(B, self.num_heads, H * W, 1, self.head_dim)

        # k_windows: (B, H, H, W, w, w, D) → (B, H, H*W, w*w, D)
        k_windows = k_windows.reshape(B, self.num_heads, H * W, w * w, self.head_dim)
        v_windows = v_windows.reshape(B, self.num_heads, H * W, w * w, self.head_dim)

        # Attention scores: (B, H, H*W, 1, w*w)
        attn = (q_flat @ k_windows.transpose(-2, -1)) * self.scale

        if self.causal:
            # Causal mask within window (not typically used for vision)
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

    Uses L2-normalized queries/keys for stability, with delta-rule update.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
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

        # L2 Normalize queries and keys
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


# ─────────────────────────────────────────────────────────────────
# Transformer Block (used in both H and L streams)
# ─────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Standard Transformer block with Post-Norm (Paper uses RMSNorm)

    Order: Attention → Residual → RMSNorm → FFN → Residual → RMSNorm
    """

    def __init__(
        self,
        config: HRMConfig,
        attention: Optional[nn.Module] = None,
        ffn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config

        # Attention
        if attention is None:
            if config.attention_type == "mha":
                attention = MultiHeadAttention(
                    config.hidden_size, config.num_heads, config.causal
                )
            elif config.attention_type == "dsa":
                attention = DynamicSparseAttention(
                    config.hidden_size, config.num_heads, config.dsa_window
                )
            elif config.attention_type == "delta":
                attention = DeltaAttention(config.hidden_size, config.num_heads)
            else:
                raise ValueError(f"Unknown attention_type: {config.attention_type}")
        self.attention = attention

        # FFN
        if ffn is None:
            if config.ffn_type == "swiglu":
                ffn = SwiGLU(config.hidden_size, config.ffn_expansion)
            elif config.ffn_type == "moe":
                ffn = DenseRoutedMoE(
                    config.hidden_size,
                    config.moe_num_experts,
                    config.ffn_expansion,
                )
            elif config.ffn_type == "delta_net":
                ffn = DeltaNetFFN(config.hidden_size, config.ffn_expansion)
            else:
                raise ValueError(f"Unknown ffn_type: {config.ffn_type}")
        self.ffn = ffn

        # Norms
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        # Handle CLS token for DSA - DSA only works on spatial tokens (no CLS)
        # For MHA/Delta, we keep CLS token and pass full sequence
        if spatial_shape is not None and self.config.attention_type == "dsa":
            # Split CLS token and spatial tokens
            cls_token = x[:, :1]  # (B, 1, D)
            x_spatial = x[:, 1:]  # (B, H*W, D)
            input_to_attn = x_spatial
        else:
            cls_token = None
            input_to_attn = x

        # Attention + Residual + Norm
        if hasattr(self.attention, 'forward') and 'spatial_shape' in self.attention.forward.__code__.co_varnames:
            attn_out = self.attention(input_to_attn, spatial_shape=spatial_shape)
        else:
            attn_out = self.attention(input_to_attn)

        # Re-attach CLS token if it was split
        if cls_token is not None:
            attn_out = torch.cat([cls_token, attn_out], dim=1)

        x = self.norm1(x + attn_out)

        # FFN + Residual + Norm
        ffn_out = self.ffn(x)
        if isinstance(ffn_out, tuple):  # MoE returns (output, aux_info)
            ffn_out = ffn_out[0]
        x = self.norm2(x + ffn_out)

        return x


# ─────────────────────────────────────────────────────────────────
# HRM Streams (H and L)
# ─────────────────────────────────────────────────────────────────

class HRMStream(nn.Module):
    """
    Single HRM stream (either High-level or Low-level).

    Contains `num_layers_per_stream` TransformerBlocks.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers_per_stream)
        ])

    def forward(
        self,
        x: torch.Tensor,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, spatial_shape)
        return x


# ─────────────────────────────────────────────────────────────────
# Adaptive Halting Module (Algorithm 1 - Corrected)
# ─────────────────────────────────────────────────────────────────

class AdaptiveHaltingModule(nn.Module):
    """
    Adaptive Halting Mechanism (Corrected Algorithm 1)

    Paper Algorithm 1 fixed:
    - Single exit point with break
    - K assigned once per path
    - Handles max_steps case
    - Explicit halting probability computation
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.halt_head = nn.Linear(config.hidden_size, 1)
        self.threshold = config.halt_threshold
        self.max_steps = config.max_steps

    def forward(
        self,
        z_H: torch.Tensor,
        step: int,
    ) -> Tuple[bool, torch.Tensor, int]:
        """
        Check if should halt.

        Args:
            z_H: High-level state (B, L, D)
            step: Current step index (0-indexed)

        Returns:
            should_halt: bool
            p_halt: Halting probabilities per sample (B,)
            K: Number of steps executed (step + 1 if halting)
        """
        # Mean pool over sequence: (B, L, D) → (B, D)
        z_H_mean = z_H.mean(dim=1)

        # Halting probability
        p_halt = torch.sigmoid(self.halt_head(z_H_mean)).squeeze(-1)  # (B,)

        # Check halting condition (after at least one full H↔L cycle)
        if step >= 1:
            should_halt = (p_halt > self.threshold).all()
            if should_halt:
                return True, p_halt, step + 1

        return False, p_halt, step + 1


# ─────────────────────────────────────────────────────────────────
# Hierarchical Vision Transformer (Main Backbone)
# ─────────────────────────────────────────────────────────────────

class HierarchicalVisionTransformer(nn.Module):
    """
    I-HRM Backbone: Hierarchical Vision Transformer with Adaptive Halting

    Architecture (Fig 1, Sec 3.1):
    1. Tokenizer (ImagePatchEmbed or VideoTokenizer3D)
    2. CLS token + PosEmbed
    3. Dual-stream recurrent processing (H-layer ↔ L-layer)
    4. Adaptive halting
    5. Output: final z_H + reasoning steps K

    Adaptive Halting (Algorithm 1):
    - Initialize z_H, z_L from Gaussian + learnable buffer I_b
    - For k=0 to K_max-1:
        input_L = z_H + S0
        z_L = L_stream(z_L, input_L)
        z_H = H_stream(z_H, z_L)
        if k >= 1 and p_halt > τ: halt
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        # Tokenizer
        self.tokenizer = ImagePatchEmbed(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_chans=config.in_channels,
            embed_dim=config.hidden_size,
        )

        # Positional embeddings + CLS token
        self.num_patches = config.num_patches
        self.seq_len = config.seq_len

        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        if config.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, config.hidden_size))

        # Learnable initialization buffer I_b
        self.register_buffer(
            "I_b",
            torch.randn(1, self.seq_len, config.hidden_size) * config.init_std
        )

        # Dual streams
        self.H_stream = HRMStream(config)
        self.L_stream = HRMStream(config)

        # Adaptive halting
        self.halting = AdaptiveHaltingModule(config)

        # Output norm
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)

        # Classification head (if num_classes specified)
        if config.num_classes > 0:
            self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        else:
            self.classifier = None

        # Initialize
        self._init_weights()

    def _init_weights(self):
        if self.config.use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=self.config.init_std)
        if self.config.use_pos_embed:
            nn.init.trunc_normal_(self.pos_embed, std=self.config.init_std)

    def forward(
        self,
        x: torch.Tensor,
        return_steps: bool = False,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Args:
            x: Input images (B, C, H, W) or video (B, C, T, H, W)
            return_steps: If True, return (features, steps_info)
            return_logits: If True, return classification logits instead of features

        Returns:
            features: (B, L, D) - final high-level representations (by default)
            logits: (B, num_classes) - if return_logits=True and classifier exists
            steps_info: dict with halt_probs, halt_steps (if return_steps=True)
        """
        B = x.shape[0]

        # Tokenize: (B, C, H, W) → (B, L, D)
        x = self.tokenizer(x)  # (B, D, H/p, W/p) → flatten to (B, L, D)

        # Add CLS token
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        if self.config.use_pos_embed:
            x = x + self.pos_embed

        # Persistent input S0 (re-injected each step)
        S0 = x

        # Initialize recurrent states from I_b
        z_H = torch.randn_like(S0) * self.config.init_std + self.I_b
        z_L = torch.randn_like(S0) * self.config.init_std + self.I_b

        # Determine spatial shape for DSA
        spatial_shape = None
        if self.config.attention_type == "dsa":
            h = w = int(math.sqrt(self.config.num_patches))
            spatial_shape = (h, w)

        # Hierarchical reasoning loop (Algorithm 1)
        K = 0
        halt_probs_list = []
        for k in range(self.config.max_steps):
            # L-stream: perceptual refinement (bottom-up)
            input_L = z_H + S0
            z_L = self.L_stream(input_L, spatial_shape)

            # H-stream: semantic integration (top-down)
            z_H = self.H_stream(z_H + z_L, spatial_shape)

            # Halting check
            should_halt, p_halt, K = self.halting(z_H, k)
            halt_probs_list.append(p_halt)
            if should_halt:
                break

        # Final norm
        z_H = self.norm(z_H)

        # Optionally compute logits
        if return_logits and self.classifier is not None:
            if self.config.use_cls_token:
                cls_features = z_H[:, 0]  # (B, D)
                output = self.classifier(cls_features)  # (B, num_classes)
            else:
                pooled = z_H.mean(dim=1)  # (B, D)
                output = self.classifier(pooled)  # (B, num_classes)
        else:
            output = z_H  # (B, L, D) - return features

        if return_steps:
            steps_info = {
                "halt_probs": torch.stack(halt_probs_list, dim=1) if halt_probs_list else None,
                "halt_steps": K,
            }
            return output, steps_info
        return output