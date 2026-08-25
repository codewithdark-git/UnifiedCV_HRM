"""
FFN Variants for I-HRM

1. SwiGLU - Standard gated FFN
2. DenseRoutedMoE - Dense-routed Mixture of Experts (prevents collapse)
3. DeltaNetFFN - Convolutional DeltaNet variant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network"""

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
    Dense-Routed Mixture-of-Experts

    Key properties (Paper):
    - Dense routing: All experts get gradient for every token (prevents collapse)
    - Softmax routing weights sum to 1 per token
    - No auxiliary load balancing loss needed (dense routing naturally balances)
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        expansion: float = 4.0,
        bias: bool = False,
        expert_type: str = "swiglu",
    ):
        super().__init__()
        self.num_experts = num_experts
        hidden = int(dim * expansion)

        # Router produces dense weights per token
        self.router = nn.Linear(dim, num_experts, bias=bias)

        # Expert networks
        self.experts = nn.ModuleList([
            SwiGLU(dim, expansion) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x: (B, L, D) input tokens

        Returns:
            output: (B, L, D)
            aux: dict with router_logits, router_weights, entropy, etc.
        """
        B, L, D = x.shape

        # Router logits and weights
        logits = self.router(x)  # (B, L, E)
        weights = F.softmax(logits, dim=-1)  # Dense routing, sum to 1

        # Expert outputs
        expert_outs = torch.stack([exp(x) for exp in self.experts], dim=-1)  # (B, L, D, E)

        # Weighted combination
        out = (expert_outs * weights.unsqueeze(-2)).sum(dim=-1)  # (B, L, D)

        # Auxiliary info for monitoring/analysis
        entropy = -(weights * weights.log()).sum(-1).mean()

        aux = {
            "router_logits": logits,
            "router_weights": weights,
            "entropy": entropy,
            "expert_usage": weights.mean(dim=(0, 1)),  # (E,) - how much each expert used
            "max_weight": weights.max().item(),
            "min_weight": weights.min().item(),
        }

        return out, aux


class DeltaNetFFN(nn.Module):
    """
    Convolutional DeltaNet FFN

    Uses depthwise separable conv for local mixing + global gating.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        kernel_size: int = 3,
        activation: str = "silu",
    ):
        super().__init__()
        hidden = int(dim * expansion)

        self.proj_up = nn.Linear(dim, hidden * 2)
        self.dwconv = nn.Conv1d(
            hidden, hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden,
        )
        self.proj_down = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape

        x_up = self.proj_up(x)  # (B, L, 2*hidden)
        gate, value = x_up.chunk(2, dim=-1)  # (B, L, hidden)

        # Depthwise conv on value
        value = value.transpose(1, 2)  # (B, hidden, L)
        value = self.dwconv(value)
        value = value.transpose(1, 2)  # (B, L, hidden)

        # Gated output
        if hasattr(F, 'silu'):
            gate_act = F.silu(gate)
        else:
            gate_act = torch.sigmoid(gate)  # fallback

        out = gate_act * value
        return self.proj_down(out)


def create_ffn(
    ffn_type: str,
    dim: int,
    **kwargs,
) -> nn.Module:
    """Factory function for FFN modules."""
    if ffn_type == "swiglu":
        return SwiGLU(dim, **kwargs)
    elif ffn_type == "moe":
        return DenseRoutedMoE(dim, **kwargs)
    elif ffn_type == "delta_net":
        return DeltaNetFFN(dim, **kwargs)
    else:
        raise ValueError(f"Unknown FFN type: {ffn_type}")