
import torch
import torch.nn as nn

from minivllm import layernorm_ops


class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        layernorm_ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon
        )
        return out


class Qwen3_5RMSNorm(nn.Module):
    """Per-head RMSNorm used by Qwen3.5/3.8 attention.

    Qwen checkpoints store a zero-centered scale. The effective multiplier is
    ``1 + weight``, unlike :class:`RMSNorm`, whose stored weight is used
    directly.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        var = torch.mean(x_fp32 ** 2, dim=-1, keepdim=True)
        normalized = x_fp32 * torch.rsqrt(var + self.variance_epsilon)
        out = (1 + self.weight.float()) * normalized
        return out.to(dtype=input_dtype)
