import torch
from torch import nn
from .multihead_self_attention import MultiHeadAttention
from .rmsnorm import RMSNorm
from .position_wise_feed_forward import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, device=None, dtype=None) -> None:
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}

        self.rmsnorm1 = RMSNorm(d_model, **kwargs)
        self.attention = MultiHeadAttention(d_model, num_heads, **kwargs)
        self.rmsnorm2 = RMSNorm(d_model, **kwargs)
        self.swiglu = SwiGLU(d_model, d_ff, **kwargs)

    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        # rms-norm + mha
        y = self.rmsnorm1(x)
        y = self.attention(y, rope)
        x += y
    
        # rms-norm + ffn(swiglu)
        y = self.rmsnorm2(x)
        y = self.swiglu(y)

        return x + y


if __name__ == "__main__":
    d_model = 64
    d_ff = 128
    num_heads = 8
    max_seq_len = 1024
    theta = 10000

    model = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)

    x = torch.randn((6, 12, d_model))
    y = model(x)
    print(y.shape)