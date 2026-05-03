import math
import torch
from torch import nn
from minivllm.utils import softmax

class ScaleDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor=None
    ) -> torch.Tensor:
        """
            q: [batch, ..., seq_len, d_k]
            k: [batch, ..., seq_len, d_k]
            v: [batch, ..., seq_len, d_v]
            mask: [seq_len, seq_len] | None
        """
        d_k = k.size()[-1]

        k_t = k.transpose(-2, -1)
        score = (q @ k_t) / math.sqrt(d_k)

        if mask is not None:
            score = score.masked_fill(mask == False, -float("inf"))

        score = softmax(score, dim=-1)

        v = score @ v
        return v


if __name__ == "__main__":
    model = ScaleDotProductAttention()

    seq_len = 8
    q = torch.randn((2, 1, seq_len, 64))
    k = torch.randn((2, 1, seq_len, 64))
    v = torch.randn((2, 1, seq_len, 128))
    mask = [[True if x <= i else False for x in range(seq_len)] for i in range(seq_len)]
    mask = torch.tensor(mask)
    print("mask: \n", mask)

    out = model(q, k, v, mask)
    print(out)