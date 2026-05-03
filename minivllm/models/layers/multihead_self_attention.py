import torch
from torch import nn
from .linear import Linear
from .scale_dot_product_attention import ScaleDotProductAttention
from .position_encoding import RotaryPositionalEmbedding as RoPE

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None) -> None:
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}

        self.num_heads = num_heads
        self.attention = ScaleDotProductAttention()
        self.w_qkv = Linear(d_model, 3 * d_model, **kwargs)
        self.w_o = Linear(d_model, d_model, **kwargs)

        # self.rope = RoPE(d_model // num_heads, max_seq_len, theta, **kwargs)


    def forward(self, x: torch.Tensor, rope: RoPE=None, token_positions=None) -> torch.Tensor:
        # x: [batch, ..., seq_len, d_model]
        qkv: torch.Tensor = self.w_qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
        # print(q.shape, k.shape, v.shape)

        # q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # qkv: [batch, ..., seq_len, d_model] -> [batch, ..., seq_len, num_heads, d_k]
        #                                     -> [batch, ..., num_heads, seq_len, d_k]
        qkv_shape = q.size()[:-1]
        
        q_h = q.view(*qkv_shape, self.num_heads, -1).contiguous().transpose(-3, -2)
        k_h = k.view(*qkv_shape, self.num_heads, -1).contiguous().transpose(-3, -2)
        v_h = v.view(*qkv_shape, self.num_heads, -1).contiguous().transpose(-3, -2)

        seq_len = q_h.size()[-2]
        mask = [[True if x <= i else False for x in range(seq_len)] for i in range(seq_len)]
        mask = torch.tensor(mask).to(q.device)

        # rope
        if rope is not None:
            if not token_positions:
                token_positions = torch.arange(0, seq_len, device=q.device)
            q_h = rope(q_h, token_positions)
            k_h = rope(k_h, token_positions)

        # v_h: [batch, ..., num_heads, seq_len, d_k] -> [batch, ..., num_heads, seq_len, d_k]
        new_v_h: torch.Tensor = self.attention(q_h, k_h, v_h, mask)
        # v_h: [batch, ..., num_heads, seq_len, d_k] -> [batch, ..., seq_len, num_heads, d_k]
        #                                            -> [batch, ..., seq_len, num_heads * d_k]
        v_concat = new_v_h.transpose(-3, -2).contiguous().view(*qkv_shape, -1)
        out = self.w_o(v_concat)
        return out

if __name__ == "__main__":
    d_model = 64
    num_heads = 8
    model = MultiHeadAttention(d_model, num_heads)

    x = torch.randn((6, 12, d_model))
    out = model(x)
