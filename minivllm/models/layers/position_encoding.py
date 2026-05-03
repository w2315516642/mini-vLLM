import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000, device=None, dtype=None) -> None:
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}

        rotation = torch.empty((max_seq_len, d_model, 2), **kwargs)

        thetas = 1.0 / (base ** (torch.arange(0, d_model, 2, **kwargs) / d_model))
        for m in range(max_seq_len):
            m_thetas = m * thetas
            rotation[m, :, 0] = m_thetas.cos().repeat_interleave(2)
            rotation[m, :, 1] = m_thetas.sin().repeat_interleave(2)

        self.register_buffer("rotation", rotation)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        f"""
            x: [batch, ..., seq_len, d_model]
            tp: [batch, ..., seq_len]
        """
        x_cos = x
        x_sin = self.rotate_interleaved(x)
        
        rotation = self.rotation[token_positions]
        out = x_cos * rotation[..., 0] + x_sin * rotation[..., 1]
        return out

    @staticmethod
    def rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        stack = torch.stack((-x_odd, x_even), dim=-1)
        return stack.flatten(-2)

if __name__ == "__main__":
    d_model = 128

    model = RotaryPositionalEmbedding(d_model)

    x = torch.tensor([i for i in range(8)])
    y = model.rotate_interleaved(x)
    print(y)