import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None) -> None:
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones((d_model,), **kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        factor = (
            torch.sum(x[..., :] ** 2, dim=-1, keepdim=True)
            / self.d_model 
            + self.eps
        )
        out = x / torch.sqrt(factor) * self.weight
        return out.to(in_dtype)


if __name__ == "__main__":
    d_model = 128
    rms = RMSNorm(d_model)

    x = torch.randn((1, d_model))
    print(f"shape of in_features: {x.shape}")

    y = rms(x)
    print(y.shape)