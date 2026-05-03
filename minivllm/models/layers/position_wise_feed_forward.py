import torch
from torch import nn
from .linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None) -> None:
        super().__init__()

        kwargs = {"device": device, "dtype": dtype}
        
        self.weight1 = Linear(d_model, d_ff, **kwargs)
        self.weight2 = Linear(d_ff, d_model, **kwargs)
        self.weight3 = Linear(d_model, d_ff, **kwargs)

        # self.weight1 = nn.Parameter(torch.empty((d_ff, d_model), **kwargs))
        # self.weight2 = nn.Parameter(torch.empty((d_model, d_ff), **kwargs))
        # self.weight3 = nn.Parameter(torch.empty((d_ff, d_model), **kwargs))

        # sigma = torch.sqrt(torch.tensor(2 / (d_model + d_ff)))
        # self.weight1 = nn.init.trunc_normal_(self.weight1, std=sigma, a=-3 * sigma, b=3 * sigma)
        # self.weight2 = nn.init.trunc_normal_(self.weight2, std=sigma, a=-3 * sigma, b=3 * sigma)
        # self.weight3 = nn.init.trunc_normal_(self.weight3, std=sigma, a=-3 * sigma, b=3 * sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.Swish(self.weight1(x))
        v = self.weight3(x)
        out = self.weight2(y * v)
        return out
    
    @staticmethod
    def Swish(x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        out = x * torch.sigmoid(x)
        return out.to(in_dtype)


if __name__ == "__main__":
    d_model = 64
    d_ff = 128
    model = SwiGLU(d_model, d_ff)
    w1 = torch.randn((d_model, d_ff))
    model.weight1.weight.data = w1
    x = torch.randn((6, 12, d_model), dtype=torch.float16)

    y = model(x)
    
    