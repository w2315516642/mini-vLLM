import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int,
        device: torch.device=None, 
        dtype: torch.dtype=None
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        kwargs = {"device": device, "dtype": dtype}

        self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), **kwargs)
            )
        self.weight = nn.init.trunc_normal_(self.weight, a=-3, b=3)
            

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, seq_len]
        # pytorch会自动抽取对应行，返回一个[batch, seq_len, embedding_dim]的向量
        out = self.weight[token_ids]
        return out

if __name__ == "__main__":
    num_embeddings = 1024
    embedding_dim = 512
    seq_len = 100
    batch_size = 64

    assert seq_len <= num_embeddings

    weights = torch.randn((num_embeddings, embedding_dim))

    model = Embedding(num_embeddings, embedding_dim, weights)

    model_ = nn.Embedding(num_embeddings, embedding_dim)

    x = torch.randint(0, num_embeddings - 1, (batch_size, seq_len))
    y: torch.Tensor = model(x)
    y_ = model_(x)

    estimate_size = batch_size * seq_len * embedding_dim * 4 / 1024
    actual_size = y.nelement() * y.element_size() / 1024

    print(f"std1: {torch.std(y)}, std2: {torch.std(y_)}")
    print(f"output shape: {y.shape}")
    print(f"size of output: {actual_size}/{estimate_size} kb")
    print(f"data type of out: {y.dtype}")