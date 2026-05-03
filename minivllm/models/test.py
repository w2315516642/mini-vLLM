import torch
from torch import nn
from typing import List

from .layers.embedding import Embedding
from .layers.transformer_block import TransformerBlock
from .layers.position_encoding import RotaryPositionalEmbedding as RoPE
from .layers.linear import Linear
from .layers.rmsnorm import RMSNorm
from minivllm.utils import softmax, top_p_filter

from minivllm.configs.test_model_config import TestConfig as Config


class TestModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: int,
        device: torch.device=None,
        dtype: torch.dtype=None
    ) -> None:
        super().__init__()

        self.context_length = context_length

        kwargs = {"device": device, "dtype": dtype}

        self.embedding = Embedding(vocab_size, d_model, **kwargs)
        self.rope = RoPE(d_model // num_heads, context_length, theta, **kwargs)
        
        self.transformers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, **kwargs)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model, **kwargs)
        self.linear = Linear(d_model, vocab_size, **kwargs)
    
    def forward(self, token_ids: torch.Tensor | List[int]) -> torch.Tensor:
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids) 
        with record_function_or_nullcontext("embedding"):
            # embedding
            x = self.embedding(token_ids)
        
        with record_function_or_nullcontext("Transformer block forward"):
            # forward
            for layer in self.transformers:
                x = layer(x, self.rope)

        with record_function_or_nullcontext("LM head"):
            x = self.norm(x)
            x = self.linear(x)
            "注意这里不需要 softmax"
        return x
    
    @torch.no_grad()
    def generate(
        self, 
        prompt_tokens: List[int], 
        temperature: float = 1,
        top_p: float | None = None,
        max_len: int | None = None,
        eos_id: int = 0,
        eps: float = 1e-6
    ) -> List[int]:
        self.eval()
        
        device = self.linear.weight.device
        if max_len is None:
            max_len = self.context_length
        total_tokens = torch.tensor(prompt_tokens, device=device)
        for _ in range(max_len):
            # 防止超出训练时的最大上下文长度
            with record_function_or_nullcontext("tensor cat"):
                in_tokens = (
                    total_tokens
                    if total_tokens.size()[-1] <= self.context_length 
                    else total_tokens[..., -self.context_length:]
                )
            with record_function_or_nullcontext("forward"):
                # (batch_size, seq_len, vocab_size)
                logits = self(in_tokens)[:, -1, :]
            
            if temperature < eps:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                with record_function_or_nullcontext("top-p"):
                    logits = top_p_filter(logits, top_p)     
                with record_function_or_nullcontext("softmax"):
                    probs = softmax(logits)
                with record_function_or_nullcontext("sample"):
                    next_token = torch.multinomial(probs, num_samples=1)
            
            total_tokens = torch.cat((total_tokens, next_token), dim=-1)

            if next_token == eos_id:
                break
        
        return total_tokens[..., len(prompt_tokens[0]):].tolist()

    @classmethod
    def from_config(cls, config: Config):
        model_config = config.model
        return cls(
            vocab_size=model_config.vocab_size,
            context_length=model_config.context_length,
            num_layers=model_config.num_layers,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
            theta=model_config.theta,
            device=config.device,
            dtype=config.dtype
        )