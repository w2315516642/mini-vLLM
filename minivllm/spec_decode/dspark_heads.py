"""Low-cost sequential and confidence heads used by DSpark."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


@dataclass
class MarkovBlockOutput:
    """Outputs needed by target verification and confidence scheduling."""

    token_ids: torch.Tensor
    logits: torch.Tensor
    previous_embeddings: torch.Tensor


class VanillaMarkov(nn.Module):
    """Rank-factorized first-order transition bias.

    The heavy DFlash backbone produces every row of ``base_logits`` in one
    forward pass. This head then performs only a small embedding lookup and
    vocabulary projection per position, making each row depend on the token
    sampled immediately before it.
    """

    def __init__(self, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        if vocab_size <= 0 or markov_rank <= 0:
            raise ValueError("vocab_size and markov_rank must be positive")
        self.vocab_size = int(vocab_size)
        self.markov_rank = int(markov_rank)
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(
            self.markov_rank, self.vocab_size, bias=False
        )

    def get_previous_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Markov token ids must use an integer dtype")
        return self.markov_w1(token_ids.long())

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.get_previous_embeddings(token_ids))

    def apply_teacher_forced(
        self,
        base_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply transition bias when all previous tokens are already known."""
        self._validate_block_inputs(base_logits, previous_token_ids)
        return base_logits + self.compute_step_bias(previous_token_ids)

    def sample_block(
        self,
        base_logits: torch.Tensor,
        first_previous_token_ids: torch.Tensor,
        sampler: Optional[StepSampler] = None,
    ) -> MarkovBlockOutput:
        """Sample a block left-to-right while reusing parallel base logits."""
        if base_logits.ndim != 3:
            raise ValueError("base_logits must have shape [batch, block, vocab]")
        batch_size, block_size, vocab_size = base_logits.shape
        if vocab_size != self.vocab_size:
            raise ValueError(
                f"Expected vocabulary size {self.vocab_size}, got {vocab_size}"
            )
        if first_previous_token_ids.shape != (batch_size,):
            raise ValueError(
                "first_previous_token_ids must have shape [batch]"
            )
        if sampler is None:
            sampler = lambda logits, _step: torch.argmax(logits, dim=-1)

        sampled_tokens = []
        corrected_logits = []
        previous_embeddings = []
        previous_token_ids = first_previous_token_ids.long()
        for step_idx in range(block_size):
            step_previous = self.get_previous_embeddings(previous_token_ids)
            step_logits = base_logits[:, step_idx] + self.markov_w2(step_previous)
            next_token_ids = sampler(step_logits, step_idx)
            if next_token_ids.shape != (batch_size,):
                raise ValueError("DSpark step sampler must return shape [batch]")
            previous_embeddings.append(step_previous)
            corrected_logits.append(step_logits)
            sampled_tokens.append(next_token_ids.long())
            previous_token_ids = next_token_ids.long()

        if block_size == 0:
            return MarkovBlockOutput(
                token_ids=torch.empty(
                    (batch_size, 0), dtype=torch.long, device=base_logits.device
                ),
                logits=base_logits,
                previous_embeddings=base_logits.new_empty(
                    (batch_size, 0, self.markov_rank)
                ),
            )
        return MarkovBlockOutput(
            token_ids=torch.stack(sampled_tokens, dim=1),
            logits=torch.stack(corrected_logits, dim=1),
            previous_embeddings=torch.stack(previous_embeddings, dim=1),
        )

    def _validate_block_inputs(
        self,
        base_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> None:
        if base_logits.ndim != 3:
            raise ValueError("base_logits must have shape [batch, block, vocab]")
        if base_logits.shape[-1] != self.vocab_size:
            raise ValueError("base_logits vocabulary does not match Markov head")
        if previous_token_ids.shape != base_logits.shape[:2]:
            raise ValueError(
                "previous_token_ids must match base_logits [batch, block]"
            )


class DSparkConfidenceHead(nn.Module):
    """Predict and calibrate conditional target-acceptance probabilities."""

    def __init__(
        self,
        hidden_size: int,
        markov_rank: int,
        *,
        with_markov: bool = True,
        max_block_size: int = 32,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or max_block_size <= 0:
            raise ValueError("hidden_size and max_block_size must be positive")
        if with_markov and markov_rank <= 0:
            raise ValueError("with_markov requires a positive markov_rank")
        self.hidden_size = int(hidden_size)
        self.markov_rank = int(markov_rank)
        self.with_markov = bool(with_markov)
        input_size = self.hidden_size + (
            self.markov_rank if self.with_markov else 0
        )
        self.proj = nn.Linear(input_size, 1)
        self.register_buffer(
            "sts_temperatures",
            torch.ones(max_block_size, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have shape [batch, block, hidden_size]"
            )
        features = hidden_states
        if self.with_markov:
            expected = (*hidden_states.shape[:2], self.markov_rank)
            if previous_embeddings is None or previous_embeddings.shape != expected:
                raise ValueError(
                    "previous_embeddings must match [batch, block, markov_rank]"
                )
            features = torch.cat(
                (hidden_states, previous_embeddings.to(hidden_states.dtype)),
                dim=-1,
            )
        return self.proj(features.to(self.proj.weight.dtype)).squeeze(-1)

    def probabilities(
        self,
        hidden_states: torch.Tensor,
        previous_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raw = self(hidden_states, previous_embeddings)
        block_size = raw.shape[1]
        if block_size > self.sts_temperatures.numel():
            raise ValueError(
                "Confidence block is longer than the STS temperature table"
            )
        temperatures = self.sts_temperatures[:block_size].to(raw.device)
        return torch.sigmoid(raw.float() / temperatures.unsqueeze(0))

    def set_sts_temperatures(self, temperatures: torch.Tensor) -> None:
        temperatures = torch.as_tensor(
            temperatures, dtype=torch.float32, device=self.sts_temperatures.device
        )
        if temperatures.ndim != 1 or temperatures.numel() == 0:
            raise ValueError("STS temperatures must be a non-empty vector")
        if temperatures.numel() > self.sts_temperatures.numel():
            raise ValueError("STS temperature vector exceeds configured capacity")
        if not torch.isfinite(temperatures).all() or (temperatures <= 0).any():
            raise ValueError("STS temperatures must be finite and positive")
        self.sts_temperatures[:temperatures.numel()].copy_(temperatures)
