"""Exact speculative rejection sampling for one linear draft block."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class RejectionSamplingResult:
    token_ids: Tuple[int, ...]
    target_row_indices: Tuple[int, ...]
    committed_tokens: int
    last_committed_index: int
    accepted_draft_tokens: int
    all_accepted: bool


def logits_to_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    """Apply the engine's sampling transforms and return normalized rows."""
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("Sampling logits must have a non-empty vocabulary")
    if temperature <= 0.0:
        raise ValueError("Rejection sampling requires positive temperature")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    vocab_size = logits.shape[-1]
    if top_k == -1:
        top_k = vocab_size
    if not 1 <= top_k <= vocab_size:
        raise ValueError("top_k must be -1 or within the vocabulary")

    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    sorted_probs[(cumulative - sorted_probs) > top_p] = 0.0
    sorted_probs[..., top_k:] = 0.0
    filtered = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


def step_sampling_probs(
    logits: torch.Tensor,
    *,
    output_history: Sequence[int],
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
) -> torch.Tensor:
    """Build one transformed distribution for autoregressive draft sampling."""
    if logits.ndim != 1:
        raise ValueError("Step logits must be one-dimensional")
    adjusted = logits.float().clone()
    if output_history and (presence_penalty or frequency_penalty):
        history = torch.as_tensor(
            list(output_history), dtype=torch.long, device=adjusted.device
        )
        counts = torch.zeros_like(adjusted)
        counts.scatter_add_(0, history, torch.ones_like(history).float())
        adjusted -= presence_penalty * (counts > 0)
        adjusted -= frequency_penalty * counts
    return logits_to_probs(
        adjusted,
        temperature=max(float(temperature), 1e-5),
        top_p=top_p,
        top_k=top_k,
    )


def target_block_probs(
    logits: torch.Tensor,
    *,
    output_history: Sequence[int],
    draft_token_ids: Sequence[int],
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
) -> torch.Tensor:
    """Transform each target row with the history visible at that position."""
    drafts = [int(token_id) for token_id in draft_token_ids]
    if logits.ndim != 2 or logits.shape[0] != len(drafts) + 1:
        raise ValueError("Target block logits must include the bonus row")
    adjusted = logits.float().clone()
    if presence_penalty or frequency_penalty:
        vocab_size = logits.shape[-1]
        counts = torch.zeros(
            vocab_size, dtype=adjusted.dtype, device=adjusted.device
        )
        history = torch.as_tensor(
            list(output_history), dtype=torch.long, device=adjusted.device
        )
        if history.numel():
            counts.scatter_add_(0, history, torch.ones_like(history).float())
        for row in range(adjusted.shape[0]):
            adjusted[row] -= presence_penalty * (counts > 0)
            adjusted[row] -= frequency_penalty * counts
            if row < len(drafts):
                counts[drafts[row]] += 1.0
    return logits_to_probs(
        adjusted,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


def draft_block_probs(
    logits: torch.Tensor,
    *,
    output_history: Sequence[int],
    draft_token_ids: Sequence[int],
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float,
    frequency_penalty: float,
) -> torch.Tensor:
    """Transform saved draft rows with their autoregressive token history."""
    drafts = [int(token_id) for token_id in draft_token_ids]
    if logits.ndim != 2 or logits.shape[0] != len(drafts):
        raise ValueError("Draft logits must contain one row per sampled token")
    # Reuse the target implementation by appending a disposable bonus row.
    extended = torch.cat((logits, logits[-1:].clone()), dim=0)
    return target_block_probs(
        extended,
        output_history=output_history,
        draft_token_ids=drafts,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )[:-1]


def _sample_row(
    probabilities: torch.Tensor,
    generator: Optional[torch.Generator],
) -> int:
    return int(torch.multinomial(
        probabilities, num_samples=1, generator=generator
    ).item())


def rejection_sample_block(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: Sequence[int],
    *,
    is_eos: Callable[[int], bool],
    ignore_eos: bool,
    generator: Optional[torch.Generator] = None,
    uniforms: Optional[Sequence[float]] = None,
) -> RejectionSamplingResult:
    """Sample exactly from the target while reusing accepted draft tokens."""
    drafts = tuple(int(token_id) for token_id in draft_token_ids)
    if not drafts:
        raise ValueError("Rejection sampling requires at least one draft token")
    expected_target_shape = (len(drafts) + 1, target_probs.shape[-1])
    expected_draft_shape = (len(drafts), target_probs.shape[-1])
    if target_probs.shape != expected_target_shape:
        raise ValueError("Target probabilities must include one bonus row")
    if draft_probs.shape != expected_draft_shape:
        raise ValueError("Draft probabilities must contain one row per draft")
    if torch.any(target_probs < 0) or torch.any(draft_probs < 0):
        raise ValueError("Sampling probabilities must be non-negative")
    if not torch.allclose(
        target_probs.sum(-1), torch.ones_like(target_probs.sum(-1)), atol=1e-5
    ) or not torch.allclose(
        draft_probs.sum(-1), torch.ones_like(draft_probs.sum(-1)), atol=1e-5
    ):
        raise ValueError("Sampling probability rows must sum to one")
    if uniforms is not None and len(uniforms) != len(drafts):
        raise ValueError("One acceptance uniform is required per draft token")

    output_tokens = []
    target_rows = []
    for index, draft_token_id in enumerate(drafts):
        p = float(target_probs[index, draft_token_id])
        q = float(draft_probs[index, draft_token_id])
        acceptance = min(1.0, p / q) if q > 0.0 else 0.0
        uniform = (
            float(uniforms[index])
            if uniforms is not None
            else float(torch.rand((), device=target_probs.device, generator=generator))
        )
        if uniform > acceptance:
            residual = (target_probs[index] - draft_probs[index]).clamp_min(0)
            residual_sum = residual.sum()
            if float(residual_sum) <= 0.0:
                # This can only occur from finite-precision cancellation.
                residual = target_probs[index]
            else:
                residual = residual / residual_sum
            correction = _sample_row(residual, generator)
            output_tokens.append(correction)
            target_rows.append(index)
            return RejectionSamplingResult(
                token_ids=tuple(output_tokens),
                target_row_indices=tuple(target_rows),
                committed_tokens=index + 1,
                last_committed_index=index,
                accepted_draft_tokens=index,
                all_accepted=False,
            )

        output_tokens.append(draft_token_id)
        target_rows.append(index)
        if is_eos(draft_token_id) and not ignore_eos:
            return RejectionSamplingResult(
                token_ids=tuple(output_tokens),
                target_row_indices=tuple(target_rows),
                committed_tokens=index + 2,
                last_committed_index=index + 1,
                accepted_draft_tokens=index + 1,
                all_accepted=index + 1 == len(drafts),
            )

    bonus = _sample_row(target_probs[-1], generator)
    output_tokens.append(bonus)
    target_rows.append(len(drafts))
    return RejectionSamplingResult(
        token_ids=tuple(output_tokens),
        target_row_indices=tuple(target_rows),
        committed_tokens=len(drafts) + 1,
        last_committed_index=len(drafts),
        accepted_draft_tokens=len(drafts),
        all_accepted=True,
    )


__all__ = [
    "RejectionSamplingResult",
    "draft_block_probs",
    "logits_to_probs",
    "rejection_sample_block",
    "step_sampling_probs",
    "target_block_probs",
]
