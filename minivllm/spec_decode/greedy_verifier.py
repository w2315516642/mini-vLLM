"""Greedy target verification shared by MTP and DSpark."""

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import torch


@dataclass(frozen=True)
class GreedyVerificationResult:
    """Accepted outputs and target rows that assigned their probabilities."""

    token_ids: Tuple[int, ...]
    logit_indices: Tuple[int, ...]
    committed_tokens: int
    last_committed_index: int
    accepted_draft_tokens: int
    all_accepted: bool


def verify_greedy_block(
    target_logits: torch.Tensor,
    draft_token_ids: Sequence[int],
    *,
    is_eos: Callable[[int], bool],
    ignore_eos: bool,
) -> GreedyVerificationResult:
    """Verify a linear draft block and emit one correction or bonus token.

    Row ``i`` predicts the token after verification input ``i``. Therefore row
    zero checks the first draft token and the final row supplies the bonus when
    every draft token is accepted.
    """
    drafts = tuple(int(token_id) for token_id in draft_token_ids)
    if not drafts:
        raise ValueError("Greedy verification requires at least one draft token")
    if target_logits.ndim != 2 or target_logits.shape[0] != len(drafts) + 1:
        raise ValueError(
            "Target logits must contain one row per draft plus a bonus row"
        )

    output_tokens = []
    logit_indices = []
    for index, draft_token_id in enumerate(drafts):
        target_token_id = int(torch.argmax(target_logits[index]).item())
        if target_token_id != draft_token_id:
            output_tokens.append(target_token_id)
            logit_indices.append(index)
            return GreedyVerificationResult(
                token_ids=tuple(output_tokens),
                logit_indices=tuple(logit_indices),
                committed_tokens=index + 1,
                last_committed_index=index,
                accepted_draft_tokens=index,
                all_accepted=False,
            )

        output_tokens.append(draft_token_id)
        logit_indices.append(index)
        if is_eos(draft_token_id) and not ignore_eos:
            return GreedyVerificationResult(
                token_ids=tuple(output_tokens),
                logit_indices=tuple(logit_indices),
                committed_tokens=index + 2,
                last_committed_index=index + 1,
                accepted_draft_tokens=index + 1,
                all_accepted=index + 1 == len(drafts),
            )

    bonus_token_id = int(torch.argmax(target_logits[-1]).item())
    output_tokens.append(bonus_token_id)
    logit_indices.append(len(drafts))
    return GreedyVerificationResult(
        token_ids=tuple(output_tokens),
        logit_indices=tuple(logit_indices),
        committed_tokens=len(drafts) + 1,
        last_committed_index=len(drafts),
        accepted_draft_tokens=len(drafts),
        all_accepted=True,
    )


__all__ = ["GreedyVerificationResult", "verify_greedy_block"]
