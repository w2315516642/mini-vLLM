"""Confidence-scheduled target verification for DSpark draft blocks."""

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class VerificationCostProfile:
    """Measured target latency at monotonically increasing token counts."""

    token_counts: Tuple[int, ...]
    latency_ms: Tuple[float, ...]
    draft_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.token_counts or len(self.token_counts) != len(self.latency_ms):
            raise ValueError("Cost profile requires paired token and latency values")
        if any(left >= right for left, right in zip(
            self.token_counts, self.token_counts[1:]
        )):
            raise ValueError("Cost profile token counts must be increasing")
        if any(value <= 0.0 for value in self.latency_ms):
            raise ValueError("Cost profile latency must be positive")
        if self.draft_latency_ms < 0.0:
            raise ValueError("Draft latency must be non-negative")

    @classmethod
    def linear(
        cls,
        max_tokens: int,
        *,
        fixed_ms: float = 1.0,
        per_token_ms: float = 0.05,
        draft_latency_ms: float = 0.0,
    ) -> "VerificationCostProfile":
        if max_tokens <= 0 or fixed_ms <= 0.0 or per_token_ms < 0.0:
            raise ValueError("Invalid linear verification cost profile")
        if max_tokens == 1:
            return cls(
                token_counts=(1,),
                latency_ms=(fixed_ms + per_token_ms,),
                draft_latency_ms=draft_latency_ms,
            )
        return cls(
            token_counts=(1, max_tokens),
            latency_ms=(fixed_ms + per_token_ms, fixed_ms + per_token_ms * max_tokens),
            draft_latency_ms=draft_latency_ms,
        )

    def estimate_ms(self, num_tokens: int) -> float:
        if num_tokens <= 0:
            raise ValueError("Verification token count must be positive")
        if len(self.token_counts) == 1:
            return self.latency_ms[0] + self.draft_latency_ms
        if num_tokens <= self.token_counts[0]:
            return self.latency_ms[0] + self.draft_latency_ms
        elif num_tokens >= self.token_counts[-1]:
            left = len(self.token_counts) - 2
        else:
            left = next(
                index for index in range(len(self.token_counts) - 1)
                if self.token_counts[index] <= num_tokens <= self.token_counts[index + 1]
            )
        x0, x1 = self.token_counts[left:left + 2]
        y0, y1 = self.latency_ms[left:left + 2]
        ratio = (num_tokens - x0) / (x1 - x0)
        return y0 + ratio * (y1 - y0) + self.draft_latency_ms


@dataclass(frozen=True)
class AdaptiveVerificationPlan:
    draft_widths: Dict[int, int]
    expected_output_tokens: float
    estimated_tokens_per_second: float
    target_tokens: int


class AdaptiveVerificationPlanner:
    """Choose globally valuable draft prefixes under the target token budget."""

    def __init__(
        self,
        cost_profile: VerificationCostProfile,
        *,
        min_survival_probability: float = 0.0,
    ) -> None:
        if not 0.0 <= min_survival_probability <= 1.0:
            raise ValueError("min_survival_probability must be in [0, 1]")
        self.cost_profile = cost_profile
        self.min_survival_probability = min_survival_probability

    def plan(
        self,
        confidence_by_seq: Mapping[int, Sequence[float]],
        *,
        max_total_tokens: int,
        fixed_target_tokens: int = 0,
    ) -> AdaptiveVerificationPlan:
        if max_total_tokens <= 0 or fixed_target_tokens < 0:
            raise ValueError("Invalid adaptive verification token budget")
        seq_ids = list(confidence_by_seq)
        base_tokens = fixed_target_tokens + len(seq_ids)
        if base_tokens > max_total_tokens:
            raise ValueError("Token budget cannot fit one anchor per sequence")

        candidates = []
        for seq_id, confidence_values in confidence_by_seq.items():
            survival = 1.0
            for position, confidence in enumerate(confidence_values):
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("DSpark confidence values must be in [0, 1]")
                survival *= confidence
                if survival >= self.min_survival_probability:
                    candidates.append((survival, int(seq_id), position))

        # Survival probabilities are non-increasing within each sequence. The
        # global order therefore preserves prefix closure without extra repair.
        candidates.sort(key=lambda item: (-item[0], item[2], item[1]))
        capacity = max_total_tokens - base_tokens
        candidates = candidates[:capacity]
        baseline_output = float(len(seq_ids))
        baseline_sps = 1000.0 * baseline_output / self.cost_profile.estimate_ms(
            max(base_tokens, 1)
        )
        best_count = 0
        best_expected = baseline_output
        best_sps = baseline_sps
        expected = baseline_output
        for count, (survival, _, _) in enumerate(candidates, start=1):
            expected += survival
            target_tokens = base_tokens + count
            sps = 1000.0 * expected / self.cost_profile.estimate_ms(target_tokens)
            if sps > best_sps:
                best_count = count
                best_expected = expected
                best_sps = sps

        widths = {seq_id: 0 for seq_id in seq_ids}
        for _, seq_id, position in candidates[:best_count]:
            if position != widths[seq_id]:
                raise RuntimeError("Adaptive candidate order violated prefix closure")
            widths[seq_id] += 1
        return AdaptiveVerificationPlan(
            draft_widths=widths,
            expected_output_tokens=best_expected,
            estimated_tokens_per_second=best_sps,
            target_tokens=base_tokens + best_count,
        )


__all__ = [
    "AdaptiveVerificationPlan",
    "AdaptiveVerificationPlanner",
    "VerificationCostProfile",
]
