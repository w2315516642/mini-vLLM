"""Normalized runtime configuration for a DSpark draft checkpoint."""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


def _read(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


@dataclass(frozen=True)
class DSparkConfig:
    """Small, validated view of fields used by the inference runtime.

    The published checkpoints duplicate several values at the root and in
    ``dspark_config``/``dflash_config``. Normalizing them once keeps model and
    scheduler code independent from that serialization detail.
    """

    block_size: int
    mask_token_id: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    target_layer_ids: Tuple[int, ...]
    markov_rank: int
    enable_confidence_head: bool
    confidence_head_with_markov: bool
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int

    @classmethod
    def from_hf_config(cls, config: Any) -> "DSparkConfig":
        nested = (
            _read(config, "dspark_config")
            or _read(config, "dflash_config")
            or {}
        )

        def value(name: str, default: Any = None) -> Any:
            nested_value = _read(nested, name)
            return _read(config, name, default) if nested_value is None else nested_value

        normalized = cls(
            block_size=int(value("block_size", 0)),
            mask_token_id=int(value("mask_token_id", -1)),
            vocab_size=int(value("draft_vocab_size", value("vocab_size", 0))),
            hidden_size=int(value("hidden_size", 0)),
            intermediate_size=int(value("intermediate_size", 0)),
            num_hidden_layers=int(value("num_hidden_layers", 0)),
            num_attention_heads=int(value("num_attention_heads", 0)),
            num_key_value_heads=int(value("num_key_value_heads", 0)),
            head_dim=int(value("head_dim", 0)),
            target_layer_ids=tuple(int(i) for i in value("target_layer_ids", ())),
            markov_rank=int(value("markov_rank", 0)),
            enable_confidence_head=bool(value("enable_confidence_head", False)),
            confidence_head_with_markov=bool(
                value("confidence_head_with_markov", True)
            ),
            rms_norm_eps=float(value("rms_norm_eps", 1e-6)),
            rope_theta=float(value("rope_theta", 10000.0)),
            max_position_embeddings=int(value("max_position_embeddings", 0)),
        )
        normalized.validate()
        return normalized

    @property
    def verification_width(self) -> int:
        """Target inputs per round: one anchor plus all draft proposals."""
        return self.block_size + 1

    def validate(self) -> None:
        positive = {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "markov_rank": self.markov_rank,
            "max_position_embeddings": self.max_position_embeddings,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                "DSpark configuration fields must be positive: "
                + ", ".join(invalid)
            )
        if self.mask_token_id < 0 or self.mask_token_id >= self.vocab_size:
            raise ValueError(
                "mask_token_id must be within the draft vocabulary: "
                f"{self.mask_token_id} not in [0, {self.vocab_size})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "DSpark query heads must be divisible by key/value heads"
            )
        if not self.target_layer_ids:
            raise ValueError("DSpark requires at least one target auxiliary layer")
        if tuple(sorted(set(self.target_layer_ids))) != self.target_layer_ids:
            raise ValueError(
                "target_layer_ids must be strictly increasing and unique"
            )
        if self.confidence_head_with_markov and not self.enable_confidence_head:
            raise ValueError(
                "confidence_head_with_markov requires enable_confidence_head"
            )
