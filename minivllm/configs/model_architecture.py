"""Normalized language-model structure used by the runtime.

Qwen3.8 is published with a multimodal root config. The dimensions consumed by
the language-model runtime live in ``text_config``, while Llama stores the same
fields directly on the root config. This module provides one normalized view so
the cache, attention, and parallel layers do not need model-specific branches.

Stage 1 learning task
---------------------
Implement ``ModelArchitecture.from_hf_config`` and
``ModelArchitecture.verify_parallelism``. Keep the object immutable after it
has been built: later stages will pass it to workers and cache engines as a
description of model structure, not as mutable runtime state.
"""

from dataclasses import dataclass
from typing import Any, Tuple

from transformers import PretrainedConfig


FULL_ATTENTION = "full_attention"
LINEAR_ATTENTION = "linear_attention"
SUPPORTED_LAYER_TYPES = frozenset({FULL_ATTENTION, LINEAR_ATTENTION})


@dataclass(frozen=True)
class ModelArchitecture:
    """Model dimensions after flat and nested HF configs are normalized.

    Attributes:
        root_config: Original config returned by ``AutoConfig``.
        text_config: Config that owns the language-model dimensions. This is
            the root config for Llama and ``root_config.text_config`` for
            Qwen3.8.
        architectures: Architecture names used by the model registry.
        hidden_size: Residual-stream width.
        num_hidden_layers: Number of language-model decoder layers.
        num_attention_heads: Global number of query heads.
        num_key_value_heads: Global number of key/value heads.
        head_size: Width of one attention head. Qwen3.8 provides this through
            ``head_dim``; it cannot be inferred from ``hidden_size``.
        layer_types: Per-layer token mixer type.
    """

    root_config: PretrainedConfig
    text_config: PretrainedConfig
    architectures: Tuple[str, ...]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_size: int
    layer_types: Tuple[str, ...]

    @classmethod
    def from_hf_config(
        cls,
        hf_config: PretrainedConfig,
    ) -> "ModelArchitecture":
        """Build a validated normalized view of a Hugging Face config.

        Required behavior is specified by
        ``tests/test_model_architecture.py``. In particular, support both a
        flat Llama config and a multimodal Qwen config with nested
        ``text_config``. Raise ``ValueError`` with a useful field name when a
        structural invariant is violated.
        """
        # TODO(student): Stage 1 assignment. Do not special-case model names.
        raise NotImplementedError

    @property
    def num_full_attention_layers(self) -> int:
        """Return the number of layers that allocate paged KV cache."""
        # TODO(student): Stage 1 assignment.
        raise NotImplementedError

    @property
    def num_linear_attention_layers(self) -> int:
        """Return the number of layers that allocate recurrent state."""
        # TODO(student): Stage 1 assignment.
        raise NotImplementedError

    @property
    def is_hybrid(self) -> bool:
        """Whether the model mixes full and linear attention layers."""
        # TODO(student): Stage 1 assignment.
        raise NotImplementedError

    def get_num_attention_heads(self, tensor_parallel_size: int) -> int:
        """Return query heads owned by one tensor-parallel rank."""
        # TODO(student): Stage 1 assignment.
        raise NotImplementedError

    def get_num_kv_heads(self, tensor_parallel_size: int) -> int:
        """Return KV heads owned by one tensor-parallel rank."""
        # TODO(student): Stage 1 assignment.
        raise NotImplementedError

    def get_num_layers(self, pipeline_parallel_size: int) -> int:
        """Return decoder layers owned by one pipeline-parallel rank."""
        # TODO(student): Stage 1 assignment.
        raise NotImplementedError

    def verify_parallelism(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
    ) -> None:
        """Verify that model dimensions can be partitioned as requested.

        Validate positive parallel sizes and exact divisibility for query
        heads, KV heads, and decoder layers. Error messages should include the
        invalid dimension and requested parallel size.
        """
        # TODO(student): Stage 1 assignment.
        raise NotImplementedError


def _get_config_value(config: Any, field_name: str) -> Any:
    """Read one required config value.

    This helper is intentionally left as part of the assignment because clear
    missing-field errors are important when a newly released model changes its
    config schema.
    """
    # TODO(student): Stage 1 assignment.
    raise NotImplementedError
