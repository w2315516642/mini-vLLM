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
        root_config = hf_config
        architectures = tuple(
            _get_default_config_value(hf_config, "architectures", ())
        )

        text_config = _get_default_config_value(
            hf_config, "text_config", hf_config
        )

        hidden_size = _get_config_value(text_config, "hidden_size")
        num_hidden_layers = _get_config_value(text_config, "num_hidden_layers")
        num_attention_heads = _get_config_value(text_config, "num_attention_heads")

        head_size = getattr(text_config, "head_dim", None)
        if head_size is None:
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by "
                    f"num_attention_heads ({num_attention_heads}) when "
                    "head_dim is not provided"
                )
            head_size = hidden_size // num_attention_heads

        num_key_value_heads = _get_default_config_value(
            text_config,
            "num_key_value_heads",
            num_attention_heads,
        )

        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            layer_types = [FULL_ATTENTION] * num_hidden_layers
        else:
            for layer in layer_types:
                if layer not in SUPPORTED_LAYER_TYPES:
                    raise ValueError(
                        f"layer {layer} is not supported yet"
                    )
        layer_types = tuple(layer_types)

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"query heads ({num_attention_heads}) must be grouped "
                f"evenly by KV heads ({num_key_value_heads})"
            )
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types contains {len(layer_types)} entries, but "
                f"num_hidden_layers is {num_hidden_layers}"
            )

        return cls(
            root_config=root_config,
            text_config=text_config,
            architectures=architectures,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_size=head_size,
            layer_types=layer_types,
        )

    @property
    def num_full_attention_layers(self) -> int:
        """Return the number of layers that allocate paged KV cache."""
        return sum(layer == FULL_ATTENTION for layer in self.layer_types)

    @property
    def num_linear_attention_layers(self) -> int:
        """Return the number of layers that allocate recurrent state."""
        return sum(layer == LINEAR_ATTENTION for layer in self.layer_types)

    @property
    def is_hybrid(self) -> bool:
        """Whether the model mixes full and linear attention layers."""
        return self.num_full_attention_layers != 0 and self.num_linear_attention_layers != 0

    def get_num_attention_heads(self, tensor_parallel_size: int) -> int:
        """Return query heads owned by one tensor-parallel rank."""
        if tensor_parallel_size <= 0:
            raise ValueError(
                "tensor_parallel_size must be positive, but got "
                f"{tensor_parallel_size}"
            )

        if self.num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({self.num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )
        return self.num_attention_heads // tensor_parallel_size

    def get_num_kv_heads(self, tensor_parallel_size: int) -> int:
        """Return KV heads owned by one tensor-parallel rank."""
        if tensor_parallel_size <= 0:
            raise ValueError(
                "tensor_parallel_size must be positive, but got "
                f"{tensor_parallel_size}"
            )

        if self.num_key_value_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of KV heads ({self.num_key_value_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )
        return self.num_key_value_heads // tensor_parallel_size

    def get_num_layers(self, pipeline_parallel_size: int) -> int:
        """Return decoder layers owned by one pipeline-parallel rank."""
        if pipeline_parallel_size <= 0:
            raise ValueError(
                "pipeline_parallel_size must be positive, but got "
                f"{pipeline_parallel_size}"
            )

        if self.num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({self.num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size})."
            )
        return self.num_hidden_layers // pipeline_parallel_size

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
        self.get_num_attention_heads(tensor_parallel_size)
        self.get_num_kv_heads(tensor_parallel_size)
        self.get_num_layers(pipeline_parallel_size)


def _get_default_config_value(config: Any, field_name: str, default: Any) -> Any:
    candidate = getattr(config, field_name, None)
    return candidate if candidate is not None else default


def _get_config_value(config: Any, field_name: str) -> Any:
    """Read one required config value.

    This helper is intentionally left as part of the assignment because clear
    missing-field errors are important when a newly released model changes its
    config schema.
    """
    try:
        candidate = getattr(config, field_name, None)
    except AttributeError as e:
        raise ValueError(
            f"required config field {field_name} is missing from {type(config).__name__}"
        ) from e
    if candidate is None:
        raise ValueError(
            f"required config field {field_name} is None in {type(config).__name__}"
        )
    return candidate
