"""Speculative decoding algorithms shared by model and scheduler code."""

from minivllm.spec_decode.dspark_config import DSparkConfig
from minivllm.spec_decode.dspark_heads import (
    DSparkConfidenceHead,
    MarkovBlockOutput,
    VanillaMarkov,
)

__all__ = [
    "DSparkConfig",
    "DSparkConfidenceHead",
    "MarkovBlockOutput",
    "VanillaMarkov",
]
