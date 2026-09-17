"""Stage 9: resident FP8 weights with fused W8A16 inference.

The explicit dequantizer is an oracle only. Production forward uses the Triton
kernel, never a whole-weight dequantization followed by a separate GEMM.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class FP8BlockConfig:
    block_size: Tuple[int, int]
    modules_to_not_convert: Tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, config: Mapping) -> "FP8BlockConfig":
        """Parse fp8/e4m3, dynamic activation metadata, and positive block sizes.

        Require quant_method='fp8', activation_scheme='dynamic', and a two-item
        weight_block_size (positive Python ints, not bools). fmt defaults to
        'e4m3'. modules_to_not_convert defaults to an empty list; accept only
        nonempty, well-formed dot-separated names. Explicit per-tensor flags
        must be false. Unknown quantization methods/formats are errors, not
        a signal to silently load floating weights. Do not mutate config.
        """
        if not isinstance(config, Mapping):
            raise ValueError("quantization_config must be a mapping")
        for name, expected in (("quant_method", "fp8"), ("activation_scheme", "dynamic")):
            if config.get(name) != expected:
                raise ValueError(f"{name} must be {expected!r}")
        if config.get("fmt", "e4m3") != "e4m3":
            raise ValueError("fmt must be 'e4m3'")
        blocks = config.get("weight_block_size")
        if (not isinstance(blocks, (list, tuple)) or len(blocks) != 2
                or any(type(b) is not int or b <= 0 for b in blocks)):
            raise ValueError("weight_block_size must contain two positive integers")
        for name in ("weight_per_tensor", "act_per_tensor"):
            if name in config and config[name] is not False:
                raise ValueError(f"{name} must be false for block FP8")
        exclusions = config.get("modules_to_not_convert", [])
        if not isinstance(exclusions, (list, tuple)) or any(
            not isinstance(name, str) or any(not part or part != part.strip()
                                           for part in name.split("."))
            for name in exclusions
        ):
            raise ValueError("modules_to_not_convert must contain dot-separated module names")
        return cls(tuple(blocks), tuple(exclusions))

    def should_quantize(self, source_modules: Sequence[str]) -> bool:
        """Decide for one ordinary projection or a complete packed group.

        Normalize model.language_model.* to model.* in both inputs/exclusions.
        Match exact names or dot-delimited descendants (not arbitrary prefixes).
        Empty groups are invalid. All excluded -> False; none -> True;
        partially excluded packed groups -> ValueError naming the group.
        """
        if not source_modules or isinstance(source_modules, str):
            raise ValueError("source_modules must be a nonempty sequence of module names")

        def normalize(name):
            if name.startswith("model.language_model."):
                return "model." + name.removeprefix("model.language_model.")
            return name

        exclusions = tuple(normalize(name) for name in self.modules_to_not_convert)
        excluded = [any(normalize(name) == prefix or normalize(name).startswith(prefix + ".")
                        for prefix in exclusions) for name in source_modules]
        if any(excluded) and not all(excluded):
            raise ValueError(f"Partially excluded packed FP8 group: {tuple(source_modules)}")
        return not all(excluded)


def resolve_fp8_config(root_config, text_config):
    """Preserve nested model configs; reject ambiguous metadata."""
    root = getattr(root_config, "quantization_config", None)
    text = getattr(text_config, "quantization_config", None)
    if root is not None and text is not None and root != text:
        raise ValueError("Root and text quantization_config disagree")
    raw = root if root is not None else text
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or raw.get("quant_method") != "fp8":
        raise ValueError("Only block FP8 quantization_config is supported")
    return FP8BlockConfig.from_dict(raw)


def _validate_fp8_weight_metadata(weight, scale, block_size):
    """Shared shape/dtype checks; never read CUDA tensor values here."""
    if (not isinstance(block_size, (tuple, list)) or len(block_size) != 2
            or any(type(b) is not int or b <= 0 for b in block_size)):
        raise ValueError("block_size must contain two positive integers")
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("weight must be a 2D float8_e4m3fn tensor")
    n, k = weight.shape
    if n <= 0 or k <= 0:
        raise ValueError("weight dimensions must be positive")
    bn, bk = block_size
    if scale.dtype != torch.float32 or tuple(scale.shape) != ((n+bn-1)//bn, (k+bk-1)//bk):
        raise ValueError("weight_scale_inv must be FP32 with the ceil-based block shape")
    if weight.device != scale.device:
        raise ValueError("weight and weight_scale_inv must be on the same device")


def dequantize_fp8_block_weight(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: Tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Test oracle ONLY: materialize W[i,j] * S[i//block_n,j//block_k].

    Used by tests, not Fp8Linear.forward. Its rounding order defines the fused
    kernel's semantics: multiply in FP32, then round weights to activation dtype.
    """
    _validate_fp8_weight_metadata(weight, weight_scale_inv, block_size)
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Reference output dtype must be FP32, FP16 or BF16")
    bn, bk = block_size
    scales = weight_scale_inv.repeat_interleave(bn, 0).repeat_interleave(bk, 1)
    scales = scales[:weight.shape[0], :weight.shape[1]]
    return (weight.float() * scales).to(dtype).contiguous()


@torch.no_grad()
def fused_fp8_linear(inputs, weight, weight_scale_inv, block_size):
    """CUDA FP16/BF16 activation + E4M3FN weight, no dense weight temporary.

    Arbitrary leading activation dimensions are flattened to M. reshape may
    copy a non-viewable activation, but weight/scale retain their own strides.
    CPU/FP32 execution is deliberately rejected, not silently sent to the oracle.
    """
    _validate_fp8_weight_metadata(weight, weight_scale_inv, block_size)
    if not inputs.is_cuda or inputs.device != weight.device:
        raise ValueError("Fused FP8 linear requires inputs/weight/scale on the same CUDA device")
    if inputs.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Fused FP8 linear requires FP16 or BF16 activations")
    if inputs.ndim < 2 or inputs.shape[-1] != weight.shape[1]:
        raise ValueError("inputs must have rank >= 2 and last dimension equal to weight K")
    x = inputs.reshape(-1, weight.shape[1])
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=inputs.dtype, device=inputs.device)
    if x.shape[0]:
        # Lazy import keeps CPU configuration/loading tests independent of Triton.
        from minivllm.model_executor.layers.fp8_triton import launch_fp8_linear
        with torch.cuda.device(inputs.device):
            launch_fp8_linear(x, weight, weight_scale_inv, output, block_size)
    return output.reshape(*inputs.shape[:-1], weight.shape[0])


class Fp8Linear(nn.Module):
    """TP=1 projection with compressed storage and the original return contract."""

    def __init__(self, input_size, output_size, block_size, *, device="cpu",
                 returns_tuple=False):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.block_size = block_size
        self.returns_tuple = returns_tuple
        block_n, block_k = block_size
        self.weight = nn.Parameter(torch.empty(
            output_size, input_size, device=device, dtype=torch.float8_e4m3fn,
        ), requires_grad=False)
        # No default scale=1: the loader must prove every scale shard was read.
        self.weight_scale_inv = nn.Parameter(torch.empty(
            (output_size + block_n - 1) // block_n,
            (input_size + block_k - 1) // block_k,
            device=device, dtype=torch.float32,
        ), requires_grad=False)

    def forward(self, inputs: torch.Tensor):
        """Preserve the projection interface; the kernel owns fused computation."""
        output = fused_fp8_linear(inputs, self.weight, self.weight_scale_inv, self.block_size)
        return (output, None) if self.returns_tuple else output


@torch.no_grad()
def load_fp8_shard(
    layer: Fp8Linear, tensor: torch.Tensor, *, kind: str,
    row_offset: int, num_rows: int, source_name: str,
) -> None:
    """Copy a weight or scale shard after validating it completely.

    kind is 'weight' or 'scale'. Offsets/counts ALWAYS describe weight rows.
    Require positive num_rows, nonnegative offset, destination bounds and
    block-aligned start/internal end. A tail is allowed at the matrix end.
    Weights must be E4M3FN with exact [num_rows,K] shape and finite values.
    Scales must be FP32 with exact [ceil(num_rows/block_n),ceil(K/block_k)]
    shape, all finite and strictly positive. No silent dtype conversion.
    Errors include source_name; no mutation on invalid input. Ordinary
    copy_ may transfer CPU tensors to the destination. This is startup-only.
    Duplicate/missing source coverage remains the model loader's job.
    """
    bn, bk = layer.block_size
    if (type(row_offset) is not int or type(num_rows) is not int
            or row_offset < 0 or num_rows <= 0
            or row_offset + num_rows > layer.output_size):
        raise ValueError(f"{source_name}: invalid weight row range")
    end = row_offset + num_rows
    if row_offset % bn or (end != layer.output_size and end % bn):
        raise ValueError(f"{source_name}: packed weight row boundary must be block-aligned")
    if kind == "weight":
        expected_shape, dtype = (num_rows, layer.input_size), torch.float8_e4m3fn
        destination = layer.weight[row_offset:end]
    elif kind == "scale":
        expected_shape = ((num_rows + bn - 1)//bn, (layer.input_size + bk - 1)//bk)
        dtype = torch.float32
        # Source offsets use weight rows; scale storage uses quantization blocks.
        destination = layer.weight_scale_inv[row_offset//bn:(end + bn - 1)//bn]
    else:
        raise ValueError(f"{source_name}: kind must be 'weight' or 'scale'")
    if tensor.dtype != dtype or tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{source_name}: expected {dtype} {expected_shape}, "
                         f"got {tensor.dtype} {tuple(tensor.shape)}")
    # Check values once at startup, never in the per-token path. FP8 isfinite
    # is not implemented on every backend, so validate its FP32 representation.
    values = tensor.float()
    valid = torch.isfinite(values)
    if kind == "scale":
        valid = valid & (values > 0)
    if not bool(valid.all()):
        raise ValueError(f"{source_name}: values must be finite" +
                         (" and scales strictly positive" if kind == "scale" else ""))
    destination.copy_(tensor)
