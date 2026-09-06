"""Inference-only W8A16 Linear: FP8 weights stay compressed in device memory.

Unlike FP8 Tensor Core GEMM, this also works on Ampere. Each tile decodes
E4M3FN bytes, applies the checkpoint's block scale, and feeds BF16/FP16
operands to Tensor Cores. No full dequantized weight tensor is allocated.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3fn_to_float(bits):
    # Software decoding avoids emitting FP8 conversion instructions unavailable
    # on SM80. E4M3FN has bias 7, subnormals, signed zero and two NaN encodings.
    bits = bits.to(tl.uint32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    magnitude = tl.where(exponent == 0, mantissa.to(tl.float32) * 0.001953125,
                         normal.to(tl.float32, bitcast=True))
    signed = magnitude.to(tl.uint32, bitcast=True) | ((bits & 128) << 24)
    return tl.where((bits & 127) == 127, float("nan"),
                    signed.to(tl.float32, bitcast=True))


@triton.jit
def _linear_kernel(X, W, S, Bias, Y, Partial,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   X0: tl.constexpr, X1: tl.constexpr,
                   W0: tl.constexpr, W1: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr,
                   BN: tl.constexpr, BK: tl.constexpr,
                   HAS_BIAS: tl.constexpr, SPLIT: tl.constexpr,
                   TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    # Visit nearby M tiles before moving along N so they reuse compressed
    # weight tiles in L2, especially for MLP weights larger than the cache.
    tiles_n = tl.cdiv(N, TN)
    group = pid // (8 * tiles_n)
    first_m = group * 8
    group_m = tl.minimum(tl.cdiv(M, TM) - first_m, 8)
    pid_m = first_m + (pid % (8 * tiles_n)) % group_m
    pid_n = (pid % (8 * tiles_n)) // group_m
    m = pid_m * TM + tl.arange(0, TM)
    n = pid_n * TN + tl.arange(0, TN)
    acc = tl.zeros((TM, TN), tl.float32)
    for block in range(tl.cdiv(K, TK * SPLIT)):
        k_base = (block * SPLIT + split) * TK
        k = k_base + tl.arange(0, TK)
        x = tl.load(X + m[:, None] * X0 + k[None, :] * X1,
                    (m[:, None] < M) & (k[None, :] < K), other=0)
        raw = tl.load(W + k[:, None] * W1 + n[None, :] * W0,
                      (k[:, None] < K) & (n[None, :] < N), other=0)
        if BN % TN == 0 and BK % TK == 0:
            # Qwen's 128x128 blocks share one scale across this entire tile.
            # Keep it scalar instead of staging a broadcast KxN scale matrix.
            scale = tl.load(S + (pid_n * TN // BN) * S0 + (k_base // BK) * S1,
                            k_base < K, other=0)
        else:
            scale = tl.load(S + (n[None, :] // BN) * S0 + (k[:, None] // BK) * S1,
                            (k[:, None] < K) & (n[None, :] < N), other=0)
        # Match the existing reference's scale cast and weight rounding.
        scale = scale.to(x.dtype).to(tl.float32)
        w = (_e4m3fn_to_float(raw) * scale).to(x.dtype)
        acc = tl.dot(x, w, acc)
    offsets = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        if HAS_BIAS:
            acc += tl.load(Bias + n, n < N, other=0).to(tl.float32)[None, :]
        tl.store(Y + offsets, acc, mask)
    else:
        # Small-M GEMMs need extra CTAs to occupy the GPU. Split only K and
        # reduce in FP32; scratch scales with the output, never with the weight.
        tl.store(Partial + split * M * N + offsets, acc, mask)


@triton.jit
def _finish_split(Partial, Bias, Y, MN: tl.constexpr, N: tl.constexpr,
                  SPLIT: tl.constexpr, HAS_BIAS: tl.constexpr,
                  BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLIT):
        total += tl.load(Partial + split * MN + i, i < MN, other=0)
    if HAS_BIAS:
        total += tl.load(Bias + i % N, i < MN, other=0).to(tl.float32)
    tl.store(Y + i, total, i < MN)


def fp8_linear(input_, weight, scale, block_size, bias=None, out=None, *,
               _launch_config=None):
    """Return input @ dequant(weight).T, optionally into a TP output buffer."""
    if input_.device.type != "cuda" or input_.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Fused FP8 Linear requires CUDA FP16/BF16 inputs")
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise ValueError("weight must be an E4M3FN matrix")
    n, k = weight.shape
    bn, bk = block_size
    if input_.ndim < 1 or input_.shape[-1] != k or min(bn, bk, k, n) <= 0:
        raise ValueError("Invalid Linear input shape or quantization block size")
    if scale.shape != (triton.cdiv(n, bn), triton.cdiv(k, bk)):
        raise ValueError("scale shape must match the FP8 weight blocks")
    if scale.dtype != torch.float32 or weight.device != input_.device or scale.device != input_.device:
        raise ValueError("FP8 weights and FP32 scales must share the input device")
    shape = (*input_.shape[:-1], n)
    if bias is not None and (bias.shape != (n,) or bias.dtype != input_.dtype or bias.device != input_.device):
        raise ValueError("bias must match the output width, dtype and device")
    if out is not None and (out.shape != shape or out.dtype != input_.dtype
                            or out.device != input_.device or not out.is_contiguous()):
        raise ValueError("out must be contiguous and match the Linear output")
    x = input_.reshape(-1, k)
    m = x.shape[0]
    y = torch.empty(shape, dtype=input_.dtype, device=input_.device) if out is None else out
    if m == 0:
        return y
    split = 8 if m <= 16 and k >= 1024 else 1
    tm = min(128, max(16, triton.next_power_of_2(m)))
    tn, tk = 64, 32
    warps, stages = 4, 1
    # Benchmark-only override. Runtime keeps the existing deterministic choice
    # until candidate configurations have been measured on the deployment GPU.
    if _launch_config is not None:
        tm, tn, tk, split, warps, stages = _launch_config
        if (tm not in (16, 32, 64, 128) or tn not in (32, 64, 128)
                or tk not in (32, 64, 128) or split not in (1, 2, 4, 8)
                or warps not in (4, 8) or stages not in (1, 2, 3)):
            raise ValueError("Unsupported FP8 Linear launch configuration")
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32) if split > 1 else y
    with torch.cuda.device(input_.device):
        _linear_kernel[(triton.cdiv(m, tm) * triton.cdiv(n, tn), split)](
            x, weight.view(torch.uint8), scale, bias, y, partial, m, n, k,
            *x.stride(), *weight.stride(), *scale.stride(), bn, bk,
            bias is not None, split, tm, tn, tk, num_warps=warps, num_stages=stages)
        if split > 1:
            _finish_split[(triton.cdiv(m * n, 256),)](
                partial, bias, y, m * n, n, split, bias is not None, 256)
    return y
