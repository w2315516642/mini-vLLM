"""Fused dequantization/GEMM, imported only for CUDA execution.

W8A16: FP8 storage, on-chip FP32 rescaling then FP16/BF16 dot operands.
No separate dequantization launch, FP8 activation quantizer, or dense W buffer.
"""
import triton
import triton.language as tl


@triton.jit
def fp8_linear_kernel(
    X, W, S, Y,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm: tl.constexpr, stride_xk: tl.constexpr,
    stride_wn: tl.constexpr, stride_wk: tl.constexpr,
    stride_sn: tl.constexpr, stride_sk: tl.constexpr,
    GROUP_N: tl.constexpr, GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """One program owns Y[BLOCK_M, BLOCK_N] and reduces over all K tiles.

    X is [M,K], W is STORED [N,K], S is [ceil(N/GROUP_N),ceil(K/GROUP_K)].
    Load W as a [BLOCK_K,BLOCK_N] tile for dot; do not transpose global memory.
    For each (k,n) lane, scale is S[n//GROUP_N,k//GROUP_K], even if a GEMM
    tile crosses several quantization groups. Form weight*scale in FP32,
    then cast to X.dtype.element_ty before tl.dot; accumulate in FP32.
    Mask M/N/K tails, including scale reads; store only valid Y elements.
    Output is contiguous [M,N]. Inputs are read-only. No scratch pointers.
    """
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + k_offsets
        x = tl.load(X + m[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=(m[:, None] < M) & (k[None, :] < K), other=0)
        # Read W[n,k] into a logical [K,N] tile: no global transpose or scratch.
        mask = (k[:, None] < K) & (n[None, :] < N)
        w = tl.load(W + n[None, :] * stride_wn + k[:, None] * stride_wk,
                    mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S + (n[None, :] // GROUP_N) * stride_sn
                    + (k[:, None] // GROUP_K) * stride_sk, mask=mask, other=0)
        # Quantization groups need not align with this program's GEMM tile.
        restored = (w * s).to(X.dtype.element_ty)
        acc = tl.dot(x, restored, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc,
             mask=(m[:, None] < M) & (n[None, :] < N))


def launch_fp8_linear(inputs, weight, scale, output, block_size):
    """Fixed, inspectable tiles before any performance tuning."""
    m, k = inputs.shape
    n = weight.shape[0]
    fp8_linear_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
        inputs, weight, scale, output, m, n, k,
        *inputs.stride(), *weight.stride(), *scale.stride(),
        *block_size, BLOCK_M=16, BLOCK_N=32, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )
