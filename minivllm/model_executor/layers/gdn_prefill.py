"""State-resident GDN prefill for packed or rectangular token batches.

Time is still recurrent. Parallel reductions over K replace serial per-thread
K loops, and each CTA retains a disjoint value-column tile of state in FP32
registers across the entire sequence. State is loaded/stored only once.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill(Q, K, V, G, Beta, State, Out, Cu, Lengths,
             H: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
             T: tl.constexpr, PACKED: tl.constexpr, LIMITED: tl.constexpr,
             BK: tl.constexpr, BV: tl.constexpr):
    bh = tl.program_id(0)
    batch, head = bh // H, bh % H
    ik = tl.arange(0, BK)
    iv = tl.program_id(1) * BV + tl.arange(0, BV)
    state_ptr = State + bh * DK * DV + ik[:, None] * DV + iv[None, :]
    mask = (ik[:, None] < DK) & (iv[None, :] < DV)
    state = tl.load(state_ptr, mask, other=0)
    if PACKED:
        start = tl.load(Cu + batch)
        count = tl.load(Cu + batch + 1) - start
    else:
        start, count = batch * T, T
    if LIMITED:
        count = tl.minimum(count, tl.load(Lengths + batch))
    for token in range(count):
        offset = (start + token) * H + head
        q = tl.load(Q + offset * DK + ik, ik < DK, other=0).to(tl.float32)
        k = tl.load(K + offset * DK + ik, ik < DK, other=0).to(tl.float32)
        v = tl.load(V + offset * DV + iv, iv < DV, other=0).to(tl.float32)
        decay = tl.exp(tl.load(G + offset))
        beta = tl.load(Beta + offset).to(tl.float32)
        # State uses [K,V], matching HybridCache and PD snapshots exactly.
        state *= decay
        predicted = tl.sum(k[:, None] * state, axis=0)
        delta = beta * (v - predicted)
        state += k[:, None] * delta[None, :]
        result = tl.sum(q[:, None] * state, axis=0)
        tl.store(Out + offset * DV + iv, result, iv < DV)
    tl.store(state_ptr, state, mask)


def gdn_prefill(query, key, value, log_decay, beta, state,
                cu_seqlens=None, lengths=None):
    """Consume prepared Q/K and update the caller-owned FP32 state in place.

    Packed offsets/accepted lengths are device metadata validated by their CPU
    owner. No CUDA scalar reads or host synchronization are needed here.
    """
    packed = cu_seqlens is not None
    ndim = 3 if packed else 4
    if query.ndim != ndim or key.shape != query.shape or value.shape[:-1] != query.shape[:-1]:
        raise ValueError("GDN Q/K/V shapes must agree on tokens, batch and heads")
    if log_decay.shape != query.shape[:-1] or beta.shape != query.shape[:-1]:
        raise ValueError("GDN decay/beta shapes must match token and head dimensions")
    h, dk, dv = query.shape[-2], query.shape[-1], value.shape[-1]
    batch = cu_seqlens.numel() - 1 if packed else query.shape[0]
    if min(batch, h, dk, dv) <= 0 or state.shape != (batch, h, dk, dv):
        raise ValueError("GDN state must have shape [batch, heads, key_dim, value_dim]")
    tensors = (query, key, value, log_decay, beta, state)
    if not query.is_cuda or any(t.device != query.device or not t.is_contiguous() for t in tensors):
        raise ValueError("GDN tensors must be contiguous and share a CUDA device")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16) or any(
        t.dtype != query.dtype for t in (key, value, beta)
    ) or log_decay.dtype != torch.float32 or state.dtype != torch.float32:
        raise ValueError("GDN requires matching Q/K/V/beta dtypes and FP32 decay/state")
    for metadata in (cu_seqlens, lengths):
        if metadata is not None and (metadata.ndim != 1 or metadata.dtype != torch.int32
                                     or metadata.device != query.device or not metadata.is_contiguous()):
            raise ValueError("GDN metadata must be contiguous CUDA int32 vectors")
    if lengths is not None and lengths.numel() != batch:
        raise ValueError("GDN accepted lengths must have one entry per sequence")
    out = torch.empty_like(value)
    with torch.cuda.device(query.device):
        _prefill[(batch * h, triton.cdiv(dv, 16))](
            query, key, value, log_decay, beta, state, out, cu_seqlens, lengths,
            h, dk, dv, 0 if packed else query.shape[1], packed, lengths is not None,
            triton.next_power_of_2(dk), 16, num_warps=4, enable_fp_fusion=False)
    return out
