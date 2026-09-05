"""Tiled Markov projection with only tile maxima written to global memory."""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Base, Previous, Weight, Scores, Ids,
             B: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
             BS: tl.constexpr, PS: tl.constexpr, WS: tl.constexpr,
             NT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for offset in range(triton.cdiv(R, BK)):
        k = offset * BK + rk
        a = tl.load(Previous + rows[:, None] * PS + k[None, :],
                    (rows[:, None] < B) & (k[None, :] < R), 0)
        w = tl.load(Weight + cols[None, :] * WS + k[:, None],
                    (cols[None, :] < V) & (k[:, None] < R), 0)
        acc = tl.dot(a, w, acc)
    base = tl.load(Base + rows[:, None] * BS + cols[None, :],
                   (rows[:, None] < B) & (cols[None, :] < V), 0)
    scores = tl.where(cols[None, :] < V, acc + base, -float('inf'))
    best = tl.max(scores, 1)
    ids = tl.min(tl.where((scores == best[:, None]) & (cols[None, :] < V),
                          cols[None, :], 2147483647), 1)
    dst = rows * NT + tl.program_id(1)
    tl.store(Scores + dst, best, rows < B)
    tl.store(Ids + dst, ids, rows < B)


@triton.jit
def _finish(Scores, Ids, Output, NT: tl.constexpr, BLOCK: tl.constexpr):
    tiles = tl.arange(0, BLOCK)
    offsets = tl.program_id(0) * NT + tiles
    scores = tl.load(Scores + offsets, tiles < NT, -float('inf'))
    ids = tl.load(Ids + offsets, tiles < NT, 2147483647)
    best = tl.max(scores, 0)
    chosen = tl.min(tl.where(scores == best, ids, 2147483647), 0)
    tl.store(Output + tl.program_id(0), chosen)


def tiled_markov_argmax(base, previous, weight):
    """FP16/BF16 dot with FP32 accumulation; ties select the smallest ID.

    Reduction order differs from the scalar CUDA implementation. Near-tied
    scores can select a different proposal; target verification is unchanged.
    """
    b, v = base.shape
    rank = previous.shape[1]
    tiles = triton.cdiv(v, 128)
    scores = torch.empty((b, tiles), dtype=torch.float32, device=base.device)
    ids = torch.empty((b, tiles), dtype=torch.int32, device=base.device)
    output = torch.empty(b, dtype=torch.int64, device=base.device)
    _partial[(triton.cdiv(b, 16), tiles)](
        base, previous, weight, scores, ids, b, v, rank,
        base.stride(0), previous.stride(0), weight.stride(0), tiles,
        16, 128, 32, num_warps=4)
    _finish[(b,)](scores, ids, output, tiles, triton.next_power_of_2(tiles))
    return output
