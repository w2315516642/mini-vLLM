"""Compare operator paths on identical data, excluding JIT and state resets.

Default Linear shapes are representative matrices, not a claimed model trace.
Pass --linear-shape N,K to measure a particular local TP weight shard.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import torch
import torch.nn.functional as F

from minivllm.model_executor.layers.fp8_linear import fp8_linear
from minivllm.model_executor.layers.gated_delta_net_cuda import (
    gated_delta_rule_varlen, prepare_gated_delta_qk,
)
from minivllm.model_executor.parallel_utils.tensor_parallel.layers import (
    FP8_FUSED_MAX_TOKENS, dequantize_fp8_block_weight,
)


def measure(fn, repeats, reset=None):
    for _ in range(3):
        if reset:
            reset()
        fn()
    torch.cuda.synchronize()
    times = []
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        if reset:
            reset()
        begin.record()
        fn()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end))
    return {"median_ms": statistics.median(times), "mean_ms": statistics.mean(times)}


def compare(old, new, repeats, reset=None):
    # Use both orders to reduce bias from clock/thermal drift.
    a, b = measure(old, repeats, reset), measure(new, repeats, reset)
    b2, a2 = measure(new, repeats, reset), measure(old, repeats, reset)
    baseline = (a["median_ms"] + a2["median_ms"]) / 2
    optimized = (b["median_ms"] + b2["median_ms"]) / 2
    return {"baseline_ms": baseline, "optimized_ms": optimized,
            "speedup": baseline / optimized}


def linear_case(m, n, k, dtype, repeats):
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand((n + 127) // 128, (k + 127) // 128, device="cuda") * .02
    old = lambda: F.linear(x, dequantize_fp8_block_weight(weight, scale, (128, 128), dtype))
    new = lambda: fp8_linear(x, weight, scale, (128, 128))
    expected, actual = old(), new()
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)
    error = (actual.float() - expected.float()).square().mean().sqrt().item()
    del expected, actual
    result = compare(old, new, repeats)
    # Peak allocation above live inputs includes output and split-K scratch.
    memory = {}
    for name, fn in (("baseline", old), ("optimized", new)):
        torch.cuda.synchronize()
        live = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = fn()
        torch.cuda.synchronize()
        memory[name + "_extra_mib"] = (torch.cuda.max_memory_allocated() - live) / 2**20
        del output
    return {"operator": "fp8_linear", "shape_mnk": [m, n, k], "rmse": error,
            "runtime_backend": "fused" if m <= FP8_FUSED_MAX_TOKENS else "dequantize_cublas",
            **result, **memory}


def gdn_case(batch, length, heads, dim, dtype, repeats):
    from minivllm import gated_delta_net_ops as ops
    q, k = prepare_gated_delta_qk(
        torch.randn(batch * length, heads, dim, device="cuda", dtype=dtype),
        torch.randn(batch * length, heads, dim, device="cuda", dtype=dtype))
    v = torch.randn_like(q)
    g = -torch.rand(batch * length, heads, device="cuda") * .01
    beta = torch.rand_like(g, dtype=dtype)
    initial = torch.randn(batch, heads, dim, dim, device="cuda") * .1
    state = initial.clone()
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length
    output = torch.empty_like(v)

    def old():
        ops.gated_delta_rule_varlen(output, q, k, v, g, beta, state, cu, length, None)
        return output

    def new():
        return gated_delta_rule_varlen(q, k, v, g, beta, state, cu, length)

    reset = lambda: state.copy_(initial)
    old()
    expected_state = state.clone()
    reset()
    actual = new()
    torch.testing.assert_close(actual, output, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=2e-3)
    result = compare(old, new, repeats, reset)
    return {"operator": "gdn_prefill", "shape_bthd": [batch, length, heads, dim], **result}


def matrix_shape(value):
    try:
        n, k = map(int, value.split(","))
        if min(n, k) > 0:
            return n, k
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Linear shape must be positive N,K")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("all", "linear", "gdn"), default="all")
    parser.add_argument("--linear-shape", type=matrix_shape, action="append")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 512])
    parser.add_argument("--gdn-lengths", type=int, nargs="+", default=[16, 64, 512])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=48)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="build/benchmarks/fused-operators.json")
    args = parser.parse_args()
    if min(*args.tokens, *args.gdn_lengths, args.batch_size, args.heads,
           args.head_dim, args.repeats) <= 0:
        parser.error("Shapes and repeat count must be positive")
    torch.manual_seed(483)
    dtype = getattr(torch, args.dtype)
    rows = []
    if args.operator in ("all", "linear"):
        for n, k in args.linear_shape or [(5120, 5120), (17408, 5120)]:
            for m in args.tokens:
                row = linear_case(m, n, k, dtype, args.repeats)
                rows.append(row)
                print(json.dumps(row), flush=True)
    if args.operator in ("all", "gdn"):
        for length in args.gdn_lengths:
            row = gdn_case(args.batch_size, length, args.heads, args.head_dim, dtype, args.repeats)
            rows.append(row)
            print(json.dumps(row), flush=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    commit = git.stdout.strip() if git.returncode == 0 else None
    result = {"gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "git_commit": commit,
              "config": vars(args), "results": rows,
              "git_error": git.stderr.strip() if git.returncode else None}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
