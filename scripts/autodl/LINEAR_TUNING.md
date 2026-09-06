# FP8 Linear launch configuration scan

This scan does not change runtime defaults or dequantize the resident model.
It compares candidate fused kernels with the current fused runtime, not with
the slower dequantize-plus-GEMM reference used by the normal operator benchmark.
All candidates are compiled and checked against that numerical reference before
timing. Timings include the split-K reduction and scratch/output allocation path.

```bash
CUDA_DEVICES=0 DTYPE=bfloat16 REPEATS=30 \
BENCH_OUTPUT=build/benchmarks/linear-tuning-a800.json \
bash scripts/autodl/benchmark_fused_inference.sh \
  --operator linear --tune-linear --tokens 16 32 48 64 \
  --linear-shape 5120,5120 --linear-shape 17408,5120
```

N,K above are representative shapes, not an exhaustive model-weight inventory.
Use `weight=(N,K)` from the Linear NVTX labels to scan other actual local TP
shards. Scan output projection and MLP down-projection shapes too before choosing
runtime configurations. The Python entry point accepts the same flags:
`python -m benchmarks.benchmark_fused_inference`.

The JSON contains every configuration (TM,TN,TK,split,warps,stages), error,
paired-order timing versus the baseline, scratch size and a provisional winner.
The runtime configuration itself is included as a timing control. Results are
warm-cache microbenchmarks, not model speedups. Repeat on A800 without concurrent
GPU work, and confirm winners in end-to-end Target and DSpark runs. Do not deploy
settings chosen on a different GPU solely because they won there. First-run JIT
may take time; no C++ extension rebuild is needed for this scan.
