#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
configure_cuda
enter_repo
command -v nsys >/dev/null || { echo "nsys is not on PATH" >&2; exit 1; }

export CUDA_DEVICES="${CUDA_DEVICES:-0}" TP_SIZE="${TP_SIZE:-1}"
export BATCH_SIZE="${BATCH_SIZE:-16}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export INPUT_LEN="${INPUT_LEN:-512}" OUTPUT_LEN="${OUTPUT_LEN:-128}"
export NUM_BATCHES="${NUM_BATCHES:-1}" WARMUP="${WARMUP:-2}"
name="${BENCH_MODE:-target}-b${BATCH_SIZE}-k${NUM_SPECULATIVE_TOKENS:-7}"
prefix="${NSYS_OUTPUT:-build/benchmarks/nsys-${name}}"
mkdir -p "$(dirname -- "$prefix")"
export BENCH_OUTPUT="${BENCH_OUTPUT:-${prefix}-generation.json}"
# cudaProfilerStart/Stop bracket only the selected Decode window. The process
# continues after collection, allowing result files and cleanup to finish.
exec nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
  --output "$prefix" \
  bash scripts/autodl/benchmark_generation.sh \
    --nsys-capture-output "${prefix}-window.json" \
    --nsys-skip-steps "${NSYS_SKIP_STEPS:-5}" \
    --nsys-capture-steps "${NSYS_CAPTURE_STEPS:-20}" "$@"
