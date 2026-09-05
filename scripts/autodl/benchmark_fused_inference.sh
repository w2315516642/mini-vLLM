#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
configure_cuda
enter_repo
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# One GPU; benchmark actual TP shard shapes with --linear-shape N,K.
exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}" \
  python -m benchmarks.benchmark_fused_inference \
  --dtype "${DTYPE:-bfloat16}" --batch-size "${BATCH_SIZE:-16}" \
  --repeats "${REPEATS:-10}" \
  --output "${BENCH_OUTPUT:-build/benchmarks/fused-operators.json}" "$@"
