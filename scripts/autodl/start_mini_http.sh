#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
configure_cuda
enter_repo
spec_args=()
case "${BENCH_MODE:-dspark}" in
  target) ;;
  dspark) spec_args+=(--draft-model "$DRAFT_MODEL" --num-speculative-tokens 3) ;;
  *) echo 'BENCH_MODE must be target or dspark' >&2; exit 2 ;;
esac
exec env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID python -m benchmarks.mini_http_server \
  --model "$TARGET_MODEL" --dtype bfloat16 --tensor-parallel-size 1 \
  --max-num-seqs "${BATCH_SIZE:-1}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization 0.85 --seed 42 --port "${PORT:-8000}" \
  --disable-log-stats "${spec_args[@]}"
