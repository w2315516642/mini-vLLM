#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
CONDA_ENV="${UPSTREAM_ENV:-/root/autodl-tmp/envs/vllm-upstream}"
activate_minivllm_env
enter_repo
mode="${BENCH_MODE:-dspark}"
exec python -m benchmarks.benchmark_vllm_serving \
  --url "${VLLM_URL:-http://127.0.0.1:8000}" --mode "$mode" \
  --tokenizer "$TARGET_MODEL" \
  --concurrency "${BATCH_SIZE:-1}" \
  --dataset "${DATASET:-$REPO_ROOT/build/datasets/sharegpt-heldout-100.jsonl}" \
  --output "${BENCH_OUTPUT:-build/benchmarks/upstream-${mode}-b${BATCH_SIZE:-1}.json}" \
  "$@"
