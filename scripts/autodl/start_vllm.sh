#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
CONDA_ENV="${UPSTREAM_ENV:-/root/autodl-tmp/envs/vllm-upstream}"
activate_minivllm_env
configure_cuda
enter_repo
mode="${BENCH_MODE:-dspark}"
spec_args=()
case "$mode" in
  target) ;;
  dspark)
    spec=$(python -c 'import json,sys; print(json.dumps(dict(method="dspark",model=sys.argv[1],num_speculative_tokens=3)))' "$DRAFT_MODEL")
    spec_args+=(--speculative-config "$spec") ;;
  *) echo 'BENCH_MODE must be target or dspark' >&2; exit 2 ;;
esac
exec env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID vllm serve "$TARGET_MODEL" \
  --served-model-name target --host 127.0.0.1 --port "${PORT:-8000}" \
  --dtype bfloat16 --tensor-parallel-size 1 --max-model-len 2048 \
  --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --generation-config vllm --seed 42 "${spec_args[@]}"
