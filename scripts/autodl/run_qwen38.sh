#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"

activate_minivllm_env
configure_cuda
enter_repo

exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1}" python \
  "${SCRIPT_DIR}/run_generation.py" \
  --model "${TARGET_MODEL}" \
  --tensor-parallel-size "${TP_SIZE:-2}" \
  --dtype "${DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}" \
  --max-num-seqs "${MAX_NUM_SEQS:-4}" \
  --num-speculative-tokens "${QWEN_SPECULATIVE_TOKENS:-0}" \
  --prompt "${PROMPT:-用三句话说明投机解码。}" \
  --max-tokens "${MAX_TOKENS:-64}" \
  --warmup "${WARMUP:-0}" --requests "${REQUESTS:-1}" \
  "$@"
