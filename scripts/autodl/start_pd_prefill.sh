#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"

activate_minivllm_env
configure_cuda
enter_repo

draft_args=()
if [[ "${PD_ENABLE_DSPARK:-0}" == "1" ]]; then
  draft_args+=(
    --draft-model "${DRAFT_MODEL}"
    --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS:-7}"
  )
  if [[ "${SPECULATIVE_ADAPTIVE:-1}" == "1" ]]; then
    draft_args+=(--speculative-adaptive)
  fi
fi

exec env CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_DEVICES:-0}" python \
  -m minivllm.entrypoints.pd_server \
  --model "${PD_MODEL:-${TARGET_MODEL}}" \
  --dtype "${PD_DTYPE:-auto}" \
  --tensor-parallel-size "${PD_TP_SIZE:-1}" \
  --gpu-memory-utilization "${PD_GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-batched-tokens "${PD_MAX_NUM_BATCHED_TOKENS:-1024}" \
  --max-num-seqs "${PD_MAX_NUM_SEQS:-4}" \
  --pd-role prefill --pd-transfer-backend tcp \
  --pd-endpoint-id "${PREFILL_ENDPOINT_ID:-p}" \
  --pd-hostname "${PREFILL_DATA_ADDRESS:-127.0.0.1:14000}" \
  --pd-peer-endpoint-id "${DECODE_ENDPOINT_ID:-d}" \
  --pd-peer-hostname "${DECODE_DATA_ADDRESS:-127.0.0.1:14100}" \
  --control-address "${PREFILL_CONTROL_ADDRESS:-127.0.0.1:15000}" \
  --control-authkey "${PD_AUTHKEY:-mini-vllm}" \
  "${draft_args[@]}" "$@"
