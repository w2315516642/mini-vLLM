#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
configure_cuda
enter_repo
export CUDA_DEVICE_ORDER=PCI_BUS_ID

role="${BENCH_ROLE:-unified}"
role_args=(--pd-role "${role}")
case "${role}" in
  unified) ;;
  prefill)
    role_args+=(--pd-transfer-backend tcp --pd-endpoint-id p --pd-peer-endpoint-id d
      --pd-hostname "${PREFILL_DATA_ADDRESS:-127.0.0.1:14000}"
      --pd-peer-hostname "${DECODE_DATA_ADDRESS:-127.0.0.1:14100}")
    ;;
  decode)
    role_args+=(--pd-transfer-backend tcp --pd-endpoint-id d --pd-peer-endpoint-id p
      --pd-hostname "${DECODE_DATA_ADDRESS:-127.0.0.1:14100}"
      --pd-peer-hostname "${PREFILL_DATA_ADDRESS:-127.0.0.1:14000}")
    ;;
  *) echo "BENCH_ROLE must be unified, prefill, or decode" >&2; exit 2 ;;
esac
: "${BENCH_NAME:?Set a distinct BENCH_NAME, such as u0, u1, p, or d}"
: "${CONTROL_ADDRESS:?Set CONTROL_ADDRESS, such as 127.0.0.1:15000}"
exec env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1}" python -m benchmarks.serve_engine \
  --model "${TARGET_MODEL}" --dtype "${DTYPE:-bfloat16}" \
  --tensor-parallel-size "${TP_SIZE:-2}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
  --seed "${SEED:-42}" --control-address "${CONTROL_ADDRESS}" \
  --control-authkey "${PD_AUTHKEY:-mini-vllm}" \
  --info-file "${BENCH_INFO:-build/benchmarks/server-${BENCH_NAME}.json}" \
  "${role_args[@]}" "$@"
