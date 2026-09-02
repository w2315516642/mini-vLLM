#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"

activate_minivllm_env
configure_cuda
enter_repo

exec python -m benchmarks.benchmark_pd \
  --prefill-control "${PREFILL_CONTROL_ADDRESS:-127.0.0.1:15000}" \
  --decode-control "${DECODE_CONTROL_ADDRESS:-127.0.0.1:15100}" \
  --control-authkey "${PD_AUTHKEY:-mini-vllm}" \
  --prompt "${PD_PROMPT:-Explain paged KV cache briefly.}" \
  --requests "${PD_BENCHMARK_REQUESTS:-10}" \
  --warmup "${PD_BENCHMARK_WARMUP:-2}" \
  --max-tokens "${PD_MAX_TOKENS:-64}" \
  "$@"
