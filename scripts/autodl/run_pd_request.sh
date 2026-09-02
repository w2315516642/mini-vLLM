#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"

activate_minivllm_env
configure_cuda
enter_repo

exec python -m minivllm.entrypoints.pd_generate \
  --prefill-control "${PREFILL_CONTROL_ADDRESS:-127.0.0.1:15000}" \
  --decode-control "${DECODE_CONTROL_ADDRESS:-127.0.0.1:15100}" \
  --control-authkey "${PD_AUTHKEY:-mini-vllm}" \
  --prompt "${PD_PROMPT:-介绍一下 Prefill Decode 分离。}" \
  --max-tokens "${PD_MAX_TOKENS:-64}" \
  "$@"
