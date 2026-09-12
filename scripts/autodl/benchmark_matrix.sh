#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
CONDA_ENV="${UPSTREAM_ENV:-/root/autodl-tmp/envs/vllm-upstream}"
activate_minivllm_env
enter_repo
exec python -m benchmarks.run_framework_matrix "$@"
