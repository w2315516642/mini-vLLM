#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
mini_conda_env="$CONDA_ENV"
CONDA_ENV="${UPSTREAM_ENV:-/root/autodl-tmp/envs/vllm-upstream}"
activate_minivllm_env
# Keep the orchestrator's Python in upstream, but pass the mini environment
# to child launchers even when CONDA_ENV was exported by the caller.
export CONDA_ENV="$mini_conda_env"
enter_repo
exec python -m benchmarks.run_framework_matrix "$@"
