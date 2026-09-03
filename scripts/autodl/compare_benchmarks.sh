#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
enter_repo
exec python -m benchmarks.compare_results "$@"
