#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"
activate_minivllm_env
configure_cuda
enter_repo
export PYTHONPATH="tests${PYTHONPATH:+:${PYTHONPATH}}"

# Existing scheduler fixtures replace imports, so run modules in isolation.
for module in tests.test_benchmark_metrics tests.test_benchmark_runner \
  tests.test_benchmark_spec_stats tests.test_benchmark_rpc \
  tests.test_streaming tests.test_pd_streaming tests.test_pd_rpc \
  tests.test_dspark_verification tests.test_dspark_adaptive_planner \
  tests.test_prefix_cache_scheduler tests.test_pd_scheduler_handoff \
  tests.test_benchmark_generation_cuda; do
  python -m unittest -v "${module}"
done
