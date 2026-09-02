#!/usr/bin/env bash

set -euo pipefail
source "$(dirname -- "$0")/common.sh"

activate_minivllm_env
configure_cuda
enter_repo
export PYTHONPATH=tests

run_test_modules() {
  local module
  for module in "$@"; do
    echo "-- ${module}"
    python -m unittest -v "${module}"
  done
}

export MINIVLLM_RUN_CUDA_GDN_TESTS=1
export MINIVLLM_RUN_CUDA_GQA_TESTS=1
export MINIVLLM_RUN_CUDA_QWEN_ATTENTION_TESTS=1
export MINIVLLM_RUN_CUDA_QWEN_HYBRID_TESTS=1
export MINIVLLM_RUN_CUDA_DSPARK_TESTS=1

echo "== Qwen3.8 Python/CUDA =="
run_test_modules \
  tests.test_model_architecture tests.test_model_registry \
  tests.test_fp8_weights tests.test_gated_delta_net_reference \
  tests.test_gated_delta_net_cuda_contract tests.test_gated_delta_net_cuda \
  tests.test_gqa tests.test_gqa_cuda \
  tests.test_qwen_gated_attention tests.test_qwen_gated_attention_cuda \
  tests.test_qwen_hybrid_model tests.test_qwen_hybrid_model_cuda \
  tests.test_qwen_multimodal

echo "== DSpark Python/CUDA =="
run_test_modules \
  tests.test_dspark_adaptive_planner tests.test_dspark_cache \
  tests.test_dspark_heads tests.test_dspark_markov_cuda \
  tests.test_dspark_model tests.test_dspark_model_cuda \
  tests.test_dspark_rejection_sampler tests.test_dspark_runtime \
  tests.test_dspark_verification

echo "== Prefill/Decode disaggregation =="
run_test_modules \
  tests.test_pd_transfer_contract tests.test_pd_cache_layout \
  tests.test_pd_control_plane tests.test_pd_p2p_backend \
  tests.test_pd_qwen_continuity tests.test_pd_engine_bridge \
  tests.test_pd_rpc tests.test_pd_scheduler_handoff \
  tests.test_pd_topology tests.test_pd_transfer_manager
