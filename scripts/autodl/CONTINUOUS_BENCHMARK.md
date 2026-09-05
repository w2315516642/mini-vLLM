# Fixed-concurrency generation benchmark

`LOAD_MODE=batch` (default) retains the original fixed-batch benchmark.
`LOAD_MODE=continuous` admits up to `BATCH_SIZE` requests and replaces completed
requests before the next engine step. No scheduler or kernel changes are needed.

```bash
export CUDA_DEVICES=0 TP_SIZE=1 DTYPE=bfloat16
export DATASET=/root/autodl-tmp/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
for mode in target dspark-static; do
  LOAD_MODE=continuous BENCH_MODE="$mode" BATCH_SIZE=16 NUM_BATCHES=10 \
  INPUT_LEN=512 OUTPUT_LEN=128 NUM_SPECULATIVE_TOKENS=3 \
  MAX_NUM_BATCHED_TOKENS=8192 \
  bash scripts/autodl/benchmark_generation.sh
done
```

Default results end in `-b16-continuous.json`, separate from fixed-batch results.
The JSON config records `load_mode` and `includes_prefill_and_drain`.
`NUM_BATCHES * BATCH_SIZE` still determines the total request count (160 above);
it no longer creates barriers between groups in continuous mode.

This is closed-loop fixed concurrency, not an arrival-rate benchmark or a
guarantee of 16 decoding rows per step. Newly admitted requests need prefill.
Per-request latency starts immediately before admission; unsubmitted work is
not counted as queueing time. The overall timing window includes initial fill,
all measured prefill, refill overhead, and the complete final drain. Warmup and
model loading remain excluded. No tail samples are discarded.

Compare Target and DSpark using the same load mode, prompts, concurrency,
output length, memory/token budgets, and profiling settings. Do not claim a
speedup by comparing continuous DSpark against fixed-batch Target. Keep the old
batch results to quantify the effect of workload scheduling separately.
Per-batch prefix priming is not supported in continuous mode.
