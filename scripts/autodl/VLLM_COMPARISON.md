# 上游 vLLM B1 对照

只测 target-only 和原版 DSpark，不涉及 ft20。安装好的独立环境默认是
`/root/autodl-tmp/envs/vllm-upstream`，可用 `UPSTREAM_ENV` 覆盖。
不需要在该环境安装 mini-vLLM，客户端只导入仓库的 benchmark 工具。
客户端依赖 requests、transformers、prometheus-client，通常由上游 vLLM 环境提供。

## 运行

在仓库根目录，终端一启动服务并保留完整启动日志：

```bash
mkdir -p build/benchmarks
set -o pipefail
BENCH_MODE=dspark bash scripts/autodl/start_vllm.sh 2>&1 | tee build/benchmarks/upstream-dspark-server.log
```

等 `/health` 返回 200，在终端二运行：

```bash
BENCH_MODE=dspark bash scripts/autodl/benchmark_vllm.sh
```

已经使用同样配置启动服务时，无需重启，可直接运行客户端。
默认数据是 `build/datasets/sharegpt-heldout-100.jsonl`。需要时用 DATASET 指定绝对路径。
客户端预热 2 次，再顺序测 100 条输入；B1、512 输入 token、128 输出 token、贪心采样。
不要同时运行其他客户端，否则全局计数器差值会被污染。

停止第一组服务，确认 GPU 释放后，终端一运行：

```bash
BENCH_MODE=target bash scripts/autodl/start_vllm.sh 2>&1 | tee build/benchmarks/upstream-target-server.log
```

终端二运行：

```bash
BENCH_MODE=target bash scripts/autodl/benchmark_vllm.sh
```

输出是 `build/benchmarks/upstream-{dspark,target}-b1.json`。已有结果不覆盖；
第二轮用 `BENCH_OUTPUT` 指定新路径，并交换两组运行顺序。
启动脚本不自动终止已有服务。客户端的 BENCH_MODE 不会改变服务配置。

## 解释结果

核对两组 workload SHA 和 mini-vLLM 的 heldout SHA 相同。
客户端使用相同采样函数并发送 token ID，不应用聊天模板；检查实际服务用的 tokenizer 与本地一致。
记录服务日志和 JSON 中服务版本，不能只用客户端 mode 标签证明完整配置一致。

先算上游 DSpark 吞吐 / 上游 target 吞吐，再与 mini-vLLM 自身的加速比比较。
平均接收长度不含额外目标 token，等于 accepted_draft_tokens / verification_rounds。
分位置计数为 position_0 等字段。结果保留输出 token ID，可按 request_id 比较两组输出。
不同算子数值差异可能导致贪心输出分歧，需要检查分歧而不是假定完全相同。

TTFT 从客户端发送请求到收到首批 token，E2E 到收到最后一批 token。
TPOT=(最后一批到达时间-首批到达时间)/127；不能据此推断每个 token 的到达时间或 ITL P99。
吞吐计时包括请求传输和消费完整响应，不包括预热、模型加载和指标等待。
因此不要将 HTTP 绝对时延直接等同于 mini-vLLM 本地 Python 接口时延。

Prometheus 指标可能延迟更新。脚本在预热后、计量后各等待 10 秒再读取，保存前后原始计数。
这不是服务端 flush 保证。若统计仍有滞后，先排查再用 `--metrics-settle-seconds 20` 重测。
不能把一小段 smoke 生成的累计计数当作正式评测结果。

参考：
- https://docs.vllm.ai/en/latest/features/per_request_metrics/
- https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dspark/
