# vLLM / mini-vLLM 对照矩阵

本轮固定原版 DSpark，比较 2 框架 x 2 并发(B1/B16) x 2 模式(target/DSpark K3)。
客户端使用同一份代码、相同 token ID、贪心采样、512 输入和 128 输出，独立预热两批。
两轮交换顺序；每组独立启动服务。默认 token budget 8192，区别于之前 B1 的 2048，须重跑基线。
CUDA Graph、量化等保留各框架正常路径，不保证底层优化配置相同；完整启动日志用于追溯。

## 准备

在 codex/benchmark 分支根目录运行。上游环境默认 `/root/autodl-tmp/envs/vllm-upstream`；
mini 环境默认 `vllm`。可分别设置 UPSTREAM_ENV、CONDA_ENV。
mini 服务只依赖自身推理环境，HTTP 适配器用标准库；客户端在上游环境运行。
保持原有 100 条文件作为 B1 输入；为 B16 准备 512 条独立输入：

```bash
conda activate /root/autodl-tmp/envs/vllm-upstream
python -m benchmarks.prepare_heldout \
  --source /root/autodl-tmp/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --exclude /root/autodl-tmp/SpecForge/cache/dataset/sharegpt-smoke/sharegpt_train_regen.jsonl \
    /root/autodl-tmp/SpecForge/cache/dataset/sharegpt-smoke/sharegpt_eval_regen.jsonl \
  --tokenizer /root/autodl-tmp/models/Qwen3.8-27B-FP8 \
  --count 512 --output build/datasets/sharegpt-heldout-512.jsonl
```

停止之前手动启动的服务，确认 GPU 空闲。8000 端口被占用时脚本会拒绝运行，不会杀其他进程。
确认当前代码和权重不在运行中变更。矩阵较耗时，不要在小 smoke 请求上声称正式性能结论。

```bash
bash scripts/autodl/benchmark_matrix.sh \
  --b1-dataset build/datasets/sharegpt-heldout-100.jsonl \
  --b16-dataset build/datasets/sharegpt-heldout-512.jsonl \
  --output-dir build/benchmarks/matrix-r1
```

输出目录必须不存在；默认完整两轮，共 16 次服务启动。
先跑一部分可加 `--frameworks vllm --batches 1 --rounds 1`，以后用新的输出目录跑完整矩阵。
每组保留结果 JSON、命令配置和服务日志。失败停止矩阵，保留已完成数据，不自动更换参数。
脚本仅终止自己启动的独立进程组。

B16 是闭环并发，完成后立即补充下一条，不是整批屏障。
结果包含每条请求的提交/完成时间及客户端 inflight 变化，包含最后收尾，不删慢样本。
客户端 inflight 不等于 GPU 实际 batch，GPU batch/验证形状必须结合服务日志和 profiler 检查。
HTTP 适配器会产生开销，但没有修改 mini-vLLM 调度器、模型或 CUDA 算子。
mini 适配器仅绑定 127.0.0.1，只接受固定工作负载，不能作为通用生产服务。

## 汇总

完整运行后自动生成 summary.md 和 token_comparison.json；也可单独运行：

```bash
python -m benchmarks.summarize_framework_matrix build/benchmarks/matrix-r1
```

逐轮展示吞吐/TPOT/TTFT/接收长度，计算框架内加速比和跨框架吞吐比。
比较前检查 workload，保存输出首个 token 分歧位置，不能把有生成差异的样本直接视作完全等价。
不推断逐 token ITL，不自动将吞吐差距归因于某个算子。

## 独立 Nsight 采样

确保 nsys 在 PATH 中，仍然先停止已有服务。使用新目录：

```bash
bash scripts/autodl/benchmark_matrix.sh \
  --b1-dataset build/datasets/sharegpt-heldout-100.jsonl \
  --b16-dataset build/datasets/sharegpt-heldout-512.jsonl \
  --output-dir build/benchmarks/matrix-profile --profile --rounds 1
```

每组预热后开启 CUDA profiler，计量 B1 两条 / B16 三十二条请求后关闭。
捕获包含 prefill 和 drain 的短窗口，不是假称“纯稳态 decode”；图中需要选取相同活跃 batch 的 decode 范围。
上游使用官方 `--profiler-config.profiler cuda` 和 /start_profile、/stop_profile。
mini 适配器在唯一引擎线程调用 cudaProfilerStart/Stop，仅适用于本脚本的 TP1 本地 worker。
采样报告与性能报告严格分目录。nsys 版本、CUDA Graph node tracing 支持须在 AutoDL 验证。

分析顺序：先 target 的 Linear/GDN/Attention/lm_head/空闲间隔，再 DSpark 的 draft/verify/恢复。
同时看调用次数、绝对耗时和输入形状，不相加重叠的 CPU NVTX 与 GPU 时间。
本地逻辑测试不代表 A800 端到端验证，真实结论须由生成的结果与报告支持。

参考：https://docs.vllm.ai/en/stable/contributing/profiling/
