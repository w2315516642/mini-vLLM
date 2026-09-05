# B16 Worker 分段计时

在现有模型、DATASET 环境变量下运行：

```bash
BENCH_MODE=target bash scripts/autodl/profile_generation_stages.sh
BENCH_MODE=dspark-static NUM_SPECULATIVE_TOKENS=3 bash scripts/autodl/profile_generation_stages.sh
BENCH_MODE=dspark-adaptive NUM_SPECULATIVE_TOKENS=7 bash scripts/autodl/profile_generation_stages.sh
python -m benchmarks.summarize_stage_profile build/benchmarks/stages-*.json
```

脚本默认单卡、B16、512 输入、128 输出、2 批预热和 3 批测量。
沿用 benchmark_generation.sh 的模型、数据集和精度设置。
stage JSON 按 rank 保存每个执行步，不跨 TP rank 相加。
默认最多记录 2048 步，可通过 STAGE_PROFILE_MAX_STEPS 调整；
limit_reached 表示触及上限，后续执行不再创建事件。

## 字段与边界

- prepare_inputs：构造输入和挂接草稿概率。
- state_snapshot：获取状态槽及创建验证前状态快照。
- target_model：完整目标 forward，包含采样、验证和回放输入收集。
- draft_context_kv：收集目标 hidden states 并更新草稿上下文 KV。
- state_replay：判断拒绝、回放接受前缀及释放事务。
- draft_proposal：生成下一轮草稿，含元数据构造和结果转回 CPU。

stream_ms 是当前 CUDA stream 两个事件之间的墙钟时间，包含空闲、
CPU 供给不足和 stream 等待，不是纯 kernel 时间；host_ms 是同一阶段
的 CPU 墙钟时间，包含调用内的同步等待。两者重叠，不能相加。
事件仅在全部测量结束后统一同步读取，预热不计入。
框架调度、流式文本处理及 Worker 前半段的 cache swap 等不在分段范围内。

counts 保存每步活跃请求数、真实 Prefill 请求数、投机请求数、
有效输入 token 数、验证草稿 token 数、拒绝请求数和实际回放请求数。
请求计数是每步出现次数，汇总后不是唯一请求数。
汇总工具将真实 Prefill 或混合步与 Decode 步分开。

## 使用限制

这是诊断运行，有额外事件及 Python 记录成本。正式吞吐仍用不带
--stage-profile-output 的 benchmark_generation.sh 测量。
目标是先比较阶段占比和回放行浪费，不能仅凭阶段 stream_ms 推断
某一个 CUDA kernel 的瓶颈；必要时再用 Nsight 时间线细分。
不支持同时 --prime-prefix，以免把额外预填充混入测量。
