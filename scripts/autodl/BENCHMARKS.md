# 简历指标的实测方法

这套 bench 使用当前 `codex/benchmark` 的流式输出。此前简历里的 4.6、
58 tokens/s、40.9%、24 ms 均为占位数字，不是基准结果，也不是本脚本的目标值。
脚本只报告实际完成的请求，不会补造 Token 时间或按预期缩放结果。

## 入口与改动范围

| Python 入口 | Shell 入口 | 用途 |
| --- | --- | --- |
| `benchmarks.benchmark_generation` | `benchmark_generation.sh` | 单请求、批量推理、DSpark |
| `benchmarks.serve_engine` | `start_benchmark_server.sh` | 启动独立的混部/P/D 实例 |
| `benchmarks.benchmark_serving` | `benchmark_serving.sh` | 同一时间表下的长短请求混合负载 |
| `benchmarks.compare_results` | `compare_benchmarks.sh` | 校验对比条件、计算提升比例 |

Shell 脚本均在 `scripts/autodl/` 下，沿用 `common.sh` 的 Conda、CUDA 配置。
默认环境为 `vllm`，默认 CUDA Toolkit 路径为 `/usr/local/cuda-13.0`，按机器实际值覆盖。
本次不用重新编译 CUDA；切换到另一种 GPU 时仍需保证扩展包含其架构。

生产代码仅新增调度器投机计数、只读统计 RPC 和可选 RPC 超时。
并发压测循环在 `benchmarks/` 内，没有新增后台推理服务或修改调度策略。

## 指标口径

- `metrics.ttft_ms`：提交到第一个 Token 的时间，包含排队，不包含加载与预热。
  输入预先完成分词，因此不包含客户端分词或聊天模板处理。
- `metrics.decode_tokens_per_s`：每请求 `(N-1)/(末 Token 时间-首 Token 时间)`，
  报告均值和分位数。单请求只有一个输出 Token 时为 `null`。
- `metrics.output_tokens_per_s`：窗口内实际输出 Token 总数除以完整测量时长，
  包含 Prefill、排队与排空，不等于各请求速度相加。
- `metrics.tpot_ms`：每请求首、末 Token 间隔除以 `N-1`。PD 的迁移等待也在其中。
- `metrics.itl_ms.p99`：将所有请求的相邻 Token 间隔合并后取最近秩 P99，
  不是各请求 P99 的均值。必须同时查看 `count`，建议累计至少一万个间隔并重复实验。
- `speculative.accepted_tokens_per_round`：接受的草稿数/请求级验证轮数，
  不含锚点、修正 Token 和 bonus；接受 EOS 的边界也单独正确计数。
- `speculative.acceptance_rate`：接受数/实际送入验证的草稿数，
  自适应宽度下不会固定用 7 作分母。
- `metrics.handoff_ms`：一次完整交接的客户端墙钟耗时，包含资源预留、布局查询、
  数据传输、确认、激活及源端释放，不是纯网络传输延迟。

DSpark/MTP 一次更新可能包含多个 Token：此时 `itl_ms` 为 `null`，另外报告
`inter_update_ms`。不能把一批 Token 拆成多个零间隔来获得看似更低的 P99。
PD 的 ITL 消融先不启用投机，DSpark 主要比较解码速度与接受长度。

## 准备数据

自然语言测量支持 JSONL：

```json
{"prompt": "问题及足够长的参考上下文……"}
{"prompt": "另一条足够长的输入……"}
```

也支持 JSON 数组、项目原有 `{"prompts": [...]}` 记录和 ShareGPT 的
`conversations` 格式；ShareGPT 只取首轮输入，不拼入参考答案。
使用固定 seed 筛选足够长的输入并截断到精确长度，不自动重复文本补长。
数据不足以提供指定长度时直接报错；可用候选少于请求数时会确定性重复采样，
结果中的 `workload.unique_prompts` 会如实反映。

`SYNTHETIC=1` 显式生成随机非特殊 Token，只用于容量/入口 smoke test。
这种数据上的接受率不能当成自然语言 DSpark 效果。正式数字使用真实数据，
两组保持相同的 tokenizer、输入、输出长度与采样设置。

```bash
conda activate vllm
cd /root/autodl-tmp/mini-vLLM
export DATASET=/root/autodl-tmp/data/prompts.jsonl
export TARGET_MODEL=/root/autodl-tmp/models/Qwen3.8-27B-FP8
export DRAFT_MODEL=/root/autodl-tmp/models/Qwen3.8-27B-DSpark
export CUDA_DEVICES=0,1 TP_SIZE=2
```

## 单请求、16 并发与 DSpark

默认输入 2048 Token、输出 128 Token、贪心采样，忽略 EOS 以固定工作量。
每个模式独立启动进程，避免前一个模型占用显存。测量时不打印生成文本。

```bash
BENCH_MODE=target BATCH_SIZE=1 bash scripts/autodl/benchmark_generation.sh
BENCH_MODE=target BATCH_SIZE=16 bash scripts/autodl/benchmark_generation.sh
BENCH_MODE=dspark-static BATCH_SIZE=1 bash scripts/autodl/benchmark_generation.sh
BENCH_MODE=dspark-adaptive BATCH_SIZE=1 bash scripts/autodl/benchmark_generation.sh

bash scripts/autodl/compare_benchmarks.sh \
  build/benchmarks/target-b1.json build/benchmarks/dspark-static-b1.json
bash scripts/autodl/compare_benchmarks.sh \
  build/benchmarks/target-b1.json build/benchmarks/dspark-adaptive-b1.json
```

看 `target-b1.json` 的 TTFT 和解码速度；看 `target-b16.json` 的
`output_tokens_per_s` 作为 16 个同时提交请求的聚合吞吐。这里是重复批量测量，
不是在线稳定到达负载下的服务容量。

`NUM_BATCHES`、`WARMUP`、`INPUT_LEN`、`OUTPUT_LEN`、`MAX_NUM_SEQS`、
`MAX_NUM_BATCHED_TOKENS`、`GPU_MEMORY_UTILIZATION` 均可覆盖。
不能给两个速度对照组使用不同的精度、并发上限或 Token 预算而不注明。
自适应规划目前默认使用线性代价模型，并非自动完成硬件代价标定。

## PD 与混部的公平对比

以下示例需要四张能容纳 TP=2 副本的 GPU。两组都使用同样四张卡：
混部为两个 TP=2 副本，PD 为 P 两卡、D 两卡。不要同时运行基线与 PD 组。
RPC 使用 pickle，只能放在受信网络；每套 endpoint 只允许一个 benchmark 驱动者。

先在两个终端分别启动混部实例：

```bash
BENCH_NAME=u0 BENCH_ROLE=unified CUDA_DEVICES=0,1 TP_SIZE=2 \
  CONTROL_ADDRESS=127.0.0.1:15000 bash scripts/autodl/start_benchmark_server.sh
```

```bash
BENCH_NAME=u1 BENCH_ROLE=unified CUDA_DEVICES=2,3 TP_SIZE=2 \
  CONTROL_ADDRESS=127.0.0.1:15100 bash scripts/autodl/start_benchmark_server.sh
```

等两个终端都打印 `Benchmark server ready`，在第三个终端执行：

```bash
BENCH_TOPOLOGY=unified SHORT_OUTPUT_LEN=2048 \
  bash scripts/autodl/benchmark_serving.sh
```

停止两个混部进程，确认显存已释放，再分别启动 P、D：

```bash
BENCH_NAME=p BENCH_ROLE=prefill CUDA_DEVICES=0,1 TP_SIZE=2 \
  CONTROL_ADDRESS=127.0.0.1:15000 bash scripts/autodl/start_benchmark_server.sh
```

```bash
BENCH_NAME=d BENCH_ROLE=decode CUDA_DEVICES=2,3 TP_SIZE=2 \
  CONTROL_ADDRESS=127.0.0.1:15100 bash scripts/autodl/start_benchmark_server.sh
```

```bash
BENCH_TOPOLOGY=pd SHORT_OUTPUT_LEN=2048 \
  bash scripts/autodl/benchmark_serving.sh
bash scripts/autodl/compare_benchmarks.sh \
  build/benchmarks/unified.json build/benchmarks/pd.json
```

默认先提交 8 个 128 Token 短请求；1 秒后起，每 0.1 秒注入一个 4096 Token
长请求，共 8 个。长请求默认只输出 1 个 Token，主要制造 Prefill 负载。
P 的首 Token 在迁移前记录，D 的累计输出不会重复计数；P 驱动忙于 Prefill
或交接时，D 的独立驱动仍能推进已接管请求。基线则轮询分发到两个混部实例。

简历中的 P99 对应 `cohorts.decode.itl_ms.p99`，不要混入背景长请求。
如果提示无重叠，先调 `LONG_DELAY` 或增加 `SHORT_OUTPUT_LEN`，然后重跑两组。
重叠标记表示请求时间窗口重叠，不等于 profiler 证明 CUDA kernel 同时运行。
不要把短请求数量提高到 D 状态槽容量之外；当前交接层不是无限队列的在线服务。

`LONG_OUTPUT_LEN=128` 可将背景请求也改为完整生成，此时要为 D 留出更多状态槽。
跨 rank 数据端口从 14000/14100 开始，分别连续占用 TP 个端口，不要与控制端口冲突。
若 RPC 超时，测试报错而不是忽略该请求；重启相关服务，避免遗留状态污染下一次测量。
`--rpc-timeout` 默认 300 秒，`--timeout` 默认限制每轮总时间为 600 秒，长实验可显式调高。
Prefill 分块可能暂时没有输出，驱动不会因此插入额外 sleep；无进展由超时检测处理。

单 A800 80GB 上可以让 P、D 都用 `CUDA_DEVICES=0 TP_SIZE=1`，各进程降低
`GPU_MEMORY_UTILIZATION`，先做功能验证；脚本允许记录结果，但对比程序会拒绝将
共享同一 GPU 的进程当成独占 P/D 资源的性能证据。两张卡各跑 TP=1 也可以测试，
前提是每张卡都放得下完整副本，不能与四卡 TP=2 的数字混比。

## Prefix Cache 与 Chunked Prefill 的边界

Qwen 混合模型目前明确禁用了 Prefix Cache。不要为了跑数绕过该检查。
在支持缓存复用的 full-attention 模型上，可以使用 `PREFIX_PRIME=1`，
在每批计时前执行一次相同输入的预填充；结果需注明是预热缓存场景，
且有限缓存容量可能导致部分内容被淘汰。

```bash
# 换成受支持的 full-attention 模型，并保持两组 GPU/TP/精度/输入相同。
BENCH_OUTPUT=build/benchmarks/prefix-off.json \
  bash scripts/autodl/benchmark_generation.sh
PREFIX_PRIME=1 BENCH_OUTPUT=build/benchmarks/prefix-warm.json \
  bash scripts/autodl/benchmark_generation.sh
bash scripts/autodl/compare_benchmarks.sh \
  build/benchmarks/prefix-off.json build/benchmarks/prefix-warm.json \
  --allow-config-difference enable_prefix_caching
```

这得到的是本引擎关闭/预热 Prefix Cache 的对照，不是“较 HF 降低 76.4%”。
HF 对照尚需独立的相同条件基线，本次不生成那个百分比。
Chunked Prefill 目前无关闭开关，可以通过混部服务的 Token 预算做对照，
对比时显式允许 `max_num_batched_tokens` 差异，但必须标注为预算实验，
不能称为严格的开启/关闭消融。本次未为跑分新增调度策略。

## 结果与复核

默认输出到已忽略的 `build/benchmarks/`：汇总 `.json` 和逐请求
`.requests.jsonl`。后者记录每次输出的相对时间、Token 数、输出 Token 哈希及
首/末时间指标；不保存原始输入文本或生成文本。汇总记录测量窗口、
输入摘要、GPU UUID/显存/驱动、Torch/CUDA、Git 提交、dirty 状态与服务配置。

对比程序要求相同输入和时间表、相同总 GPU UUID、精度、运行时及主要配置。
只对有定义的指标计算百分比；多 Token 输出没有 ITL 时不强行比较。
直接调用 Python 时也需设置 `CUDA_VISIBLE_DEVICES`，使用数字 GPU 编号时设置
`CUDA_DEVICE_ORDER=PCI_BUS_ID`。Shell 入口已设置它；MIG 等映射需要单独核实。
建议测量前提交代码，并对同一配置重复数轮，保留全部结果而不是只挑最好的一次。

运行本次统计、RPC 和相关回归测试，无需 CUDA 重编译：

```bash
bash scripts/autodl/test_benchmarks.sh
```

本机已在 WSL2 的 `mini-vllm` 环境验证统计和真实本机 RPC 链路。
还可显式启用极小随机权重 Qwen 混合模型，覆盖 shell 入口、实际 GPU 推理和结果落盘：

```bash
MINIVLLM_RUN_BENCHMARK_CUDA_TESTS=1 bash scripts/autodl/test_benchmarks.sh
```

该 smoke test 离线创建模型与分词器，不下载权重，不编译扩展，也不提供 27B 性能结论。
