# AutoDL 测试脚本

这些脚本用于 `codex/benchmark` 分支，覆盖 Qwen3.8、DSpark，以及项目自行
实现的 Mooncake 风格 Prefill/Decode 分离。默认 Conda 环境为 `vllm`，CUDA
目录为 `/usr/local/cuda-13.0`，模型目录为 `/root/autodl-tmp/models`。

## 脚本顺序

| 脚本 | 用途 |
| --- | --- |
| `check_env.sh` | 检查 Git、Torch、CUDA、依赖和扩展 |
| `build.sh` | 安装依赖并编译 editable CUDA 扩展 |
| `download_models.sh` | 下载 Qwen3.8 FP8 与 DSpark checkpoint |
| `test_features.sh` | 运行 Qwen、DSpark、PD 聚焦测试 |
| `run_qwen38.sh` | 运行纯 Target；可用 `QWEN_SPECULATIVE_TOKENS=1` 测 MTP |
| `run_dspark.sh` | 运行 Target + DSpark，并打印吞吐统计 |
| `start_pd_prefill.sh` | 启动 P 服务 |
| `start_pd_decode.sh` | 启动 D 服务 |
| `run_pd_request.sh` | 发送一个 PD 请求 |
| `benchmark_pd.sh` | 测量 PD 延迟、TTFT、TPOT 和传输耗时 |

## 初次准备

```bash
cd /root/autodl-tmp/mini-vLLM
git fetch origin
git switch codex/benchmark

bash scripts/autodl/check_env.sh
bash scripts/autodl/build.sh
bash scripts/autodl/download_models.sh
bash scripts/autodl/test_features.sh
```

如果依赖已经安装，可用 `INSTALL_REQUIREMENTS=0 bash scripts/autodl/build.sh`
只重新编译扩展。目标模型下载也可改走 Hugging Face：
`TARGET_SOURCE=huggingface bash scripts/autodl/download_models.sh`。

两张 24GB GPU 要做 PD smoke test 时，可额外下载小模型：

```bash
DOWNLOAD_TARGET=0 DOWNLOAD_DRAFT=0 DOWNLOAD_PD_SMOKE_MODEL=1 \
  bash scripts/autodl/download_models.sh
export PD_MODEL=/root/autodl-tmp/models/TinyLlama-1.1B-Chat-v1.0
```

## 统一模式

两张 24GB GPU 使用 TP=2：

```bash
bash scripts/autodl/run_qwen38.sh
ray stop --force
bash scripts/autodl/run_dspark.sh
```

设置 `WARMUP=2 REQUESTS=10` 可以让两个脚本运行多轮并输出平均生成吞吐。
提示词、输出长度和显存参数分别通过 `PROMPT`、`MAX_TOKENS`、
`GPU_MEMORY_UTILIZATION`、`MAX_NUM_SEQS` 覆盖。

### 流式输出

普通生成、MTP、DSpark 和 PD 请求均可加 `--stream`，已有非流式调用不变：

```bash
bash scripts/autodl/run_qwen38.sh --stream
bash scripts/autodl/run_dspark.sh --stream
bash scripts/autodl/run_pd_request.sh --stream
```

PD 仍需先启动 P、D 服务。更新本次 Python 代码后需要重启两个服务，但不需要
重新编译 CUDA 扩展。DSpark 的每次更新可能包含多个已通过验证的 token，
不会输出未接受的草稿。流式模式的耗时包含终端输出开销；对比吞吐时不要加
`--stream`。Python API、停止词缓冲和取消请求的说明见
[流式输出设计](STREAMING.md)。

## PD 模式

PD 需要三个终端。终端 1、2 分别运行：

```bash
bash scripts/autodl/start_pd_prefill.sh
bash scripts/autodl/start_pd_decode.sh
```

终端 3 发送请求或跑 benchmark：

```bash
bash scripts/autodl/run_pd_request.sh
bash scripts/autodl/benchmark_pd.sh
```

默认 PD 不启用 DSpark。要联合测试三个功能，P、D 两个终端都必须先执行：

```bash
export PD_ENABLE_DSPARK=1
```

两张高显存 GPU 可让 P/D 各使用一张卡。两张 24GB GPU 无法放下两套 27B，
应将 `PD_MODEL` 指向较小的 `LlamaForCausalLM` checkpoint，仅验证 PD；四张
24GB GPU 则让 P/D 分别设置 `PD_TP_SIZE=2` 和各自的两张可见卡。P/D 的模型、
DSpark 开关、TP、block size 和数据端口配置必须一致。PD 不支持跨 role 的
Prefix Cache；DSpark 当前只接受文本请求。

所有默认值都在 `common.sh` 和各入口脚本顶部通过环境变量覆盖，不需要修改
脚本本身。
