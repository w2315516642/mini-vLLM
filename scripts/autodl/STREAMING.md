# 流式输出

## 在推理链路中的位置

本项目保留同步的 `LLMEngine.step()`，没有新增模型执行线程，也没有把
`yield` 放进 CUDA 算子。一次 step 完成调度、模型计算、采样、反分词后，
返回该请求当前的累计 `RequestOutput`。新链路为：

```text
LLM.generate_stream / chat_stream
  -> 现有请求预处理与 add_request
  -> LLMEngine.step（普通 decode、MTP、DSpark 共用）
  -> 累计 RequestOutput 快照
  -> OutputProcessor（记录每个 request_id / completion.index 的输出位置）
  -> Python generator -> 调用者

PDClient.generate_stream
  -> P.step -> 首 token -> 同一个 OutputProcessor -> 调用者
  -> 原有 KV / recurrent state 交接
  -> D.step -> 累计输出 -> 同一个 OutputProcessor -> 调用者
```

设计参考 vLLM 的 [OutputProcessor](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/output_processor.py)
与 [AsyncLLM.generate](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/async_llm.py)：
输出格式转换与模型计算解耦，由生成器逐次交付结果，并在取消时释放请求。
这里适配项目现有同步入口，没有搬入 vLLM 的后台 EngineCore、每请求异步队列
或 OpenAI HTTP/SSE 服务。它是实际的逐步生成，不是完整生成后再拆字符串。

## Python 用法

```python
from contextlib import closing
from minivllm import LLM, SamplingParams, RequestOutputKind

llm = LLM(model="/path/to/model", tensor_parallel_size=1)
params = SamplingParams(temperature=0.0, max_tokens=128)

with closing(llm.generate_stream("介绍一下 KV Cache", params)) as stream:
    for update in stream:
        for completion in update.outputs:
            print(completion.text, end="", flush=True)
print()
```

`generate_stream` 默认使用 `RequestOutputKind.DELTA`：`text`、`token_ids`、
`logprobs` 只包含这次新增的部分，`cumulative_logprob` 始终为累计值。
同一个请求的多条 completion 通过 `index` 区分，批量请求通过 `request_id`
区分；不能把不同请求的增量直接拼在一起。

另外两种输出模式通过 `output_kind=` 指定：

- `CUMULATIVE`：每次返回当前累计内容，适合替换 UI 中的整段文本。
- `FINAL_ONLY`：仅返回最终快照；`LLM.generate()` 使用此模式，仍返回列表，
  并按输入提示词顺序排列。

`update.is_finished()` 表示整个请求结束，`completion.finish_reason` 表示
该条 completion 结束。最后一次更新可能没有新文本或 token，但仍携带结束
标记，调用者应处理它。提前 `break` 时必须关闭生成器；上面的 `closing`
会自动调用 `close()`，让尚未完成的请求释放 KV 块及 recurrent state。

图像、视频仍使用原有 processor 和模型能力，通过 `llm.chat_stream(messages,
params)` 调用，参数 `enable_thinking` 与 `chat()` 一致。

PD 用法相同：

```python
from contextlib import closing
from minivllm.engine.pd_rpc import PDClient
from minivllm import SamplingParams

client = PDClient("127.0.0.1:15000", "127.0.0.1:15100", b"mini-vllm")
try:
    with closing(client.generate_stream(
        "介绍一下 PD 分离", SamplingParams(temperature=0.0, max_tokens=64)
    )) as stream:
        for update in stream:
            for completion in update.outputs:
                print(completion.text, end="", flush=True)
finally:
    client.close()
```

P 采样的首 token 会在传输前交付。D 的累计结果包含这个 token，但输出
处理器的偏移量不会在交接时重置，因此不会重复发送。中途关闭时，交接前
取消 P 的请求，交接后取消 D 的请求；原来的非流式与带 metrics 接口使用
同一个 PD 执行循环。

## 边界与修正

- 停止字符串不应出现在输出中，因此参考 vLLM 的
  [detokenizer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/detokenizer.py)
  暂存最长停止词长度减一的尾部字符，结束时再输出允许保留的部分。字节级
  tokenizer 产生的末尾不完整 UTF-8 也会暂存，所以收到 token 时文本可能为空。
- 修正原有反分词调用：函数返回累计文本，引擎应赋值，而非重复追加。
  特殊 token 保留在 token 进度中，但不参与文本渲染。
- DSpark/MTP 只输出 Target 已接受的结果，一次更新可以有多个 token。
  停止字符串可以出现在这批文本中间，不能只检查文本末尾。
- 输出对象持有 token/logprob 列表的快照，不与下一步仍会修改的序列共享列表。
- 非最终流式模式要求 `best_of == n` 且关闭 beam search，避免中途替换候选
  导致已经输出的内容失效。`FINAL_ONLY` 不增加这项限制。
- 同一个 `LLM` 或 `PDClient` 同时只允许一个生成调用。消费速度会反过来限制
  step 的推进，不额外堆积无界队列。PD 的一对服务也仍只应由一个客户端驱动；
  多客户端在线并发需要统一的调度循环和按请求分发结果，不属于本次改动。
- 此次未新增 HTTP/SSE 接口、远程断线自动回收或多 P 路由。Python 生成器的
  `close()` 是本次明确支持的取消路径；强杀客户端进程不等价于正常关闭。

## 测试

在项目 Conda 环境运行，无需重新编译算子：

```bash
PYTHONPATH=tests python -m unittest -v \
  tests.test_streaming tests.test_pd_streaming tests.test_qwen_multimodal
```

测试覆盖增量与累计模式、批量请求、结束标记、停止词跨 token、分裂 UTF-8、
多 token 已验证输出、旧接口兼容、取消清理，以及真实本机 RPC 连接上的
P 首 token / D 续写 / 异常传播。RPC 测试以可控引擎替身隔离模型计算，
不把它当成 27B 模型的吞吐或显存测试。完整回归入口为
`bash scripts/autodl/test_features.sh`。

### 本次验证记录

2026-09-03，在本机 WSL Ubuntu 的 `mini-vllm` Conda 环境完成：

- 新增流式测试 30 项全部通过，包含两个命令行入口的输出与计数检查。
- 原有完整功能回归 205 项：204 项通过，双卡 NCCL 用例因本机只有一张 GPU
  跳过。Qwen 混合模型、MTP、图像视频、DSpark CUDA 用例实际执行通过。
- 额外运行 Prefix Cache 块管理、调度器及 Worker 测试 17 项，全部通过。
- 本次没有改动或重新编译 CUDA 扩展，也没有进行 27B 真模型端到端吞吐测试。

修改集中在输出快照、共享输出处理器、普通/PD 生成入口和命令行脚本；
现有模型计算、缓存分配策略与数据传输协议保持不变。
