# Prefix Cache

## 当前实现

Prefix Cache 复用 token 前缀完全相同的完整 KV block。功能默认关闭，可通过 Python API 打开：

```python
from minivllm import LLM

llm = LLM(
    model="path/to/model",
    enable_prefix_caching=True,
)
```

使用 `EngineArgs` 的命令行入口也可以传入 `--enable-prefix-caching`。仓库中的
`benchmarks/benchmark_prefix_cache.py` 已显式打开该选项。

## 数据约定

- `Sequence.num_computed_tokens`：已经完成模型计算并写入 KV cache 的 token 数量。
- `Sequence.num_cached_blocks`：已经向 Prefix Cache 登记的完整 block 数量。
- `Sequence.block_hashes`：按 block 顺序组成的父子哈希链；后一个 block 的哈希包含前一个 block 的哈希。
- Prefix Cache 为每个登记的 GPU block 持有一个独立引用。
- `ref_count == 1` 表示 block 只被 Prefix Cache 持有，可以按 LRU 淘汰。

## 执行流程

1. 创建 Sequence 时计算所有完整逻辑 block 的哈希链。
2. Scheduler 准备 waiting 请求时查询最长连续命中，只把未命中后缀计入 batch token 预算。
3. BlockSpaceManager 为命中 block 增加 Sequence 引用，再为剩余逻辑 block 分配物理块。
4. Worker 只提交未计算的 prompt 后缀，并使用原序列的绝对 position 和物理 slot。
5. 无命中 prompt 继续走 xFormers；命中后缀通过 varlen paged attention 读取缓存前缀。
6. 模型执行后推进 `num_computed_tokens`，把新完成的完整 block 注册到 Prefix Cache。
7. 显存不足时只淘汰 cache-only 的 LRU block，不会淘汰正在被 Sequence 使用或本轮准备命中的 block。

## 为什么保留最后一段计算

Prefix Cache 只存 KV，不存用于采样的最终 logits。为了得到当前 prompt 最后一个 token
对应的 logits，最长命中限制为 `prompt_len - 1` 个 token：

- prompt 长度不是 block size 的整数倍时，只复用前面的完整 block。
- prompt 恰好 block 对齐时，最后一个完整 block 会重新计算。

这个约束让当前采样流程保持不变，也避免额外维护 logits cache。

## 当前边界

- 只复用完整 block，不复用部分 block。
- Prefix Cache 与同一个 Engine 实例绑定，不跨进程或重启持久化。
- 当前实现尚未加入 chunked prefill；一次 prefill 仍会计算该请求的全部未命中后缀。
- 缓存命中后缀依赖仓库内的 varlen CUDA attention kernel。
- 支持的 block size 仍为 `8`、`16`、`32`，head size 仍为 `64`、`80`、`96`、`128`。

## 后续 Chunked Prefill 接入点

Chunked prefill 不需要重做 Prefix Cache 生命周期。后续主要修改调度和输入规划：

1. Scheduler 按剩余 batch token 预算设置每条 prompt 的 `num_scheduled_tokens`，不再一次调度全部后缀。
2. Worker 去掉“本轮结束位置必须等于序列长度”的约束，只提交
   `[num_computed_tokens, num_computed_tokens + num_scheduled_tokens)`。
3. 本轮没有完成全部 prompt 时不采样，Sequence 保持 prefill 状态并在后续 step 继续调度。
4. 每个 chunk 完成后推进 `num_computed_tokens`，并只登记新完成的完整 block。

详细修改和每阶段验证结果见 `docs/prefix_cache_implementation_log.md`。
