# Prefix Cache 实现记录

本文记录 Prefix Cache 的分阶段修改过程。每个阶段单独提交，便于按提交阅读代码和回退问题。

## 约定

- `Sequence.num_computed_tokens` 表示已经写入 KV cache 的 token 数量。
- `Sequence.num_cached_blocks` 表示已经完成哈希登记的完整 block 数量。
- Prefix Cache 为登记的 GPU block 独立持有一个引用。
- `ref_count == 1` 的缓存 block 只被 Prefix Cache 持有，可以由 LRU 淘汰。
- 只复用完整 block，并保留最后一个已知 token 用于计算 logits。

## 阶段一：哈希与缓存块生命周期

状态：已完成

目标：先保证哈希链、引用计数、缓存注册和释放行为正确，不修改调度和模型执行流程。

完成内容：

- 修复 Sequence 类型注解引起的循环导入问题。
- 修复父 block 哈希与 token 参数顺序。
- 修复最长前缀匹配调用、LRU 更新和设备 allocator 使用错误。
- Prefix Cache 对 block 持有独立引用，并在显存不足时淘汰 cache-only 的 LRU block。
- 缓存注册以 `num_computed_tokens` 为准，避免登记尚未计算的逻辑 block。
- 补充哈希、命中、释放、引用计数和 LRU 淘汰的 CPU 单元测试。

验证结果：

- `python -m unittest discover -s tests -v`：4 项测试通过。
- `python -m py_compile ...`：本阶段涉及的 Python 文件语法检查通过。
- `git diff --check`：通过。

实现说明：

- 保留原有 `BlockAllocator` 行为：新 block 分配时自带第一个 Sequence 引用。
- 命中 block 在原引用基础上增加 Sequence 引用；新 block 只补同组其余 Sequence 的引用。
- 缓存登记时额外增加一个 cache-owned 引用，请求释放后 block 仍可复用。
- 分配 GPU block 时仅在 free list 为空后淘汰 `ref_count == 1` 的最老缓存 block。

## 阶段二：命中感知调度

状态：已完成

完成内容：

- 新增 `enable_prefix_caching` 配置和 CLI 参数，默认关闭。
- 关闭缓存时不创建 block hash，也不登记缓存引用，保持原执行路径。
- `SchedulerOutputs` 记录每条 Sequence 本轮实际执行的 token 数量。
- prompt 的 batch token 预算只计算前缀命中后的剩余 token。
- `SequenceGroupMetadata` 携带执行前进度和本轮调度量，供 Worker 切分输入。
- 模型执行完成后先推进 `num_computed_tokens`，再追加采样 token。
- recompute preemption 正确重置进度，并把 `SequenceGroup` 放回 waiting 队列。

验证结果：

- `python -m unittest discover -s tests -v`：新增调度测试后共 8 项测试通过。
- 覆盖缓存关闭、命中后预算、执行进度更新和 recompute 状态重置。
- 本阶段 Python 文件通过 `py_compile` 和 `git diff --check`。

## 阶段三：缓存前缀后的后缀 Prefill

状态：已完成

完成内容：

- Worker 将输入稳定排列为“无命中 prompt、命中后的 prompt 后缀、decode token”，并只为命中请求提交未计算后缀。
- prompt 后缀沿用原序列的绝对 position 和物理 block slot，不重复计算已缓存 token。
- 无命中 prompt 保留 xFormers 路径；命中后缀通过 varlen paged attention 读取已有 KV。
- 在执行后缀注意力前先写入本轮 K/V，使每个 query token 能看到缓存前缀和它之前的后缀 token。
- 修复 varlen CUDA kernel 的 packed query 偏移、因果掩码和 block size 32 模板分发。
- 修复 decode block table 最大宽度读取错误。
- 补充 Worker 输入规划测试，覆盖 token、position、slot mapping、block table 和采样分组顺序。

验证结果：

- 所有 Python 命令均通过 `conda run -n mini-vllm` 执行。
- `python -m unittest discover -s tests -v`：共 9 项测试通过。
- 本阶段 Python 文件通过 `py_compile`，代码通过 `git diff --check`。
- CUDA 扩展数值验证未完成：Windows 下首次编译 attention 扩展耗时较长，按要求中止编译并继续后续收尾；本阶段不把输入规划测试等同于 CUDA kernel 数值验证。

## 阶段四：集成验证与文档收尾

状态：已完成

完成内容：

- 修复显存满载时的容量高估：本轮准备复用的 cache-only 命中块不再计入可淘汰容量。
- 分配未命中后缀物理块前先增加命中块的 Sequence 引用，避免命中块在同次分配中被淘汰和复用。
- 增加满载缓存回归测试，覆盖“容量不足时拒绝”和“容量恰好时只淘汰非命中块”。
- Worker 输入规划测试扩展为多个不同长度的缓存后缀，验证 packed query 的累计长度和 block table。
- `benchmark_prefix_cache.py` 显式打开 `enable_prefix_caching`。
- 重写 `hash_prefix_caching.md`，记录开关、数据约定、执行流程、当前边界和 Chunked Prefill 接入点。

最终验证：

- 所有 Python 命令均通过 `conda run -n mini-vllm` 执行。
- `python -m unittest discover -s tests -v`：共 10 项测试通过。
- `python -m compileall -q minivllm tests benchmarks/benchmark_prefix_cache.py`：通过。
- `git diff --check`：通过。
- 当前 conda 环境未安装 `ray` 和 `xformers`，无法启动完整模型推理链路。
- attention CUDA 扩展首次编译耗时较长，已按要求中止；因此本轮没有完成 varlen kernel 的编译和数值对照测试，该限制不会隐藏为“已验证”。
