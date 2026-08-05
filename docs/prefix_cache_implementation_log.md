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

状态：待开始

## 阶段三：缓存前缀后的后缀 Prefill

状态：待开始

## 阶段四：集成验证与文档收尾

状态：待开始
