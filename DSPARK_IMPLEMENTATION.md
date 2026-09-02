# DSpark 参考实现记录

## 分支与目标

- 学习分支：`codex/dspark-learning`，从 `codex/qwen38-reference` 创建。
- 参考分支：`codex/dspark-reference`，按独立可验证阶段完成完整实现。
- 目标 checkpoint：`RadixArk/Qwen3.8-27B-DSpark`。
- 目标模型：`Qwen/Qwen3.8-27B-FP8`。

## 阶段状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | DSpark 配置、Markov 与 confidence 输出头 | 已完成 |
| 2 | DFlash 草稿模型、权重加载与目标层特征 | 已完成 |
| 3 | Draft KV cache 与块内双向 attention | 已完成 |
| 4 | 静态贪心块验证、调度与 GDN 状态事务 | 已完成 |
| 5 | 精确随机 rejection sampling | 未开始 |
| 6 | confidence-scheduled 自适应验证 | 未开始 |
| 7 | TP2、显存核算、融合算子与真实模型入口 | 未开始 |
| 8 | Mooncake 风格 PD 分离组合 | 未开始 |

## 阶段 1：配置与输出头

本阶段先把 checkpoint 字段归一化，并实现不依赖 Worker、Cache 和 CUDA
的低秩 Markov 头与置信度头。Markov 头保留每一步修正后的完整 logits，
供后续精确 rejection sampling 使用；置信度头支持逐位置 STS 温度校准。

验收范围：配置校验、teacher-forcing 数值、逐步采样依赖、confidence 特征
拼接与 STS 校准。

验收结果：WSL2 `mini-vllm` 环境运行 `tests.test_dspark_heads`，6 项测试
全部通过。

## 阶段 2：草稿骨干与目标层特征

本阶段实现标准 Qwen3 GQA 草稿层，不能复用目标 Qwen3.8 的门控 Full
Attention。目标模型通过默认关闭的 collector 回调暴露选定层输出；草稿
模型把这些输出拼接投影为共享 context hidden，再由每个草稿层分别生成
自己的 K/V。密集 reference 路径明确实现“历史上下文 + 块内双向注意力”，
作为下一阶段 paged CUDA 路径的数值基准。

验收结果：tiny DSpark 完成离线整块 proposal，目标层收集顺序、块内双向
可见性和 packed QKV 权重映射均通过测试；阶段 1、2 合计 11 项测试通过。

## 阶段 3：Draft KV 与块内双向注意力

Draft KV cache 镜像 Target 的物理 block id，但保存 5 个草稿层各自的 K/V。
现有 varlen paged-attention kernel 增加 `query_is_causal` 参数：Target cached
prefill 传 `true`，DFlash 当前块传 `false`。后者让块内所有 query 看见完整
当前块，同时仍用 `context_len` 屏蔽物理 block 末尾未使用的槽位。

验收结果：完成 Draft cache 物理块映射、逐层 context K/V 物化和非因果
块内 paged attention；WSL2 `mini-vllm` 环境编译 CUDA 扩展后，DSpark、
GQA 与 Qwen gated attention 共 11 项 CUDA/接口测试全部通过。

## 阶段 4：静态块验证与状态事务

本阶段把原有 MTP-1 扩展为统一的线性草稿块协议。调度器为“已知锚点 +
草稿块”精确预留物理槽位；Target 的每个 logits 行依次验证一个草稿，首个
不匹配位置输出 Target 修正 token，全部接受时再输出 bonus token。MTP 仍是
块长为 1 的同一条路径。

Qwen3.8 的 GDN state 不能像 paged KV 一样靠缩短逻辑长度隐藏错误写入。
Worker 因此在验证前保存快照：若只接受草稿前缀，则恢复原状态，再按顺序
重放锚点和已接受草稿；全接受时直接保留一次验证得到的状态。

验收结果：新增 6 项 DSpark 块验证、变长调度和重放计划测试，与原有 9 项
prefix/MTP 调度测试全部通过；原生 Qwen MTP CUDA 端到端测试继续通过。
