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
| 5 | 精确随机 rejection sampling | 已完成 |
| 6 | confidence-scheduled 自适应验证 | 已完成 |
| 7 | TP2、显存核算、融合算子与真实模型入口 | 已完成 |
| 8 | Mooncake 风格 PD 分离组合 | 已完成 |

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

## 阶段 5：精确随机验证

本阶段让 DSpark 支持非零 temperature。Draft 和 Target 都先应用相同的
temperature、top-k、top-p、presence/frequency penalty；每个草稿 token 以
`min(1, p(token) / q(token))` 的概率接受。首个拒绝位置从归一化的
`max(p - q, 0)` 分布采样修正 token，全部接受后从 Target 最后一行采 bonus。
该算法保持输出分布与直接从 Target 采样一致。

验收结果：7 项变换、接受/拒绝、经验分布及 Qwen 分发测试全部通过；4000
次固定随机种子的首 token 经验分布在 0.025 绝对误差内匹配 Target。

## 阶段 6：置信度自适应验证

本阶段把 confidence head 的逐位置概率转成“到达该位置”的 survival
probability（前缀概率连乘），再把一个 batch 内所有候选位置按边际收益统一
排序。规划器用实测 Target token 数到延迟的 cost profile 枚举停止点，选择
期望输出 token/s 最大的前缀；所选位置天然满足每条序列的前缀闭包，不会
验证第 5 个草稿却跳过第 4 个。

调度器保存每条序列本轮实际验证宽度，只向 Worker 发送对应草稿前缀，并按
该宽度计算 token budget、物理槽位和最大可提交进度。没有 confidence 或未
开启 adaptive 时仍使用完整静态块。

验收结果：4 项 cost profile、全局排序、预算限制和调度集成测试通过；同时
重跑静态块与原有 MTP 调度测试，共 19 项全部通过。

## 阶段 7：真实运行链路与融合算子

本阶段把前六阶段的独立模块接入 Worker。目标 Qwen3.8 forward 通过可选
collector 采集发布配置指定的 `5/19/33/47/61` 层输出；草稿模型复用目标
embedding 和 TP LM head，并把目标特征投影到独立 Draft KV cache。Draft
cache 镜像调度器物理 block id，额外给每个并发请求保留只在草稿 forward
期间使用的 workspace block。Target 的 swap/copy 生命周期会同步应用到
Draft cache，但 workspace 不进入调度器所有权。

发布版 `RadixArk/Qwen3.8-27B-DSpark` 为 1.86B BF16、5 层、32 个 query
head/8 个 KV head、hidden size 5120、vocab 248320。其草稿 RoPE 使用 YaRN，
因此由 Transformers 5.7 的官方 YaRN 参数函数生成频率和 attention scale。
TP 下草稿层权重按既有 Column/Row Parallel 规则切分，最终 logits 聚合后再
裁掉词表 padding；FC、Markov 和 confidence 小头保持每卡复制。

贪心 Markov 路径新增 `dspark_ops.markov_argmax`：第一级 kernel 并行计算
词表 tile 内的 `base + embedding @ weight.T` 最大值，第二级归约得到最终
token，并在相同分数时保持最小 token id，行为与 `torch.argmax` 一致。
随机路径保留完整 FP32 proposal 分布，用于下一轮精确 rejection sampling；
显存规划会预留这些分布、草稿 KV block 和 workspace 的容量。

真实模型离线入口：

```python
from minivllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.8-27B-FP8",
    draft_model="RadixArk/Qwen3.8-27B-DSpark",
    tensor_parallel_size=2,
    num_speculative_tokens=7,
    speculative_adaptive=True,
)
outputs = llm.generate(
    ["请介绍一下投机解码。"],
    SamplingParams(temperature=0.0, max_tokens=128),
)
```

当前 DSpark 在线路径只接收纯文本请求；Qwen3.8 原有图像、视频和 MTP 路径
保持可用，但含多模态位置的请求不能与 Qwen3 风格的 DSpark RoPE 混用。

验收结果：CUDA 12.8 下全量扩展编译成功；融合 Markov 算子的 FP16、BF16、
FP32 与 tie-break 数值测试、完整 tiny DSpark context/paged proposal 测试、
以及 17 项 GQA/GDN/Qwen hybrid CUDA 回归全部通过。

## 阶段 8：Mooncake 风格 PD 分离组合

本阶段把项目自有的 P/D 控制面和 cache 数据面组合到 DSpark 运行链路中，
不导入 Mooncake。P worker 完成 prompt prefill 时，除了 Target 的 full-attention
K/V 与 GDN recurrent state，还会把收集到的目标层特征投影到 Draft K/V。
传输 layout 新增独立的 `draft_key`、`draft_value` 区域，并沿用 Target 的物理
block 映射，因此 D 预留自己的 block 后可以一次接收三类持久状态。

P 不在 handoff 前生成 speculative block。随机验证所需的逐 token 完整 q
分布规模为 `draft_width * vocab_size`，把它放进控制面会破坏“小元数据、
大 cache 走数据面”的边界。D 接管后先正常消费 P 已采样的首 token，同时把
这一位置的目标特征补入收到的 Draft K/V；从该步输出开始再生成 DSpark
proposal。这样只增加一次 decode warmup，不需要扩展 handoff 协议，也保持
rejection sampling 的精确分布。

资源生命周期仍由 PD coordinator 管理：P 的 Target/Draft blocks 在所有 TP
rank 返回 ACK 前保持占用，D 的 blocks 和 GDN state slot 在传输前预留、传输
成功后才进入 runnable queue。Draft workspace 只用于本地 proposal forward，
虽然和持久 cache 一起注册，但 planner 只选择调度器拥有的 Target block id，
不会搬运 workspace 内容。

启动 P/D 时，两侧使用相同的 `--draft-model`、`--num-speculative-tokens` 和
自适应验证参数。模型本体采用 TP=2 时，真正的 P/D 分离需要两套完整副本，
即 P 两卡加 D 两卡；只有两张卡时可运行统一模式的 TP=2 DSpark，不能同时
容纳两个 TP=2 role。

验收范围：Draft K/V 跨不同物理 block id 搬运、P/D proposal 所有权、
PD handoff 生命周期、TP layout 配对、DSpark 静态/随机/自适应验证及原有
Qwen3.8 CUDA 路径回归。

验收结果：WSL2 `mini-vllm` 环境中按模块隔离运行全仓测试，212 项全部
通过；其中 21 项为显式启用 opt-in 开关后的 CUDA 数值测试。本阶段未修改
C++/CUDA 源码，复用阶段 7 的已编译扩展完成验证。
