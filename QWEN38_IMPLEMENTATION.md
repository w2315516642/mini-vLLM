# Qwen3.8 学习式适配记录

## 协作规则

- 每个阶段由 Codex 完成外围脚手架、测试和接口说明。
- 每阶段开始时，在 `docs/qwen38-learning/` 提供一份不纳入 Git 的背景与实现讲义。
- 每份讲义先说明本阶段模块在端到端推理链路中的位置、调用时机、上下游数据和对后续模块的影响，再讲模块内部实现。
- 学习者负责一个约 100 到 300 行的核心模块，部分阶段的核心模块是 CUDA 算子。
- 作业验收前不提交阶段完成 commit。
- Codex 默认只提供概念级提示，不直接覆盖作业实现。
- 每阶段通过测试和原理问答后，以中文 commit 收口。
- CUDA 验证使用 `/root/miniconda3/envs/vllm` 环境。

## 阶段状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 配置归一化与结构校验 | 已完成 |
| 2 | 模型注册与嵌套配置接入 | 已完成 |
| 3 | GQA 与独立 KV 头 | 已完成 |
| 4 | Qwen Gated Full Attention（含 Q/K RMSNorm 与门控融合算子） | 已完成 |
| 5 | Gated DeltaNet 参考实现 | 已完成 |
| 6 | Gated DeltaNet Kernel（含 Decode 状态更新与 Prefill 分块扫描算子） | 已完成 |
| 7 | Hybrid Cache | 已完成 |
| 8 | 真实小模型端到端推理 | 待验收 |
| 9 | FP8 权重 | 未开始 |
| 10 | TP=2 与 Qwen3.8-27B | 未开始 |
| 11 | 连续批处理 | 未开始 |
| 12 | Hybrid Prefix Cache | 未开始 |
| 13 | Chunked Prefill | 未开始 |
| 14 | MTP | 未开始 |
| 15 | 视觉与视频 | 未开始 |
| 16 | Thinking 与最终验收 | 未开始 |

## 阶段 1：配置归一化与结构校验

### 为什么先做这一层

Llama 的语言模型字段直接位于根配置中；Qwen3.8 的根配置同时描述视觉和语言模型，语言模型字段位于 `text_config`。此外，Qwen3.8 的 `head_dim=256`，不能使用 `hidden_size / num_attention_heads` 推导。后续 Cache、GQA、DeltaNet 和 TP 都必须依赖同一份可靠的结构描述。

### Codex 负责

- 定义 `ModelArchitecture` 的稳定接口。
- 提供平铺 Llama、嵌套 Qwen、非法结构和并行切分测试。
- 保留现有 `ModelConfig` 的行为，避免提前完成集成作业。

### 学习者负责

1. 完成 `minivllm/configs/model_architecture.py` 中全部 TODO。
2. 修改 `minivllm/configs/config.py`：
   - 构造 `self.architecture`。
   - 使用归一化后的 `text_config` 决定 dtype。
   - 将结构访问和并行校验委托给 `ModelArchitecture`。
   - 新增 `get_num_kv_heads`。
3. 修改 `minivllm/configs/__init__.py`，导出 `ModelArchitecture`。

### 约束

- 不通过模型名称判断是否为 Qwen。
- 不修改输入的 Hugging Face config。
- 缺少字段或结构非法时抛出包含字段名的 `ValueError`。
- `layer_types` 缺失时默认所有层为 `full_attention`。
- `num_key_value_heads` 缺失时退化为普通 MHA。
- 显式 `head_dim` 优先；缺失时才允许整除推导。
- 不在这一阶段修改 CacheEngine、Attention 或模型注册表。

### 验收命令

```bash
/root/miniconda3/envs/vllm/bin/python -m unittest \
  discover -s tests -p 'test_model_architecture.py' -v
```

回归测试：

```bash
/root/miniconda3/envs/vllm/bin/python -m unittest \
  discover -s tests -p 'test_prefix_cache_*.py' -v
```

### 原理验收

- 为什么 Qwen3.8 的 head size 不能通过 hidden size 除以 Q 头数得到？
- 为什么 TP 切分时 Q 头数和 KV 头数都要校验？
- 为什么 Hybrid Cache 需要稳定的逐层 `layer_types`？

### 完成记录

- 完成时间：2026-08-25。
- 学习者完成：平铺/嵌套配置归一化、显式 `head_dim`、GQA 头数、逐层注意力类型和并行切分校验，并将 `ModelConfig` 接入归一化结构。
- Codex 收尾：统一可诊断的 `ValueError`，补齐显式 `None` 回退测试，并修复 Prefix Cache 测试桩与配置测试同时收集时的 patch 隔离问题。
- 验证环境：WSL2 Ubuntu，`/home/yue/miniconda3/envs/mini-vllm`，Python 3.10.20、PyTorch 2.11.0+cu128、Transformers 5.7.0，CUDA 12.8 可用。
- 验证结果：阶段测试 19/19 通过，Prefix Cache 回归 10/10 通过，完整测试 29/29 通过，`git diff --check` 无空白错误。
- 已知限制：本阶段只验证结构归一化与现有调用链，尚未加载真实 Qwen3.8 权重；模型注册、GQA 执行路径和混合注意力将在后续既定阶段完成。
- 下一阶段：阶段 2“模型注册与嵌套配置接入”，保持未开始，等待讲义和作业脚手架准备。

## 阶段 2：模型注册与嵌套配置接入

### 为什么需要独立注册表

当前 `model_loader.py` 使用硬编码类字典，并把 Hugging Face 根配置直接传给模型构造函数。Qwen3.8-27B 的官方 architecture 是 `Qwen3_5ForConditionalGeneration`，但语言维度位于 `text_config`。本阶段将“根据根 architecture 选类”和“使用文本配置构造语言骨干”分离，并通过惰性注册避免查询配置时提前导入所有模型及 CUDA 依赖。

### Codex 负责

- 提供 `ModelRegistration`、`ModelRegistry` 的稳定类型和公开接口。
- 将 `model_loader.py` 接到归一化 architecture 与 `text_config`。
- 登记 Llama 和 Qwen3.8 的官方 architecture 字符串目标。
- 提供注册、解析、惰性导入、错误诊断和 loader 集成测试。
- 在忽略的 `docs/qwen38-learning/stage-02-model-registry.md` 提供阶段讲义。

### 学习者负责

完成 `minivllm/model_executor/models/registry.py` 中全部 `TODO(student)`：

1. 校验 architecture 和 eager/lazy target。
2. 实现重复注册与显式覆盖。
3. 按 Hugging Face architecture 顺序解析首个支持项。
4. 使用 `importlib` 惰性加载 `module:ClassName`。
5. 验证解析结果是 `torch.nn.Module` 子类。
6. 为不支持、导入失败和类型错误提供可诊断异常。

预计自然实现量约 100 到 200 行，不需要为了行数增加抽象。

### 约束

- 不根据模型仓库名称或字符串前缀猜模型族。
- 不把未知模型静默回退成 Llama。
- 不在注册或查询支持列表时导入 lazy target。
- 不改变 Hugging Face `architectures` 的声明顺序。
- 不在本阶段实现 Qwen 模型主体、Attention、DeltaNet、Cache 或权重映射。
- Qwen lazy target 在后续模型阶段落地前允许尚不可实例化，但 architecture 必须准确登记。

### 验收命令

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_model_registry.py' -v
```

回归测试：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_model_architecture.py' -v

/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_prefix_cache*.py' -v
```

### 预提交基线

- 新测试应能正常导入。
- loader 的 `text_config` 接线测试应通过。
- registry 核心测试应只在 `TODO(student)` 处失败。
- 阶段 1 与 Prefix Cache 回归应继续通过。
- 2026-08-25 基线：阶段 2 测试 2/10 通过，其余 8 项停在保留 TODO；完整测试 31/39 通过，无非预期失败。

### 原理验收

- 为什么 Qwen3.8 使用 `Qwen3_5ForConditionalGeneration` 作为注册名称？
- 为什么根配置用于选类，而 `text_config` 用于构造语言骨干？
- lazy import 为什么能降低多进程 GPU 初始化风险？
- 为什么 architecture 顺序不能使用集合重排？

### 完成记录

- 完成时间：2026-08-25。
- 学习者完成：architecture 与 eager/lazy target 校验、重复注册和显式覆盖、按声明顺序解析首个支持项、`module:ClassName` 惰性加载，以及加载结果的 `nn.Module` 子类校验。
- Codex 收尾：补齐包含 architecture、target 和支持列表的诊断信息，收窄异常捕获边界并保留 lazy 加载的原始 cause，增加通过公开接口注册 lazy target 时不提前导入的回归测试。
- 验证环境：WSL2 Ubuntu，`/home/yue/miniconda3/envs/mini-vllm`，Python 3.10.20、PyTorch 2.11.0+cu128、Transformers 5.7.0，CUDA 12.8 可用。
- 验证结果：阶段测试 11/11 通过，阶段 1 回归 19/19 通过，Prefix Cache 回归 10/10 通过，完整测试 40/40 通过；`git diff --check` 无空白错误，仅有 Windows Git 的 LF/CRLF 转换提示。
- 已知限制：本阶段只完成模型路由和嵌套文本配置接线；Qwen 的 lazy target 尚未实现，真实权重加载、GQA、混合注意力和 Cache 留在后续既定阶段。
- 下一阶段：阶段 3“GQA 与独立 KV 头”保持未开始，等待讲义和作业脚手架准备。

## 阶段 3：GQA 与独立 KV 头

### 为什么这一层位于 Qwen 模型主体之前

阶段 1 已归一化 Query/KV 头数，阶段 2 已能按 architecture 选择模型类，但当前 attention 热路径仍假设 Q/K/V 等宽：Llama 使用 `qkv.chunk(3)`，PagedAttention 用同一个 `num_heads` reshape Q/K/V，RoPE 与 paged-attention CUDA kernel 也共用 Q 头 stride。阶段 3 先在现有 Llama 路径打通紧凑 GQA，使阶段 4 的 Qwen Gated Full Attention 可以复用可靠的 Q/K/V、RoPE 和 KV Cache shape contract。

### Codex 负责

- 将 Llama 构造链接到独立 `num_key_value_heads` 和显式 `head_dim`。
- 让 packed QKV 投影和 output projection 使用正确的全局宽度。
- 将 CacheEngine 的 head 维度与块字节计算接到本地 KV 头数。
- 保留 MHA 兼容默认值并提供 GQA 比例校验。
- 提供 shape、TP 权重切片、Cache 显存、CUDA 数值 reference 和集成测试。
- 在忽略的 `docs/qwen38-learning/stage-03-gqa-independent-kv-heads.md` 提供阶段讲义。

### 学习者负责

完成以下生产路径中的全部 `TODO(student)`：

1. `minivllm/model_executor/models/llama.py`
   - `_split_qkv`：按 `[q_size, kv_size, kv_size]` 切分 rank-local packed projection。
   - `_load_qkv_weight`：分别计算全局 checkpoint 的 TP 源切片和本地 packed 目标 offset。
2. `minivllm/model_executor/layers/attention.py`
   - `_reshape_qkv`：恢复 `[T, Hq, D]` 与紧凑 `[T, Hkv, D]`。
   - `_grouped_prefill_inputs`：生成 xFormers 五维 grouped layout，并用 broadcast view 共享 K/V。
3. `csrc/pos_encoding_kernels.cu`
   - 让 Q/K 使用独立头数和 token stride 执行 GPT-NeoX RoPE。
4. `csrc/attention/attention_kernels.cu`
   - 在 decode 与 cached-prefill kernel 中实现 `query_head -> kv_head` 映射。
   - 区分 Q/output 的 `Hq` stride 与 KV Cache 的 `Hkv` stride。

预计自然实现量约 150 到 250 行，包含必要的 shape 与地址计算注释，不需要增加新框架抽象。

### 约束

- KV Cache 必须保持紧凑，不允许把整个 Cache 实体复制到 Query 头数。
- fresh prefill 使用项目当前 xFormers CUTLASS `FwOp` 的五维 GQA layout。
- 不改变 Query head 的输出宽度，output projection 仍消费全部 Query 头。
- 不改变 Hugging Face checkpoint 中独立 `q_proj`、`k_proj`、`v_proj` 的语义。
- 保持 MHA (`Hq == Hkv`) 行为兼容。
- 不在本阶段实现 Qwen 模型类、Q/K RMSNorm、attention gate、M-RoPE、head size 256 specialization、DeltaNet 或 Hybrid Cache。
- 当前仍要求 Query 头数和 KV 头数都能被 TP size 整除，不实现 KV head replication。

### 验收命令

普通 shape、权重和集成测试：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_gqa.py' -v
```

完成 CUDA 源码后先重新编译扩展：

```bash
cd /mnt/e/Projects/mini-vllm
/home/yue/miniconda3/envs/mini-vllm/bin/python -m pip install \
  -v -e . --no-build-isolation
```

然后显式开启 CUDA 数值测试：

```bash
MINIVLLM_RUN_CUDA_GQA_TESTS=1 \
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_gqa_cuda.py' -v
```

回归与完整测试：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_model_architecture.py' -v

/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_model_registry.py' -v

/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_prefix_cache*.py' -v

/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -v
```

### 预提交基线

- 2026-08-25：阶段 3 测试共 11 项，外围接线 4 项通过，4 项精确停在 Python 保留 TODO，3 项 CUDA 数值测试在未重编译前按设计跳过。
- 完整测试共 51 项：44 项通过、4 项预期 TODO 错误、3 项 CUDA 跳过；阶段 1、阶段 2 与 Prefix Cache 无额外回归。
- CUDA 源码中的 TODO 不在预提交基线编译；学习者完成后必须重新编译，并开启 `MINIVLLM_RUN_CUDA_GQA_TESTS=1` 验证真实扩展。

### 原理验收

- 为什么 GQA 的 Query 输出宽度不变，但 KV Cache 可以缩小？
- 为什么 xFormers prefill 可以 broadcast K/V，decode 却不能每步展开整个 Cache？
- 为什么 CUDA grid 使用 Query 头数，而 cache block stride 使用 KV 头数？
- 为什么 TP 权重加载必须分别计算 checkpoint 源切片和本地 packed 目标 offset？
- 为什么 RoPE 数学上允许 Q/K 头数不同，而当前 kernel 实现却不允许？

### 完成记录

- 完成时间：2026-08-26。
- 学习者完成：rank-local packed QKV 切分与 TP 权重装载、独立 Q/KV head reshape、xFormers 五维 GQA prefill、独立 Q/K stride 的 RoPE kernel，以及 decode 和 cached-prefill 的 Query head 到紧凑 KV head 映射。
- Codex 收尾：接入独立 KV 头配置和紧凑 Cache 分配，补齐 GQA shape 与 launcher 运行时校验、fresh-prefill 输出缓冲区契约，并修正 CUDA 数值测试的 `slot_mapping` dtype。
- 验证环境：WSL2 Ubuntu，`/home/yue/miniconda3/envs/mini-vllm`，Python 3.10.20、PyTorch 2.11.0+cu128、Transformers 5.7.0，CUDA 12.8 可用。
- 验证结果：CUDA 扩展编译成功；RoPE、decode、cached-prefill 三项 CUDA 数值测试分别 1/1 通过；最终完整测试 51 项通过，其中 48 项执行通过、3 项 CUDA 用例按开关跳过；`git diff --check` 无空白错误，仅有 Windows Git 的 LF/CRLF 转换提示。
- 验证说明：cached-prefill 的最终 `assert` 到 `TORCH_CHECK` 机械替换按用户要求未再次复编译；替换前同一计算路径已成功编译并通过数值测试。
- 已知限制：本阶段不包含 Qwen 模型类、Q/K RMSNorm、attention gate、M-RoPE、head size 256 specialization、DeltaNet、Hybrid Cache 或 KV head replication。
- 下一阶段：阶段 4“Qwen Gated Full Attention”保持未开始，等待讲义和作业脚手架准备。

## 阶段 4：Qwen Gated Full Attention

### 在推理链路中的位置

阶段 1 已提供嵌套文本配置和逐层 `layer_types`，阶段 2 已登记 Qwen3.8 checkpoint 使用的 `Qwen3_5ForConditionalGeneration`，阶段 3 已打通紧凑 GQA、RoPE 与 KV Cache。本阶段实现 Qwen hybrid decoder 中 `full_attention` 分支的 token mixer：它接收 decoder block 的归一化 hidden states，生成 Q/Gate/K/V，调用已有 PagedAttention，使用 gate 调制 attention output，再经过 output projection 回到残差流。

完整 Qwen 模型仍不会在本阶段提前落地。`linear_attention` 分支、Gated DeltaNet recurrent state 和 Hybrid Cache 分别属于后续既定阶段；在这些语义完整前，不使用 Llama Attention 或空操作伪装成可执行的混合模型。

### Codex 负责

- 新增 `Qwen3_5Attention` 的稳定构造、TP 尺寸、partial RoPE 与 PagedAttention 接线。
- 新增 Qwen 零中心 RMSNorm 和 sigmoid output gate 的公开接口。
- 沿用现有 `activation_ops` 扩展，提供 `sigmoid_and_mul` C++ binding 与 CUDA 作业边界。
- 为现有 decode/cached-prefill attention launcher 开启 Qwen 所需的 `head_dim=256` specialization。
- 提供 per-head Q/Gate layout、Q/K norm、TP 权重、调用顺序、CUDA 数值和 head size 256 测试。
- 在忽略的 `docs/qwen38-learning/stage-04-qwen-gated-full-attention.md` 提供包含完整推理上下文和源码链接的阶段讲义。

### 学习者负责

完成全部 `TODO(student, stage 4)`：

1. `minivllm/model_executor/layers/layer_norm.py`
   - 实现 `Qwen3_5RMSNorm.forward` 的 FP32 归一化和 `(1 + weight)` 语义。
2. `minivllm/model_executor/models/qwen3_5.py`
   - 按每个 Query head 的 `[Q | gate]` 布局实现 `_split_q_gate_kv`。
   - 实现独立 checkpoint Q/K/V 到 rank-local packed 参数的 TP 权重装载。
   - 完成 Q/K per-head norm 的 `_project_qkv` 和 full-attention `forward`。
3. `csrc/activation_kernels.cu`
   - 实现 `attention_output * sigmoid(gate)` CUDA kernel、参数校验、dtype dispatch 和 current-stream launcher。

预计自然实现量约 150 到 250 行。shape 推导、公式、实现顺序、CUDA 线程划分与参考源码均见阶段讲义。

### 约束

- `q_proj` layout 是逐 head `[q_head | gate_head]`，不能对扁平 segment 直接 `chunk(2)`。
- Q/K RMSNorm 只沿 `head_dim` 归一化，位于 RoPE 之前；V 不做该归一化。
- gate 不参与 QK 点积、不写 KV Cache，门控位于 PagedAttention 之后、`o_proj` 之前。
- Qwen norm 使用零中心 checkpoint 参数，effective weight 是 `1 + weight`。
- KV Cache 保持紧凑 GQA layout，不展开到 Query head 数。
- 本阶段只验证文本位置下的 partial RoPE 数据通路，不实现视觉三轴 M-RoPE。
- 不实现 DeltaNet、Hybrid Cache、FP8、MTP、Chunked Prefill 或完整 Qwen 模型类。
- 不把 Q/K norm、RoPE 与 gate 一次性融合成大型 kernel；本阶段只新增可独立理解和验收的 sigmoid gate 算子。

### 验收命令

Python shape、权重与集成测试：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_qwen_gated_attention.py' -v
```

完成 CUDA TODO 后重新编译：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m pip install \
  -v -e . --no-build-isolation
```

CUDA 数值测试：

```bash
MINIVLLM_RUN_CUDA_QWEN_ATTENTION_TESTS=1 \
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_qwen_gated_attention_cuda.py' -v
```

完整回归：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest discover -s tests -v
```

### 预提交基线

- 2026-08-26：阶段 4 共 12 项测试；2 项外围构造/校验测试通过，7 项精确停在 learner TODO，3 项 CUDA 测试在未重编译前按设计跳过。
- 完整测试共 63 项：50 项通过、7 项预期 TODO 错误、6 项 CUDA 按开关跳过；阶段 1~3 与 Prefix Cache 无额外回归。
- CUDA 源码中的 `sigmoid_and_mul` 目前以显式 `TORCH_CHECK` 占位，预提交基线未重新编译扩展；学习者完成后必须重新编译，并开启 CUDA 测试验证 gate 数值和 head size 256 的 decode/cached-prefill launcher。

### 原理验收

- 为什么 `q_proj` 输出不能直接在最后一维平分成 Query 与 gate？
- 为什么 Q/K RMSNorm 参数只有 `head_dim` 宽，并且 effective weight 是 `1 + weight`？
- gate 为什么不进入 KV Cache，decode 步的 gate 来自哪里？
- TP=2 时，Q checkpoint 源切片为什么是 `2 * q_size`？
- 为什么 gate 位于 PagedAttention 之后、`o_proj` 之前？
- 当前可读实现相较 vLLM 的 fused QK norm/RoPE/gate 少融合了哪些步骤？

### 完成记录

- 完成时间：2026-08-27。
- 学习者完成：Qwen 零中心 per-head RMSNorm、逐 Query head 的 Q/gate 拆分、Q/K/V TP 权重装载、Q/K norm 与 gated full-attention 前向链路，以及 FP32 中间计算的 sigmoid-and-mul CUDA 核心逻辑。
- Codex 收尾：补齐 QKV shard 错误诊断、门控算子 binding 与必要 shape 契约，开放 `head_size=256` 的 decode/cached-prefill specialization，并提供 Python、CUDA 数值和相邻 GQA 回归测试。
- 验证环境：WSL2 Ubuntu，`/home/yue/miniconda3/envs/mini-vllm`，Python 3.10.20、PyTorch 2.11.0+cu128、CUDA 12.8、NVIDIA GeForce RTX 4070 Laptop GPU、GCC/G++ 13.3.0。
- 编译结果：固定 `CC=/usr/bin/gcc`、`CXX=/usr/bin/g++` 后，全量构建并安装五个 CUDA 扩展成功；最终精简版 `activation_ops` 又从当前源码增量重编译成功。
- 验证结果：Python shape、权重与集成测试 9/9 通过；Qwen CUDA 测试 3/3 通过，覆盖 FP16/BF16/FP32 数值、shape 错误和 `head_size=256`；原有 GQA CUDA 回归 3/3 通过；完整测试共 63 项，其中 57 项执行通过、6 项 CUDA 用例按默认开关跳过，这 6 项均已显式开启并执行通过。
- 静态检查：`git diff --check` 无空白错误，仅有 Windows Git 的 LF/CRLF 转换提示。
- 编译提示：既有 `cache_kernels.cu` 仍有 `void*` 指针算术警告，既有 attention kernel 仍有未使用局部变量警告；本阶段新增 `sigmoid_and_mul` 未产生编译警告。
- 已知限制：门控算子是框架内部接口，依赖 `Qwen3_5Attention` 调用路径保证 CUDA、同 device/dtype 和 contiguous；本阶段仍不包含完整 Qwen 模型、视觉 M-RoPE、DeltaNet、Hybrid Cache、FP8、MTP 或 Chunked Prefill。
- 原理题保留作阶段复盘，本次按用户要求直接收口提交。
- 下一阶段：阶段 5“Gated DeltaNet 参考实现”保持未开始，等待讲义与作业脚手架准备。

## 阶段 5：Gated DeltaNet 参考实现

### 在推理链路中的位置

Qwen hybrid decoder 根据 `layer_types` 在 `full_attention` 与 `linear_attention` token mixer 之间切换。阶段 4 已实现 full-attention 分支；本阶段实现 linear-attention 分支的可读 PyTorch reference，覆盖 Q/K/V、z/a/b 投影、causal depthwise convolution、Gated Delta Rule、gated RMSNorm 和 output projection。

Reference 暂不接入 `Worker` 热路径：阶段 6 会用它验证 decode recurrence 与 prefill chunk kernel，阶段 7 再让 Hybrid Cache 管理 Conv State 和 Recurrent State，阶段 8 才装配完整 Qwen decoder 与真实 checkpoint。这个依赖顺序保证后续 kernel/cache 出错时仍有独立的数值 oracle。

### Codex 负责

- 定义 `GatedDeltaNetState`、reference 函数和完整层的稳定 shape/state 接口。
- 提供未分片的 Qwen projection、参数和 state 分配脚手架。
- 提供单 token 手算、因果性、prefill/decode 状态等价、gated RMSNorm、head grouping 和完整层集成测试。
- 在忽略的 `docs/qwen38-learning/stage-05-gated-deltanet-reference.md` 提供调用链、公式、shape、实现步骤和外部源码讲义。

### 学习者负责

完成 `minivllm/model_executor/layers/gated_delta_net.py` 中全部 `TODO(student, stage 5)`：

1. `causal_depthwise_conv1d_reference`
   - 实现 `[B,T,C]` depthwise causal convolution、SiLU 和可续接 Conv State。
2. `recurrent_gated_delta_rule_reference`
   - 实现 FP32 Q/K L2 normalize、Query scale，以及 decay/read/delta/write/output token recurrence。
3. `RMSNormGated.forward`
   - 实现沿 `value_head_dim` 的 FP32 RMSNorm、direct weight 和 `silu(z)` gate。
4. `Qwen3_5GatedDeltaNetReference.forward`
   - 串联五组投影、卷积、Q/K head expansion、beta/log-decay、recurrence、gated norm、flatten 和 output projection，并返回两类 final state。

预计自然实现量约 150 到 250 行，不需要为了行数增加抽象。

### 约束

- Reference 使用全局、未做 TP 切分的 `[batch, sequence, feature]` 张量；TP 留到真实模型集成阶段。
- Conv State 固定为 `[B, conv_dim, kernel_size]`，Recurrent State 固定为 `[B, Hv, Dk, Dv]`，二者都以 FP32 保存。
- `beta = sigmoid(b)`；`g = -exp(A_log) * softplus(a + dt_bias)`；实际 decay 是 `exp(g)`。
- Q/K 按最后一维 L2 normalize，只有 Q 再乘 `Dk^-0.5`。
- state 更新顺序固定为 decay、read、delta、write，再从更新后的 state 计算 output。
- Q/K heads 必须 repeat 到 value-head 数后再进入 recurrence。
- GDN 输出 gate 使用 `silu(z)`，不能复用阶段 4 的 sigmoid gate 公式。
- 不原地修改调用者传入的 initial state。
- 不依赖 FLA、causal-conv1d、vLLM 或 SGLang 的现成 kernel；本模块必须保持独立 oracle。
- 本阶段不实现 CUDA/Triton、chunk scan、Hybrid Cache、完整模型、TP、FP8、MTP 或视觉 M-RoPE。

### 验收命令

阶段 reference 测试：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_gated_delta_net_reference.py' -v
```

阶段 4 相邻回归：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_qwen_gated_attention.py' -v
```

完整回归：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest discover -s tests -v
```

### 预提交基线

- 2026-08-27：阶段 5 共 11 项测试；构造、参数/state shape 和非法 head grouping 2 项通过，其余 9 项精确停在四个 `TODO(student, stage 5)`。
- 阶段 4 Python 回归 9/9 通过。
- 完整测试共 74 项：59 项执行通过、9 项预期 learner TODO 错误、6 项 CUDA 按既有环境变量开关跳过；没有语法、导入、fixture 或非预期回归。
- 本阶段为纯 PyTorch reference，不需要重新编译 CUDA extension。

### 原理验收

- 为什么 recurrent state 是固定 `[Hv, Dk, Dv]`，而不像 KV Cache 那样随 token 数增长？
- 为什么 delta 写入的是 `v_t - k_t @ S_bar`？
- `alpha=exp(g)` 与 `beta=sigmoid(b)` 分别控制什么？
- 为什么 Q/K 都做 L2 normalize，但只有 Query 乘 `Dk^-0.5`？
- 为什么 prefill 与逐 token decode 必须得到相同 output、Conv State 和 Recurrent State？
- GDN 的 `silu(z)` gate 与 full-attention sigmoid gate 在计算链路中的位置有什么区别？
- 为什么优化 kernel 不能反过来充当本阶段 reference？

### 完成记录

- 完成时间：2026-08-30。
- 学习者完成：可续接的 causal depthwise convolution、FP32 Gated Delta Rule 递推、per-value-head gated RMSNorm，以及串联 QKV/z/a/b 投影、head grouping、两类 state 和 output projection 的完整 Gated DeltaNet reference。
- Codex 收尾：补齐卷积、递推、gated norm 和完整层的 shape/state 错误诊断，统一 FP32 state 与输出 dtype 契约，清理命名和格式，并增加 malformed-input 回归测试。
- 验证环境：WSL2 Ubuntu，`/home/yue/miniconda3/envs/mini-vllm`，Python 3.10.20、PyTorch 2.11.0+cu128，CUDA 可用。
- 验证结果：阶段 5 测试 14/14 通过，阶段 4 Qwen Gated Attention 回归 9/9 通过；完整测试共 77 项，其中 71 项执行通过、6 项既有 CUDA 用例按默认开关跳过。
- 静态检查：阶段文件无 learner TODO、`NotImplementedError` 或尾随空格；`git diff --check` 无空白错误，仅有 Windows Git 的 LF/CRLF 转换提示。
- 已知限制：本阶段是未分片、batch-major 的 PyTorch correctness oracle，尚未进入 Worker 热路径；优化 kernel、Hybrid Cache、完整模型装配和真实 checkpoint 分别留在后续既定阶段。
- 下一阶段：阶段 6“Gated DeltaNet Kernel”保持未开始，等待讲义和作业脚手架准备。

## 阶段 6：Gated DeltaNet Kernel

### 在推理链路中的位置

阶段 5 已实现 Qwen `linear_attention` 分支的独立 PyTorch oracle。本阶段优化其中两个逐 token 热点：causal depthwise convolution 的 decode state update，以及 Gated Delta Rule 的单 token decode 和分块 prefill。投影、Q/K normalize 与 query scale 位于算子上游，RMSNormGated 和 output projection 位于算子下游。

本阶段通过独立 Python wrapper 调用 CUDA 扩展并和 reference 对照，暂不接入 `Worker`。Conv State 和 Recurrent State 仍由调用者显式传入并原地更新；请求到 state slot 的映射、混合层缓存布局和生命周期由阶段 7 Hybrid Cache 负责。

### Codex 负责

- 新增独立 `gated_delta_net_ops` CUDA extension 和三个稳定 binding。
- 在 C++ 边界校验 CUDA、contiguous、shape、device、dtype 与 FP32 state 契约。
- 提供 lazy-import Python wrapper、FP32 Q/K normalize 和 query scale helper。
- 提供无需重编译的 wrapper contract 测试，以及默认跳过的 CUDA 数值测试。
- 在忽略的 `docs/qwen38-learning/stage-06-gated-deltanet-kernels.md` 提供调用链、公式、布局、线程划分、实现顺序和外部参考资料。

### 学习者负责

完成 `csrc/gated_delta_net_kernels.cu` 中全部 `TODO(student, stage 6)`：

1. `causal_conv1d_update_kernel`：原地移动短卷积窗口、写入当前 token、FP32 depthwise accumulation 和 SiLU。
2. `gated_delta_rule_decode_kernel`：以 `(batch, head)` 为 block、以 value dimension 为线程所有权，完成 decay/read/delta/write/output。
3. `gated_delta_rule_prefill_chunk_kernel`：在一个 chunk 内顺序复用 decode recurrence，并由 launcher 在当前 CUDA stream 上按 chunk 发起 kernel。
4. 三个 launcher：完成 float/half/bfloat16 dispatch、grid/block、指针传递与 current stream。

预计自然实现量约 200 到 300 行。不要改阶段 5 reference，也不需要新增框架抽象。

### 约束

- Conv State 固定为 `[B,C,K]` FP32，Recurrent State 固定为 `[B,H,Dk,Dv]` FP32，二者都原地更新。
- Q/K/V、beta 和输出允许 FP16、BF16、FP32；`log_decay`、所有递推累积和 state write 使用 FP32。
- recurrence operator 接收已经 L2 normalize 的 Q/K，且 Query 已乘 `Dk^-0.5`。
- decode 与 prefill 使用同一个 decay、read、delta、write、output 顺序。
- prefill 最后一个 chunk 可以短于 `chunk_size`；chunk 间不得清零或复制 state。
- CUDA grid 将 `(batch, head)` 展平到 `grid.x`，不依赖 `grid.y/grid.z`。
- 所有 launcher 使用 PyTorch 当前 CUDA stream，不插入 host 同步。
- 本阶段的 prefill 是顺序 chunk scan，不提前实现 FLA 的 WY 并行 chunk algorithm。
- 不实现 Hybrid Cache、packed request metadata、完整 Qwen 模型、TP、FP8、MTP 或视觉 M-RoPE。

### 验收命令

无需编译的 wrapper contract 测试：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_gated_delta_net_cuda_contract.py' -v
```

完成 CUDA TODO 后重新编译：

```bash
cd /mnt/e/Projects/mini-vllm
CC=/usr/bin/gcc CXX=/usr/bin/g++ \
/home/yue/miniconda3/envs/mini-vllm/bin/python -m pip install \
  -v -e . --no-build-isolation
```

CUDA 数值测试：

```bash
MINIVLLM_RUN_CUDA_GDN_TESTS=1 \
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_gated_delta_net_cuda.py' -v
```

Reference 与完整回归：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  discover -s tests -p 'test_gated_delta_net_reference.py' -v

/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest discover -s tests -v
```

### 预提交基线

- 2026-08-30：无需编译的阶段 6 wrapper contract 测试 7/7 通过，覆盖 FP32 Q/K 预处理、lazy extension import、三个算子的输出分配、state 原地语义和 `chunk_size` 校验。
- 阶段 5 reference 回归 14/14 通过。
- 完整测试共 89 项：78 项执行通过，5 项阶段 6 CUDA 数值测试因尚未完成并重编译 kernel 按设计跳过，另有 6 项既有 CUDA 开关测试跳过；没有导入、语法或旧功能回归。
- `csrc/gated_delta_net_kernels.cu` 保留三个 kernel 和三个 launcher 的显式 learner TODO。预提交基线不编译该扩展；完成 TODO 后必须重新编译，并开启 `MINIVLLM_RUN_CUDA_GDN_TESTS=1` 验证真实数值和 state continuation。

### 原理验收

- 为什么一个线程独占一个 value dimension 可以避免 recurrence 中的线程同步？
- 为什么跨 token 保存的 state 使用 FP32，而输出仍保持模型 dtype？
- 为什么同一 CUDA stream 上的 chunk launch 不需要逐次 host synchronize？
- 为什么 Q/K normalize 和 query scale 留在 operator 外部？
- 本阶段 chunk scan 与 FLA 的并行 chunk algorithm 有什么区别？
- 为什么请求 state slot 的所有权属于 Hybrid Cache，而不属于数值 kernel？

### 完成记录

- 完成时间：2026-08-30。
- 学习者完成：单 token causal depthwise convolution state update、以一个 block 对应一个 `(batch, head)` 的 Gated Delta Rule decode、共享 Q/K 的 chunk 内顺序 prefill，以及三个 current-stream CUDA launcher；Conv State 与 Recurrent State 均按接口原地更新。
- Codex 收尾：提供独立 C++ binding、lazy-import Python wrapper、FP32 Q/K 预处理、CUDA/reference 数值测试和 build 接线；清理尾随空格，并让 launch 错误处理保持项目现有风格一致。
- 验证环境：WSL2 Ubuntu，`/home/yue/miniconda3/envs/mini-vllm`，Python 3.10.20、PyTorch 2.11.0+cu128、CUDA Toolkit 12.8.93、NVIDIA GeForce RTX 4070 Laptop GPU（SM 8.9）、GCC/G++ 13.3.0。
- 编译结果：六个 CUDA extension 全量编译、链接和 editable install 成功；`gated_delta_net_ops` 未产生新增编译警告。PyTorch 未找到 Ninja 可执行入口而退回 distutils；既有 `cache_kernels.cu` 的 `void*` 指针算术警告和 attention kernel 的未使用变量警告仍存在。
- 验证结果：wrapper contract 7/7、Gated DeltaNet CUDA 数值测试 5/5、阶段 5 reference 14/14、GQA CUDA 回归 3/3、Qwen Gated Attention CUDA 回归 3/3 均通过；完整测试共 89 项，其中 78 项执行通过、11 项按默认 CUDA 开关跳过，这 11 项均已分别显式开启并执行通过。
- 边界与连续性：CUDA 数值测试覆盖 FP16/BF16/FP32、batch/head、非整 chunk 尾部、`chunk_size` 为 1/4/16、prefill final state 接续 decode 和 FP32 state 契约；额外 smoke 确认非 contiguous decode 输入在 binding 前置拒绝。
- 信息性计时：在 `B=2,T=64,H=4,Dk=Dv=128,FP16,chunk=16`、5 次 warmup 后，20 次 prefill 平均约 1.5592 ms，100 次 decode 平均约 0.0478 ms，最终 state 全部有限；本阶段不设置性能门槛。
- 已知限制：prefill 是同 stream 上按 chunk launch、chunk 内按 token 串行的教学实现，不是 FLA 的 WY 并行 chunk algorithm；算子仍使用 batch-major contiguous tensor 和显式 state，尚未接入 Worker、packed request metadata 或请求级 state slot。
- 下一阶段：阶段 7“Hybrid Cache”已准备讲义和作业脚手架，进入学习者实现。

## 阶段 7：Hybrid Cache

### 在推理链路中的位置

阶段 3 的 Paged KV Cache 按 token block 保存 full-attention 历史；阶段 5、6 的 Gated DeltaNet 则要求调用者为每个请求、每个 linear-attention layer 保存固定大小的 Conv State 和 Recurrent State。本阶段在两者之间增加层索引对齐的 Hybrid Cache：full-attention layer 继续引用既有 KV cache，linear-attention layer 从预分配 state pool 按稳定 request slot 读写状态。

本阶段不提前装配完整 Qwen decoder。阶段 8 会在 `Scheduler -> Worker._prepare_inputs -> model -> decoder layer` 链路上使用这里稳定下来的接口：根据实际 batch 中的 `seq_id` acquire slot，linear layer 在 kernel 前 gather state、kernel 后 scatter state，请求结束时 release，beam fork 时复制父状态。

### Codex 负责

- 新增 `GatedDeltaNetStateSpec`，从 Qwen text config 推导 Conv/Recurrent State shape。
- 定义层索引对齐的 `HybridCache` 稳定接口，并预分配每个 linear layer 的 FP32 state pool。
- 保留 full-attention layer index 到既有 KV cache 的显式映射，不改变当前 `CacheEngine` 和 PagedAttention。
- 编写 CPU 可执行的 shape、slot、gather/scatter、fork、release/reuse、reset 和错误契约测试。
- 在 `docs/qwen38-learning/stage-07-hybrid-cache.md` 提供不跟踪讲义、实现顺序和参考源码。

### 学习者作业

完成 `minivllm/worker/hybrid_cache.py` 中全部 `TODO(student, stage 7)`：

1. `RequestStateSlotAllocator`：实现稳定的 `seq_id -> slot` acquire、lookup、release 和 reset。
2. `HybridCache.acquire`：保持动态 batch 顺序；新请求的所有 linear-layer state 必须清零，已有请求不得清零。
3. `HybridCache.read_state`：按 slot gather 某一 linear layer 的连续 FP32 state batch。
4. `HybridCache.write_state`：把 kernel 更新结果 scatter 回长期 pool；由调用方保证 state 的 shape、FP32 dtype、device 和 batch 维匹配，不在逐层读写热路径增加校验。
5. `HybridCache.fork`：为 child 分配独立 slot，并复制 parent 在所有 linear layer 的 Conv/Recurrent State。
6. `HybridCache.release/reset`：释放所有权并清理 state，保证 slot 复用时不泄漏上一个请求历史。

预计自然实现量约 200 到 300 行。不要改 Scheduler、Worker、阶段 6 kernel 或完整 Qwen model；接口接线属于阶段 8。

### 约束

- `seq_id` 是长期请求身份，batch index 只是本轮位置；不得用 batch index 直接拥有 state。
- state pool 第一维固定为 `max_num_seqs`，动态 batch 只负责选择其中若干行。
- Conv State 固定为 `[slot, conv_dim, kernel_size]` FP32；Recurrent State 固定为 `[slot, H_v, D_k, D_v]` FP32。
- 同一活跃请求重复 acquire 必须得到同一 slot；不同活跃请求不得共享 slot。
- gather 返回的 batch state 被 kernel 原地更新后，必须通过 `write_state` 显式写回 pool。
- release/reuse 和 reset 都必须清理 linear-layer state；fork 必须复制所有 linear layer 且 parent/child 后续互不别名。
- full/linear layer 使用错误以及请求 ID、槽位所有权错误在 Python 管理边界处理；allocator 产生唯一且有效的 slot，read/write 不扫描 CUDA 索引值，避免逐层主机同步。
- read/write 的调用方负责提供匹配的 FP32 state 和有效索引；不保证非法输入的 Python 自定义异常或失败后原子回滚。现有张量元数据检查不涉及 CUDA 数据回传。
- 本阶段不实现 hybrid prefix-state snapshot、CPU state swap、跨 worker state transfer、TP state 分片或 CUDA Graph。
- 在阶段 8 完成 state-aware prefix 语义前，hybrid model 不得只复用 full-attention KV prefix 而跳过 GDN 前缀计算。

### 验收命令

阶段 7 聚焦测试：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  tests/test_hybrid_cache.py -v
```

阶段 5/6 状态与算子回归：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest \
  tests/test_gated_delta_net_reference.py \
  tests/test_gated_delta_net_cuda_contract.py -v
```

完整回归：

```bash
/home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest discover -s tests -v
```

### 预提交基线

- 2026-08-30：讲义、稳定接口和 19 项聚焦测试已准备；`docs/` 继续由 `.gitignore` 忽略。
- 阶段测试成功导入：配置与层布局 6 项通过，另外 13 项只在 `TODO(student, stage 7)` 指向的 slot/lifecycle 方法失败，没有语法、导入或 fixture 错误。
- 原有 89 项回归保持通过：78 项执行通过，11 项既有 CUDA 用例按环境开关跳过；本阶段是 CPU 状态管理，不需要重新编译 CUDA extension。

### 本轮修复与验证（2026-09-05）

- 按学习者要求修复 `_clear_rows`、`release`、`reset` 和 `fork`：按行原地清零；整批请求检查通过后归还槽位；保留按层缓存结构；拒绝复用活跃 child，并直接用 Python 整数槽位复制所有 linear-layer state。
- 不给 read/write 增加 state 校验；移除 `_normalize_slot_ids` 中读取 CUDA 索引值的 `torch.any` 和 `torch.unique` 检查。层类型和索引元数据检查仍保留，不读取 GPU 张量内容。
- 测试约定同步调整：read/write 的错误 shape/dtype 交给 PyTorch，不要求自定义 ValueError 或非法输入的原子回滚；补充新槽位清零、释放前全批检查、层 ID 与请求 ID 冲突、fork 所有权、固定池复用、空 batch 和读出副本独立性测试。
- 环境：WSL2 Ubuntu，`/home/yue/miniconda3/envs/mini-vllm/bin/python`，Python 3.10.20，PyTorch 2.11.0+cu128，NVIDIA GeForce RTX 4070 Laptop GPU。
- 聚焦结果：HybridCache 28 项、GDN reference 14 项、GDN wrapper contract 7 项，共 49 项全部通过。
- CUDA 同步检查：read/write 使用 CUDA 索引进行 CUDA Graph 捕获成功，连续两次 replay 正确更新池，未选择槽位保持不变。本用例只验证局部热路径，不代表完整模型 CUDA Graph 接线或吞吐基准。
- 完整回归：117 项全部执行通过，无跳过；使用已有扩展，没有重新编译。WSL 启动时有已有 localhost 代理提示，不影响测试。
- 阶段 7 已通过验收并完成收尾，提交为 `a8b65ba`；阶段 8 后续启动记录见下文。

复现命令（WSL 项目根目录）：

```bash
PYTHONPATH=tests /home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest -v \
  tests.test_hybrid_cache tests.test_gated_delta_net_reference \
  tests.test_gated_delta_net_cuda_contract

PYTHONPATH=tests MINIVLLM_RUN_CUDA_GDN_TESTS=1 \
  MINIVLLM_RUN_CUDA_GQA_TESTS=1 MINIVLLM_RUN_CUDA_QWEN_ATTENTION_TESTS=1 \
  /home/yue/miniconda3/envs/mini-vllm/bin/python -m unittest discover -s tests -v
```

### 原理验收

- 为什么动态 batch 可以使用固定大小 state pool？
- 为什么 request slot 绑定 `seq_id` 而不是本轮 batch index？
- 为什么 `index_select` gather 后必须显式 `index_copy_` scatter？
- 为什么 release/reuse 是正确性问题，而不仅是显存管理？
- 为什么 beam fork 必须复制 recurrent state，而不能只共享 slot？
- 为什么 full-attention Prefix Cache 命中不足以证明 hybrid prefix 可以直接跳过？
- Hybrid Cache、Scheduler 和 GDN kernel 各自拥有哪一部分状态与生命周期？

## 阶段 8：真实小模型端到端推理

### 当前状态与范围

- 2026-09-05 在 `codex/qwen38-learning` 基于阶段 7 提交 `a8b65ba` 准备讲义与作业，状态为“作业进行中”，不是阶段完成。
- 目标是把已经验收的 Attention、GDN CUDA 和 HybridCache 装配到真实 `LLM -> Engine -> Worker -> Model -> Sampler` 链路。
- 首次真实检查点验收使用 `Qwen/Qwen3.5-0.8B` 普通浮点权重。已核对其官方 config/权重索引及目标 `Qwen/Qwen3.8-27B` config，二者使用 Qwen3.5 schema；这只是小尺寸验收入口，不替代最终 27B 目标。
- 本阶段限定 TP=1、PP=1、单个未完成请求、`max_num_seqs=1`、文本输入。FP8、TP、动态批处理、hybrid prefix、调度级 chunked prefill、MTP、视觉/视频分别留在既有后续阶段，不增设阶段。

### 学习者作业

只完成 `minivllm/model_executor/models/qwen3_5.py` 的五处 `TODO(student, stage 8)`，预计自然实现约 180 到 280 行：

1. `Qwen3_5DecoderLayer.__init__`：按全局层类型构造 mixer、MLP 与两个 Qwen RMSNorm。
2. `Qwen3_5DecoderLayer.forward`：两次 pre-norm/residual；linear 分支读写对应层和请求槽位的状态。
3. `Qwen3_5GatedDeltaNet.forward`：复用已有参数；单 token 走 CUDA conv/decode，多 token 走 reference conv + CUDA prefill，维护 FP32 state。
4. `Qwen3_5Model.forward`：在 embedding 前截去 alignment padding，按全局层顺序运行并应用 final norm。
5. `Qwen3_5ForConditionalGeneration.load_weights_from_iterator`：语言模型权重映射、packed QKV/MLP 分片、共享 embedding 和加载完整性检查。

核心约束：padding 不推进 state；GDN Q/K 只归一化及缩放一次；packed 权重逐分片记录；未知语言模型权重不能静默跳过；read/write 热路径不增加 CUDA 张量值回传校验。建议按上述顺序实现，完成后请求“检查阶段 8”。

### Codex 已准备

- 不跟踪讲义 `docs/qwen38-learning/stage-08-end-to-end.md`：说明完整推理链路、数据维度、实现顺序、权重映射和就近参考链接，已确认被 `.gitignore` 忽略。
- 模型入口、embedding/lm_head 共享关系、Sampler 委托和 safetensors 分片读取；保留五个明确 TODO，未代写作业。
- CacheEngine 只为 full layers 分配 paged KV，保留全局层索引；修正 CPU cache 使用 GPU block 数的既有问题。
- Worker 创建长期 FP32 状态池，在 batch 边界解析请求槽位，接续 decode、重算时归零、完成/取消后释放。显存 profile 包含状态池，通过 `is_profile_run` 阻止向未分配 KV 的占位张量写入。
- Engine 在启动/请求边界拒绝尚未支持的 hybrid 功能；修正 detokenizer 返回累计文本时重复追加的问题，并补充回归测试。
- 隔离旧 Prefix Cache 测试的 `sys.modules` 替身，避免假 CacheEngine 污染后续新测试；不修改对应生产模块。
- `tests/test_qwen_stage8.py` 提供 learner 验收，`tests/test_qwen_stage8_runtime.py` 验证外围接线；`scripts/qwen_stage8_smoke.py` 供作业完成后用本地权重连续执行两次 greedy generate，并验证状态释放。

### 当前验证证据

- 环境：WSL2 Ubuntu 的 `/home/yue/miniconda3/envs/mini-vllm/bin/python`，沿用已编译扩展，没有重新编译或安装依赖。
- 既有 117 项回归加 11 项 Stage 8 外围测试，共 128 项全部执行通过，无跳过；既有 GDN/GQA/Qwen Attention CUDA 开关全部开启。
- Stage 8 聚焦基线共 19 个测试方法：11 项外围测试通过，8 个 learner 测试方法仅在保留的 TODO 失败；其中包含子测试，unittest 汇总为 13 个 `NotImplementedError`，没有其他异常。
- smoke 的 `--help` 导入检查通过；`git diff --check` 通过。尚未下载或执行真实检查点，没有模型端到端通过或性能结论。
- 完整模型 tiny logits oracle 将在作业可运行后依据安装的 Transformers 版本补充并执行；当前只有模块级 reference 对照，不能替代最终精度验收。
- 本次仅准备作业，未 commit/push。后续需学习者实现、审查、数值与真实模型验证，再按要求收尾提交。

### 运行命令

在 WSL 项目根目录并激活 `mini-vllm` 环境：

```bash
# 作业未完成前，只应出现明确的 learner TODO 错误。
PYTHONPATH=tests python -m unittest discover -s tests -p 'test_qwen_stage8*.py' -v

# 已有回归 + Codex 外围接线，不含尚未完成的 learner 测试。
PYTHONPATH=tests MINIVLLM_RUN_CUDA_GDN_TESTS=1 \
  MINIVLLM_RUN_CUDA_GQA_TESTS=1 MINIVLLM_RUN_CUDA_QWEN_ATTENTION_TESTS=1 \
  python -m unittest -v \
  test_gated_delta_net_cuda test_gated_delta_net_cuda_contract \
  test_gated_delta_net_reference test_gqa test_gqa_cuda test_hybrid_cache \
  test_model_architecture test_model_registry test_prefix_cache_blocks \
  test_prefix_cache_scheduler test_prefix_cache_worker \
  test_qwen_gated_attention test_qwen_gated_attention_cuda test_qwen_stage8_runtime

# 作业和数值测试通过后，使用已有本地普通浮点检查点。
python -m scripts.qwen_stage8_smoke --model models/Qwen3.5-0.8B
```

原理验收：说明两次 residual 的依赖关系、padding 对 recurrent state 的影响、两种 slot 的区别、profile 为什么不写 KV、packed 分片完整性，以及不能仅凭 full-attention KV 命中跳过 hybrid 前缀的原因。

### 2026-09-07：计算作业与权重加载进展

- 学习者已实现 Decoder 构造/forward、GDN CUDA forward 和 Model.forward；聚焦测试通过。按用户明确请求，最后的 `load_weights_from_iterator` 改由 Codex 实现，不继续保留为 learner TODO。
- loader 从本地 Parameter 建立检查点名字到目标参数/分片的映射；复用 QKV helper，MLP 按行段复制，其余参数直接复制。支持语言模型前缀规范化与共享 embedding，逐分片检查缺失、重复及准确形状；检查点提供两个共享别名时必须在目标精度下一致。
- 不改学习者的 forward、CUDA kernel 或缓存热路径；没有新增框架抽象。检查只在权重加载时执行。
- 扩充权重测试覆盖 GDN、两种 Norm、MLP 分片、共享/非共享 embedding、本地前缀、dtype 转换、别名前缀重复、过大 Q 矩阵和缺失项。WSL `mini-vllm` 环境下权重测试 9/9，完整回归 142/142 全部执行通过，无跳过；沿用已有扩展，没有重新编译。
- `git diff --check` 仍提示学习者 forward 中的既有尾随空白，本次未改动这些非 loader 行。讲义第 6.4 节补充实际代码阅读顺序，仍不跟踪。
- 尚未执行真实模型 smoke，也未补完完整模型 logits oracle；阶段 8 仍未完成最终验收，本轮没有 commit/push。

### 2026-09-07：保存并推送阶段 8 进展

- 按用户要求提交当前模型计算、权重加载、运行时接线、测试和 smoke 脚本，作为阶段 8 的进度保存点，不是阶段完成提交。
- 阶段状态更新为“待验收”；最近一次完整测试为 142/142 通过，之后仅清理模型文件行尾空白和更新此记录。
- 讲义保持忽略，不纳入提交。进入阶段 9 FP8 前，仍需完整小模型 logits 对照和真实检查点 prefill/decode、连续请求状态释放验证。
