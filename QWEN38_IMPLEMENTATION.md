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
| 5 | Gated DeltaNet 参考实现 | 未开始 |
| 6 | Gated DeltaNet Kernel（含 Decode 状态更新与 Prefill 分块扫描算子） | 未开始 |
| 7 | Hybrid Cache | 未开始 |
| 8 | 真实小模型端到端推理 | 未开始 |
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
