# Qwen3.8 学习式适配记录

## 协作规则

- 每个阶段由 Codex 完成外围脚手架、测试和接口说明。
- 每阶段开始时，在 `docs/qwen38-learning/` 提供一份不纳入 Git 的背景与实现讲义。
- 学习者负责一个约 100 到 300 行的核心模块，部分阶段的核心模块是 CUDA 算子。
- 作业验收前不提交阶段完成 commit。
- Codex 默认只提供概念级提示，不直接覆盖作业实现。
- 每阶段通过测试和原理问答后，以中文 commit 收口。
- CUDA 验证使用 `/root/miniconda3/envs/vllm` 环境。

## 阶段状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | 配置归一化与结构校验 | 已完成 |
| 2 | 模型注册与嵌套配置接入 | 未开始 |
| 3 | GQA 与独立 KV 头 | 未开始 |
| 4 | Qwen Gated Full Attention（含 Q/K RMSNorm 与门控融合算子） | 未开始 |
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
