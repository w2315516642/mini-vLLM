# DSpark 微调对照评测

在相同的独立评测输入上比较原版和 ft20 草稿模型，不修改推理流程。
先激活 `vllm` 环境，在仓库根目录运行：

```bash
python -m benchmarks.prepare_heldout \
  --source /root/autodl-tmp/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json \
  --exclude \
    /root/autodl-tmp/SpecForge/cache/dataset/sharegpt-smoke/sharegpt_train_regen.jsonl \
    /root/autodl-tmp/SpecForge/cache/dataset/sharegpt-smoke/sharegpt_eval_regen.jsonl \
  --tokenizer /root/autodl-tmp/models/Qwen3.8-27B-FP8 \
  --output build/datasets/sharegpt-heldout-100.jsonl

export DATASET="$PWD/build/datasets/sharegpt-heldout-100.jsonl"
bash scripts/autodl/benchmark_draft_pair.sh
```

筛选器排除训练及验证数据的同源 ID 和重复用户文本，并对实际使用的前 512 个 token 去重。
不足 100 条时直接报错，不循环补样本。旁边的 manifest 文件记录输入文件哈希、参数及所选 ID。
生成的数据和 manifest 不覆盖已有文件。

对照固定为单卡、BF16、B1、K3、输入 512、输出 128、100 个请求、2 次预热、seed 42。
运行顺序为原版、ft20、ft20、原版，共 400 个计量请求；不是短时间 smoke test。
模型路径可通过 `TARGET_MODEL`、`ORIGINAL_DRAFT_MODEL` 和 `FINETUNED_DRAFT_MODEL` 覆盖。
使用 `MODEL_ROOT` 可统一改变默认模型目录，环境选择沿用 `common.sh` 的 `CONDA_ENV`。

结果位于 `build/benchmarks/heldout-{original,ft20}-b1-r{1,2}.json`，不会覆盖已有结果。
先确认四份结果的 workload 哈希一致，再比较平均接收长度、接收率、吞吐和 TPOT。
两轮结果要分别报告，不应只挑最好的一次。首次评测后，这个集合就是验证集；不要反复根据它调参后再称其为最终测试集。

## 范围与限制

- 沿用现有 benchmark：首条用户消息，不加聊天模板，截取前 512 个 token。不是训练聊天模板下的等价评估。
- 排除规则不是语义去重，不能保证消除所有近似内容。
- 不验证输出 token 内容一致性；接收率和性能比较不能代替正确性验证。
- 固定评测文件应保持不变，训练规模扩大后需重新核对数据隔离。

筛选逻辑测试：

```bash
python -m unittest discover -s tests -p test_prepare_heldout.py -v
```
