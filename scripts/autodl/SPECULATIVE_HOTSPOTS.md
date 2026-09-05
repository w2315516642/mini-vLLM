# 验证 LM Head 与 Markov 优化

## 改动

目标侧验证位置按请求原顺序合并，一次做词表投影，随后按 offsets 切片。
不同请求的接受判断、EOS、随机拒绝采样和 logprobs 逻辑不变。
缓存容量计算新增合批 logits 工作区预留，因此 GPU block 数可能略降。

草稿 Markov 使用 Triton 分块点积，FP16/BF16 输入、FP32 累加。
多个请求复用词表权重 tile；写出每个 tile 的最大值和 ID，再做最终归约，
不生成完整的修正 scores。FP32 输入保留原 CUDA 路径。
7 步 Markov 依赖仍串行执行，不改变草稿 block size。
草稿 token IDs、confidence 每个张量只取回 CPU 一次，再在 CPU 切分。

两类 GEMM 优化可能改变浮点归约次序，不能承诺近似并列分数下 token
与旧版本逐 bit 一致。Markov 并列分数选最小 ID，改变的只是草稿候选；
目标验证规则未改变。随机路径和 greedy 都需回归，不能用接受率代替正确性。

## AutoDL 复测

无需重编译扩展，首次 Triton JIT 后保留预热。先用真实 markov_rank
覆盖下面的 128（这是代表值，不是从模型文件自动读取）：

```bash
CUDA_DEVICES=0 bash scripts/autodl/benchmark_speculative_hotspots.sh \
  --rank 128 --batch-sizes 1 16 --repeats 30
```

脚本对比原 CUDA、GEMM+argmax 和分块融合 Markov，以及逐请求/合批
LM Head，先检查结果再计时，输出额外峰值显存。LM Head 默认真实
词表248320、hidden5120，权重本身约2.37 GiB；显存不足可先缩小形状。
结果 `build/benchmarks/speculative-hotspots.json` 不代表整模型加速。

沿用之前的模型、数据集变量复测端到端及 Nsight：

```bash
CUDA_DEVICES=0 TP_SIZE=1 BATCH_SIZE=16 MAX_NUM_SEQS=16 \
MAX_NUM_BATCHED_TOKENS=8192 INPUT_LEN=512 OUTPUT_LEN=128 \
BENCH_MODE=dspark-static NUM_SPECULATIVE_TOKENS=3 \
BENCH_OUTPUT=build/benchmarks/hotspots-static-b16-k3.json \
bash scripts/autodl/benchmark_generation.sh

BENCH_MODE=dspark-static NUM_SPECULATIVE_TOKENS=3 \
NSYS_OUTPUT=build/benchmarks/nsys-hotspots-static-b16-k3 \
bash scripts/autodl/profile_decode_nsys.sh
```

检查目标 LM Head 是否每步一次大投影、原 markov_partial_argmax 是否
被分块 kernel 替代，再对比每步有效产出、TPOT 和无 profiler 吞吐。
本机笔记本结果不可外推为 A800 性能；A800 端到端需实测。
