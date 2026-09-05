# FP8 Linear 与 GDN Prefill 优化

## 推理链路中的位置

`Worker.execute_model -> Qwen layer -> Column/RowParallelLinear -> _linear`
根据当前 GEMM 的输入 Token 数 M 选择实现。FP8 权重和原有 block scale 的
加载、TP 切分方式不变，RowParallelLinear 继续直接写入 all-reduce 的固定缓冲区。

- CUDA BF16/FP16、M <= 512：融合 W8A16 kernel，在片上反量化权重 tile 后
  使用 Tensor Core 计算。少 Token 时通过 split-K 增加并行度，额外缓冲仅与输出有关。
- M > 512：仍使用原来的逐层临时反量化加 cuBLAS。微基准显示当前融合实现在
  大 Prefill 上尚未超过 cuBLAS，所以暂不默认启用。这个阈值是保守的初始选择，
  不是针对所有 GPU 的最优值，需要在 A800 上复测。
- CPU/FP32：保留原来的参考实现。

没有在启动时把整个模型还原成 BF16；所有路径的常驻权重仍是 FP8。
不要将 M > 512 的微基准中 `optimized_ms` 当成生产默认路径的耗时，
JSON 中的 `runtime_backend` 会明确标注实际选择。

`Qwen GDN -> gated_delta_rule_prefill/varlen -> gdn_prefill`
在序列长度至少 16 时使用新的状态驻留 kernel。每个 CTA 持有一个 head 的
16 个 value 列，沿 key 维并行归约，FP32 状态在序列开始时读取、结束时写回。
时间维仍按原始递推公式串行处理；这不是 WY/矩阵化的 chunk-parallel 算法。
短序列及 decode 保持原 CUDA kernel。

packed offsets、accepted lengths、state 的 [B,H,Dk,Dv] 布局和原地写入契约
不变，因此 HybridCache、PD 迁移、DSpark 接受前缀回放无需修改。

## 运行与验证

新 kernel 使用 Triton JIT。安装好已有 CUDA 扩展后，此次不需要重新编译 C++：

```bash
conda activate vllm
python -c 'import torch, triton; print(torch.__version__, triton.__version__)'
MINIVLLM_RUN_FUSED_TESTS=1 python -m unittest tests.test_fused_inference -v
```

第一次执行会针对实际 GPU 编译 kernel，正式 benchmark 要保留预热。
W8A16 的 E4M3FN 解码使用位运算，支持 SM80，不依赖原生 FP8 指令。

```bash
CUDA_DEVICES=0 DTYPE=bfloat16 REPEATS=10 \
  bash scripts/autodl/benchmark_fused_inference.sh --tokens 1 16 128 512 8192
```

默认矩阵只是代表性尺寸。用实际 TP 本地权重的 N,K 替换：

```bash
bash scripts/autodl/benchmark_fused_inference.sh \
  --operator linear --linear-shape 5120,5120 --tokens 1 16 128 512 8192
```

脚本在同一组输入上比较旧 CUDA GDN 与新路径、显式反量化 Linear 与融合路径。
先检查数值和最终状态，再以 CUDA Event 计时；JIT、预热、state 恢复不计入时间。
两种测量顺序都执行，结果包括耗时、额外峰值显存和设备信息。
脚本不加载完整模型，不能将算子 speedup 当成端到端 speedup。

之后再用 `benchmark_generation.sh` 重跑相同 B1/B16、输入输出长度与数据集。
保持原有显存利用率和 Token 预算，观察 TTFT、TPOT 与整体吞吐。

## 实现参考

- [Triton GEMM 教程：分块、split work 与 L2 复用](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [NVIDIA Ampere：Tensor Core 支持的精度](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [Gated Delta Networks 论文](https://arxiv.org/abs/2412.06464)

本轮没有改模型结构、量化文件格式或 KV/GDN cache 所有权。
FP32 归约次序变化可能带来浮点舍入差异，测试检查容差，不要求逐 bit 相同。
