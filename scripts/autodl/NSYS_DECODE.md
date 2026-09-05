# Decode 内部时间线

沿用已设置的 TARGET_MODEL、DRAFT_MODEL、DATASET，在单卡环境运行：

```bash
BENCH_MODE=target bash scripts/autodl/profile_decode_nsys.sh
BENCH_MODE=dspark-static NUM_SPECULATIVE_TOKENS=3 bash scripts/autodl/profile_decode_nsys.sh
```

默认 B16、512 输入、128 输出、2 批预热、1 批测量；预热完成后才启用
采集控制。排除真实 Prefill，跳过 5 个 Decode 步，然后抓 20 步。
NSYS_SKIP_STEPS、NSYS_CAPTURE_STEPS 可覆盖窗口长度。
NSYS_OUTPUT 指定输出前缀；同名报告存在时请改前缀，不默认覆盖旧报告。
需要 nsys 在 PATH 中；不用重编译 CUDA 扩展。

## 产物

- `.nsys-rep`：用 Nsight Systems GUI 打开完整时间线。
- `-window.json`：实际采集窗口、每步 B/M/T 和 Worker 产出 token 数。
- `-generation.json`：带采集扰动的完整 benchmark 结果，不作为正式吞吐。

window 的 complete 必须为 true 才表示采够了指定步数。如果 batch 结束后
进入下一个真实 Prefill，窗口会提前关闭，避免跨批把 Prefill 混进来。
produced_tokens 是 Worker 返回 token 数，尾部可能包含被引擎输出上限裁掉
的 token；比较单位 token 成本时优先使用远离请求结束的完整 B16 窗口。

## 标记

`decode_step` 标出活跃请求 B、有效输入 M、最大查询长度 T。投机验证在
内部使用 prompt-shaped 元数据，但不是新的 Prefill，仍会进入采集。
`decoder_layer` 包含层号，其下嵌套 Linear、Full Attention、GDN 卷积、
递推、QK 归一化和回放输入收集。Linear 标明输入与权重形状。
`sample_verify` 下的 `lm_head` 单独标注词表投影。
`draft_proposal`、`draft_model`、`lm_head:draft` 和 `draft_context_kv`
用于区分草稿生成、草稿词表投影与上下文更新。
回放的 Conv/GDN 标记嵌套在 `state_replay` 中，不要算进目标验证。

范围是 CPU NVTX 调用区间，不代表 GPU kernel 在同一时间结束。
请通过 CUDA launch 关联查看对应 GPU 工作，父子范围不能重复相加。
只有采集开始与结束进行同步，算子和引擎步之间没有新增同步。
结束时停止采集，但不杀推理进程，让结果写入及剩余请求正常完成。
当前仅支持单个本地 Worker，不支持 Ray、TP/PP 多卡、prefix prime，
也不与 stage-event profiling 同时启用。

可先导出粗粒度汇总，但不要用它代替 GPU 时间线分析：

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
  build/benchmarks/nsys-dspark-static-b16-k3.nsys-rep
```

首先比较每个完整 B16 步的 Linear、短 GDN、LM Head 和同步空隙。
等步数不等于等产出；最终应按有效产出 token 数衡量投机收益。

参考：[Nsight Systems 官方采集窗口说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。
脚本使用 cudaProfilerApi 控制窗口及 capture-range-end=stop，继续执行目标程序。
