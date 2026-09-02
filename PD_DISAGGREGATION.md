# Prefill/Decode Disaggregation Reference

This branch implements a project-owned PD path inspired by Mooncake's
KVCache-centric architecture. It does not import or depend on Mooncake.

## Architecture

The control plane moves small Python request records over an authenticated
trusted-cluster RPC channel. The data plane registers persistent cache tensors
and pushes validated byte ranges over independent TCP connections.

```text
client -> P control RPC -> P Scheduler/Worker
                              |
                              | registered cache batch push
                              v
client -> D control RPC -> D Scheduler/Worker -> decode output
```

P retains its physical blocks and Qwen recurrent-state slots until every TP
rank receives an acknowledgement. D reserves its own block IDs and state slots
before P starts the push, but the request enters D's runnable queue only after
all rank transfers complete.

Qwen hybrid requests transfer both full-attention K/V and the FP32 Gated
DeltaNet convolution/recurrent state. P samples the first output token and
transfers state after the full prompt, so D starts by consuming that output
token. MTP proposals and M-RoPE positions are control metadata; image/video
feature tensors remain on P.

## Two-GPU Local Example

Use a model that fits independently on each GPU. A true PD deployment keeps
one complete model replica on P and another on D.

Start P:

```bash
CUDA_VISIBLE_DEVICES=0 python -m minivllm.entrypoints.pd_server \
  --model /path/to/model --dtype bfloat16 \
  --pd-role prefill --pd-transfer-backend tcp \
  --pd-endpoint-id p --pd-hostname 127.0.0.1:14000 \
  --pd-peer-endpoint-id d --pd-peer-hostname 127.0.0.1:14100 \
  --control-address 127.0.0.1:15000 --control-authkey local-secret
```

Start D:

```bash
CUDA_VISIBLE_DEVICES=1 python -m minivllm.entrypoints.pd_server \
  --model /path/to/model --dtype bfloat16 \
  --pd-role decode --pd-transfer-backend tcp \
  --pd-endpoint-id d --pd-hostname 127.0.0.1:14100 \
  --pd-peer-endpoint-id p --pd-peer-hostname 127.0.0.1:14000 \
  --control-address 127.0.0.1:15100 --control-authkey local-secret
```

Send one request:

```bash
python -m minivllm.entrypoints.pd_generate \
  --prefill-control 127.0.0.1:15000 \
  --decode-control 127.0.0.1:15100 \
  --control-authkey local-secret \
  --prompt "Explain paged KV cache." --max-tokens 32
```

Run the latency benchmark with the same endpoints:

```bash
python -m benchmarks.benchmark_pd \
  --prefill-control 127.0.0.1:15000 \
  --decode-control 127.0.0.1:15100 \
  --control-authkey local-secret --requests 20
```

For TP, give each role the same `--tensor-parallel-size`. Rank `r` uses base
data port plus `r`; P and D must expose contiguous ranks with identical cache
layouts. A Qwen3.8-27B deployment using TP=2 for each replica therefore needs
four suitable GPUs, not two GPUs shared by one TP=2 replica.

## Deliberate Limits

- The control RPC uses Python pickle and must remain on a trusted network.
- The TCP data plane stages CUDA bytes through host memory; it is a readable
  correctness reference, not an RDMA replacement.
- Heterogeneous P/D TP, cross-role prefix caching, beam search, layerwise
  transfer overlap, and multi-P/multi-D load balancing are not implemented.
