"""Local operator measurements, not a whole-model or official Qwen benchmark."""
import argparse
import json
from statistics import median

import torch
from torch.nn import functional as F

from minivllm.model_executor.layers.fp8 import dequantize_fp8_block_weight, fused_fp8_linear


def measure(call, repeats):
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / repeats)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    output = call()
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    del output
    return {"median_ms": median(times), "peak_extra_bytes": extra}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    print(f"GPU={torch.cuda.get_device_name()} torch={torch.__version__}")
    torch.manual_seed(9)
    for n, k in ((1024, 1024), (3584, 1024)):
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.full((n//128, k//128), 0.02, device="cuda", dtype=torch.float32)
        for m in (1, 128):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            fused = lambda: fused_fp8_linear(x, weight, scale, (128, 128))
            # Split path is a benchmark reference ONLY, never a runtime fallback.
            split = lambda: F.linear(x, dequantize_fp8_block_weight(weight, scale, (128, 128), x.dtype))
            torch.testing.assert_close(fused(), split(), rtol=0.02, atol=0.02)
            print(json.dumps({
                "shape_mnk": [m, n, k], "dtype": str(x.dtype),
                "weight_and_scale_bytes": weight.numel() + scale.numel()*4,
                "fused": measure(fused, args.repeats),
                "split_reference": measure(split, args.repeats),
            }), flush=True)


if __name__ == "__main__":
    main()
