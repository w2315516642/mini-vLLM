"""Create a LOCAL FP8 text-weight smoke fixture, not an official quantized model.

Offline CPU conversion only. Inference never invokes this quantizer. Preserve
non-text weights and tokenizer files; write to a new directory, never in place.
"""
import argparse
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.nn import functional as F


def quantize_block_weight(weight, block_size=128):
    """Symmetric max-abs FP8 blocks; return compressed [N,K] and FP32 scales."""
    n, k = weight.shape
    b = block_size
    # Pad only during OFFLINE conversion so edge blocks use the same reduction.
    padded = F.pad(weight.float(), (0, (-k) % b, 0, (-n) % b))
    blocks = padded.reshape((n+b-1)//b, b, (k+b-1)//b, b)
    maxima = blocks.abs().amax(dim=(1, 3))
    scales = torch.where(maxima > 0, maxima / 448, torch.ones_like(maxima))
    packed = (blocks / scales[:, None, :, None]).clamp(-448, 448)
    packed = packed.reshape(padded.shape)[:n, :k].to(torch.float8_e4m3fn).contiguous()
    return packed, scales.contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    config = json.loads((source / "config.json").read_text())
    text_config = config.get("text_config", config)
    if text_config.get("model_type") != "qwen3_5_text":
        parser.error("This smoke converter supports Qwen3.5-schema text weights only")
    if config.get("quantization_config") or text_config.get("quantization_config"):
        parser.error("Source must be an unquantized checkpoint")
    if output.exists():
        parser.error("--output must be a NEW directory; existing files are never overwritten")
    files = sorted(source.glob("*.safetensors"))
    if not files:
        parser.error("No source safetensors found")
    output.mkdir(parents=True)
    weight_map, excluded = {}, []
    total_size, num_quantized = 0, 0
    for path in files:
        tensors = {}
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                weight = reader.get_tensor(name)
                module = name.removesuffix(".weight")
                is_text_linear = (name.startswith(("model.layers.", "model.language_model.layers."))
                                  and name.endswith(".weight") and weight.ndim == 2)
                # Keep the tiny decay/update-gate projections floating, as in
                # commonly distributed block-FP8 checkpoints.
                if is_text_linear and module.endswith((".in_proj_a", ".in_proj_b")):
                    excluded.append(module)
                    is_text_linear = False
                if is_text_linear:
                    tensors[name], tensors[module + ".weight_scale_inv"] = quantize_block_weight(weight)
                    num_quantized += 1
                else:
                    tensors[name] = weight
        save_file(tensors, output / path.name, metadata={"format": "pt"})
        for name, tensor in tensors.items():
            weight_map[name] = path.name
            total_size += tensor.numel() * tensor.element_size()
        print(f"Converted {path.name}", flush=True)
    config["quantization_config"] = {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": sorted(excluded) + ["lm_head", "model.visual", "mtp"],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size}, "weight_map": weight_map,
    }, indent=2) + "\n")
    # Copy tokenizer/config assets only, not original weights or weight indexes.
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".txt", ".jinja"):
            if path.name != "config.json" and not path.name.endswith(".index.json"):
                shutil.copy2(path, output / path.name)
    (output / "LOCAL_QUANTIZATION.txt").write_text(
        f"Locally converted from {source}\nBlock size 128x128; max-abs / 448.\n"
        "Text smoke fixture only; not an official checkpoint or quality benchmark.\n")
    print(f"FP8 projections={num_quantized}, tensor bytes={total_size}, output={output}")


if __name__ == "__main__":
    main()
