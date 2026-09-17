"""Full hybrid-model oracle; HF computes with independently restored weights."""
import os
import unittest

import torch

import test_qwen_stage8_logits as stage8
from minivllm.model_executor.layers.fp8 import FP8BlockConfig, Fp8Linear


@unittest.skipUnless(
    os.environ.get("MINIVLLM_RUN_CUDA_FP8_TESTS") == "1"
    and os.environ.get("MINIVLLM_RUN_QWEN_LOGITS_TESTS") == "1"
    and torch.cuda.is_available(),
    "enable FP8 CUDA and Qwen logits tests",
)
class FP8LogitsTest(stage8.Stage8LogitsTest):
    @staticmethod
    def make_models(config, dtype):
        reference, actual = stage8.Stage8LogitsTest.make_models(config, dtype)
        # Small blocks exercise packed Q/K/V, MLP, and short GDN projections.
        block = 16
        actual.configure_fp8(FP8BlockConfig((block, block)))
        checkpoint = {}
        for name, original in reference.state_dict().items():
            module_name = name.removesuffix(".weight")
            if not (name.endswith(".weight") and module_name.startswith("model.layers.")
                    and isinstance(reference.get_submodule(module_name), torch.nn.Linear)):
                checkpoint[name] = original
                continue
            n, k = original.shape
            weight = torch.empty_like(original, dtype=torch.float8_e4m3fn)
            scale = torch.empty((n + block - 1)//block, (k + block - 1)//block,
                                dtype=torch.float32, device=original.device)
            # Deliberately simple independent oracle, not the production loader
            # or dequantizer. Both models see identical quantization rounding.
            for row in range(0, n, block):
                for col in range(0, k, block):
                    tile = original[row:row+block, col:col+block]
                    s = tile.abs().max().clamp_min(1e-12) / 448
                    q = (tile / s).clamp(-448, 448).to(torch.float8_e4m3fn)
                    weight[row:row+block, col:col+block] = q
                    scale[row//block, col//block] = s
                    tile.copy_((q.float() * s).to(dtype).float())
            checkpoint[name] = weight
            checkpoint[module_name + ".weight_scale_inv"] = scale
        actual.load_weights_from_iterator(checkpoint.items())
        assert any(isinstance(m, Fp8Linear) for m in actual.modules())
        return reference, actual


if __name__ == "__main__":
    unittest.main()
