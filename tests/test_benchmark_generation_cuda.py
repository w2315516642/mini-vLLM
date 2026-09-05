"""Opt-in full CLI smoke with tiny random weights, not a performance claim."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(os.environ.get("MINIVLLM_RUN_BENCHMARK_CUDA_TESTS") == "1",
                     "Set MINIVLLM_RUN_BENCHMARK_CUDA_TESTS=1 for a tiny GPU CLI smoke")
class BenchmarkGenerationCUDATest(unittest.TestCase):
    def test_real_engine_streams_into_result_files(self):
        self._run_cli(False)

    @unittest.skipUnless(os.environ.get("MINIVLLM_RUN_NSYS_TESTS") == "1",
                         "Set MINIVLLM_RUN_NSYS_TESTS=1 with nsys installed")
    def test_short_nsys_capture(self):
        self._run_cli(True)

    def _run_cli(self, capture):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import Qwen3_5TextConfig, PreTrainedTokenizerFast
        from benchmarks.compare_results import compare_results

        with TemporaryDirectory() as root:
            root = Path(root)
            model = root / "model"
            config = Qwen3_5TextConfig(
                vocab_size=128, hidden_size=128, intermediate_size=256,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                head_dim=64, partial_rotary_factor=.5,
                linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16,
                linear_conv_kernel_dim=4,
                layer_types=["linear_attention", "full_attention"],
                max_position_embeddings=512, bos_token_id=1, eos_token_id=2,
                architectures=["Qwen3_5ForConditionalGeneration"],
                tie_word_embeddings=False, mtp_num_hidden_layers=0,
            )
            config.save_pretrained(model)
            vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}
            vocabulary.update({f"word{i}": i for i in range(3, 128)})
            tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                    bos_token="[BOS]", eos_token="[EOS]").save_pretrained(model)
            result_path = root / "result.json"
            profile_path = root / "stages.json"
            completed = subprocess.run([
                "bash", ("scripts/autodl/profile_decode_nsys.sh" if capture
                         else "scripts/autodl/benchmark_generation.sh"),
                "--use-dummy-weights", "--swap-space", "0",
                *([] if capture else ["--stage-profile-output", str(profile_path)]),
            ], cwd=Path(__file__).resolve().parents[1], env={**os.environ,
                "CONDA_ENV": Path(sys.prefix).name,
                "CONDA_SH": str(Path(sys.prefix).parents[1] / "etc/profile.d/conda.sh"),
                "TARGET_MODEL": str(model), "BENCH_MODE": "target", "DTYPE": "half",
                "CUDA_DEVICES": "0", "TP_SIZE": "1", "GPU_MEMORY_UTILIZATION": "0.15",
                "MAX_NUM_SEQS": "2", "MAX_NUM_BATCHED_TOKENS": "32",
                "INPUT_LEN": "16", "OUTPUT_LEN": "6", "BATCH_SIZE": "2",
                "NUM_BATCHES": "2", "WARMUP": "1", "SYNTHETIC": "1", "PREFIX_PRIME": "0",
                "BENCH_OUTPUT": str(result_path),
                "NSYS_OUTPUT": str(root / "trace"),
                "NSYS_SKIP_STEPS": "1", "NSYS_CAPTURE_STEPS": "2",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                text=True, capture_output=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            if capture:
                window = json.loads((root / "trace-window.json").read_text())["ranks"][0]
                self.assertTrue(window["complete"])
                self.assertEqual(len(window["records"]), 2)
                self.assertEqual(window["records"][0]["decode_step"], 2)
                self.assertTrue(all(r["M"] == 2 and r["T"] == 1 for r in window["records"]))
                self.assertGreater((root / "trace.nsys-rep").stat().st_size, 0)
                stats = subprocess.run([
                    "nsys", "stats", "--report", "nvtx_sum", "--format", "csv",
                    str(root / "trace.nsys-rep"),
                ], text=True, capture_output=True, timeout=60)
                self.assertEqual(stats.returncode, 0, stats.stdout + stats.stderr)
                for marker in ("decode_step", "decoder_layer", "layer=0", "linear:",
                               "gdn_recurrence", "gdn_conv", "full_attention",
                               "sample_verify", "lm_head:standard"):
                    self.assertIn(marker, stats.stdout)
            else:
                profile = json.loads(profile_path.read_text())
                self.assertEqual(len(profile["ranks"]), 1)
                steps = profile["ranks"][0]["steps"]
                self.assertEqual(len(steps), 12)  # Two measured batches, no warmup.
                self.assertEqual(sum(s["counts"]["prefill_requests"] > 0 for s in steps), 2)
                self.assertTrue(all("draft_proposal" in s["stages"] for s in steps))
                self.assertTrue(all(s["counts"]["replayed_requests"] == 0 for s in steps))
            result = json.loads(result_path.read_text())
            self.assertEqual(result["metrics"]["requests"], 4)
            self.assertEqual(result["metrics"]["output_tokens"], 24)
            self.assertEqual(result["metrics"]["itl_ms"]["count"], 20)
            self.assertEqual(result["speculative"]["verification_rounds"], 0)
            records = [json.loads(line) for line in result_path.with_suffix(".requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertTrue(all(record["output_tokens"] == 6 for record in records))
            self.assertEqual(compare_results(result, result)["metrics"]["ttft_ms.mean"]["speedup"], 1.)


if __name__ == "__main__":
    unittest.main()
