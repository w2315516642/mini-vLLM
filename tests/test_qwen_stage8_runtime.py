"""Tests for Codex-owned stage 8 plumbing, independent of learner TODOs."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from minivllm.model_executor.weight_utils import hf_model_weights_iterator
from minivllm.worker.worker import Worker
from minivllm.worker.cache_engine import CacheEngine
from minivllm.engine.llm_engine import LLMEngine
from minivllm.engine import llm_engine as engine_module
from test_qwen_stage8 import tiny_config, make_cache, cpu_model_layers, qwen


class Stage8RuntimeTest(unittest.TestCase):
    def make_engine(self):
        engine = LLMEngine.__new__(LLMEngine)
        engine.model_config = SimpleNamespace(
            verify_with_parallel_config=Mock(),
            architecture=SimpleNamespace(num_linear_attention_layers=1,
                                         root_config=SimpleNamespace(),
                                         text_config=tiny_config()),
        )
        engine.parallel_config = SimpleNamespace(world_size=1)
        engine.scheduler_config = SimpleNamespace(max_num_seqs=1)
        engine.cache_config = SimpleNamespace(enable_prefix_caching=False,
                                              verify_with_parallel_config=Mock())
        engine.scheduler = Mock()
        engine.scheduler.has_unfinished_seqs.return_value = False
        engine._run_workers = Mock()
        engine._verify_args()
        return engine

    def test_engine_rejects_future_stage_features_at_startup(self):
        for field, value in (("world_size", 2), ("max_num_seqs", 2),
                             ("enable_prefix_caching", True),
                             ("quantization_config", {"quant_method": "fp8"})):
            with self.subTest(field=field):
                engine = self.make_engine()
                owner = {"world_size": engine.parallel_config,
                         "max_num_seqs": engine.scheduler_config,
                         "enable_prefix_caching": engine.cache_config,
                         "quantization_config": engine.model_config.architecture.root_config}[field]
                setattr(owner, field, value)
                with self.assertRaises(ValueError):
                    engine._verify_args()

    def test_engine_abort_releases_only_when_request_is_gone(self):
        engine = self.make_engine()
        engine.has_unfinished_requests = Mock(return_value=True)
        engine.abort_request("other")
        engine._run_workers.assert_not_called()
        engine.has_unfinished_requests.return_value = False
        engine.abort_request("active")
        engine._run_workers.assert_called_once_with("release_hybrid_state")

    def test_engine_completion_releases_state(self):
        engine = self.make_engine()
        scheduled = SimpleNamespace(blocks_to_swap_in={}, blocks_to_swap_out={},
                                    blocks_to_copy={})
        engine.scheduler.scheduler.return_value = ([object()], scheduled)
        engine.scheduler.update.return_value = []
        engine.has_unfinished_requests = Mock(return_value=False)
        self.assertEqual(engine.step(), [])
        self.assertEqual(engine._run_workers.call_args.args, ("release_hybrid_state",))
        engine.scheduler.free_finished_seq_groups.assert_called_once()

    def test_decoded_text_replaces_cumulative_text_instead_of_appending(self):
        engine = self.make_engine()
        engine.tokenizer = object()
        seq = SimpleNamespace(output_tokens=["a"], output_text="a",
                              get_last_token_id=lambda: 2)
        group = SimpleNamespace(get_seqs=lambda **kwargs: [seq])
        with patch.object(engine_module, "detokenize_incrementally", return_value=("b", "ab")):
            engine._decode_sequences([group])
        self.assertEqual(seq.output_text, "ab")
        self.assertEqual(seq.output_tokens, ["a", "b"])

    def test_safetensors_shards_are_read_on_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            save_file({"a": torch.ones(2)}, str(Path(folder) / "model-1.safetensors"))
            save_file({"b": torch.zeros(3)}, str(Path(folder) / "model-2.safetensors"))
            weights = dict(hf_model_weights_iterator(folder))
            self.assertEqual(set(weights), {"a", "b"})
            torch.testing.assert_close(weights["a"], torch.ones(2))
            self.assertEqual(weights["b"].device.type, "cpu")

    def test_empty_checkpoint_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                list(hf_model_weights_iterator(folder))

    def test_attention_reads_nested_rope_parameters(self):
        with cpu_model_layers():
            attention = qwen.Qwen3_5Attention(tiny_config())
        self.assertEqual(attention.rotary_dim, 16)
        inv_freq = 1.0 / (10000000.0 ** (torch.arange(0, 16, 2) / 16))
        expected = torch.cat([inv_freq.cos(), inv_freq.sin()])
        torch.testing.assert_close(attention.attn.cos_sin_cache[1], expected)

    def make_worker(self):
        worker = Worker.__new__(Worker)
        worker.hybrid_cache = make_cache(tiny_config())
        worker.hybrid_seq_id = None
        worker.hybrid_num_computed = 0
        return worker

    def test_profile_attention_does_not_write_placeholder_kv(self):
        with cpu_model_layers():
            attention = qwen.Qwen3_5Attention(tiny_config())
        projected = (torch.zeros(2, attention.q_size),) * 4
        placeholder = (torch.empty(0), torch.empty(0))
        with patch.object(attention, "_project_qkv", return_value=projected), \
             patch.object(attention.attn, "forward", return_value=torch.zeros(2, attention.q_size)) as run_attn, \
             patch.object(attention.gate_fn, "forward", side_effect=lambda output, gate: output):
            attention(torch.arange(2), torch.zeros(2, 16), placeholder,
                      SimpleNamespace(is_profile_run=True), None)
        args = run_attn.call_args.args
        self.assertIsNone(args[4])
        self.assertIsNone(args[5])

    def metadata(self, seq_id, start, length, prompt):
        return SimpleNamespace(
            seq_data={seq_id: object()}, is_prompt=prompt,
            num_computed_tokens={seq_id: start}, num_scheduled_tokens={seq_id: length},
        )

    def test_worker_keeps_state_across_decode_and_releases_completion(self):
        worker = self.make_worker()
        batch = SimpleNamespace()
        worker._prepare_hybrid_state([self.metadata(10, 0, 5, True)], batch)
        pool = worker.hybrid_cache._linear_state_pools[0]
        pool.conv_state.fill_(7)
        worker._prepare_hybrid_state([self.metadata(10, 5, 1, False)], batch)
        self.assertEqual(batch.state_slot_ids.tolist(), [0])
        self.assertTrue(torch.all(pool.conv_state == 7))
        self.assertEqual(worker.hybrid_num_computed, 6)
        worker.release_hybrid_state()
        self.assertIsNone(worker.hybrid_seq_id)
        self.assertEqual(worker.hybrid_cache.num_active_slots, 0)
        self.assertEqual(torch.count_nonzero(pool.conv_state).item(), 0)

    def test_recompute_resets_state_and_invalid_continuation_is_rejected(self):
        worker = self.make_worker()
        batch = SimpleNamespace()
        worker._prepare_hybrid_state([self.metadata(10, 0, 5, True)], batch)
        pool = worker.hybrid_cache._linear_state_pools[0]
        pool.conv_state.fill_(7)
        worker._prepare_hybrid_state([self.metadata(10, 0, 5, True)], batch)
        self.assertEqual(torch.count_nonzero(pool.conv_state).item(), 0)
        for metadata in (self.metadata(20, 5, 1, False),
                         self.metadata(10, 4, 1, False),
                         self.metadata(10, 4, 1, True)):
            with self.assertRaises(ValueError):
                worker._prepare_hybrid_state([metadata], batch)

    def test_cache_allocates_only_full_layers_and_counts_their_bytes(self):
        engine = CacheEngine.__new__(CacheEngine)
        engine.num_layers = 2
        engine.layer_types = ("linear_attention", "full_attention")
        engine.num_gpu_blocks, engine.num_cpu_blocks = 3, 1
        engine.block_size, engine.head_size, engine.num_kv_heads = 4, 64, 1
        engine.dtype = torch.float32
        original_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs)

        with patch("torch.empty", side_effect=cpu_empty):
            gpu, cpu = engine.allocate_gpu_cache(), engine.allocate_cpu_cache()
        self.assertEqual(gpu[0], (None, None))
        self.assertEqual(cpu[0], (None, None))
        self.assertEqual(gpu[1][0].shape[0], 3)
        self.assertEqual(cpu[1][0].shape[0], 1)
        config = SimpleNamespace(
            get_head_size=lambda: 64, get_num_layers=lambda p: 2,
            get_num_kv_heads=lambda p: 1, dtype=torch.float32,
            architecture=SimpleNamespace(num_linear_attention_layers=1,
                                         num_full_attention_layers=1),
        )
        self.assertEqual(CacheEngine.get_cache_block_size(4, config, object()),
                         2 * 4 * 64 * 4)


if __name__ == "__main__":
    unittest.main()
