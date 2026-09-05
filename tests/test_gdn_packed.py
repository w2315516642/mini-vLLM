import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from minivllm.model_executor.layers.gated_delta_net import (
    causal_depthwise_conv1d_reference,
)
from minivllm.model_executor.layers.gated_delta_net_cuda import (
    causal_conv1d_varlen, gated_delta_rule_prefill,
    gated_delta_rule_varlen, prepare_gated_delta_qk,
)
from minivllm.worker.gdn_replay import GatedDeltaNetReplay, replay_buffer_bytes
from minivllm.worker.hybrid_cache import GatedDeltaNetStateSpec, HybridCache


RUN_CUDA = (os.environ.get("MINIVLLM_RUN_CUDA_GDN_TESTS") == "1"
            and torch.cuda.is_available())


class ReplayContractTest(unittest.TestCase):
    def test_reserve_counts_snapshot_and_compact_inputs(self):
        spec = GatedDeltaNetStateSpec(2, 4, 3, 5, 4)
        state = 2 * (spec.conv_dim * 4 + 4 * 3 * 5) * 4
        inputs = 6 * ((spec.conv_dim + 4 * (3 + 5 + 1)) * 2 + 4 * 4)
        self.assertEqual(replay_buffer_bytes(spec, 3, 2, 6, torch.bfloat16),
                         3 * (state + inputs))

    def test_record_excludes_unrelated_prompts_and_preserves_snapshot_order(self):
        metadata = SimpleNamespace(prompt_seq_ids=[10, 20, 30], prompt_lens=[9, 2, 3],
                                   slot_mapping=torch.arange(14))
        snapshot = SimpleNamespace(seq_ids=(30, 20), layer_states={})
        replay = GatedDeltaNetReplay(metadata, snapshot)
        self.assertEqual(replay.token_indices.tolist(), [11, 12, 13, 9, 10])
        self.assertEqual(replay.cu_seqlens.tolist(), [0, 3, 5])
        self.assertEqual(replay.query_lens, [3, 2])
        with self.assertRaisesRegex(ValueError, "Committed GDN"):
            replay.commit(None, {30: 4, 20: 1})
        with self.assertRaisesRegex(ValueError, "Committed GDN"):
            replay.commit(None, {30: 0, 20: 1})


@unittest.skipUnless(RUN_CUDA, "enable MINIVLLM_RUN_CUDA_GDN_TESTS")
class PackedGdnCudaTest(unittest.TestCase):
    def test_varlen_conv_and_gdn_match_independent_sequences(self):
        torch.manual_seed(341)
        lengths = [1, 3, 67, 0, 2]
        offsets = [0, 1, 4, 71, 71, 73]
        cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for limits in (lengths, [0, 1, 64, 0, 1]):
                with self.subTest(dtype=dtype, limits=limits):
                    q, k = prepare_gated_delta_qk(
                        torch.randn(73, 3, 7, device="cuda", dtype=dtype),
                        torch.randn(73, 3, 7, device="cuda", dtype=dtype))
                    v = torch.randn(73, 3, 11, device="cuda", dtype=dtype)
                    decay = -torch.rand(73, 3, device="cuda")
                    beta = torch.rand(73, 3, device="cuda", dtype=dtype)
                    state = torch.randn(5, 3, 7, 11, device="cuda")
                    initial = state.clone()
                    qkv = torch.randn(73, 17, device="cuda", dtype=dtype)
                    weight = torch.randn(17, 4, device="cuda", dtype=dtype)
                    conv = torch.randn(5, 17, 4, device="cuda")
                    conv_initial = conv.clone()
                    accepted = (None if limits == lengths else
                                torch.tensor(limits, dtype=torch.int32, device="cuda"))
                    out = gated_delta_rule_varlen(q, k, v, decay, beta, state,
                                                 cu, max(lengths), accepted)
                    conv_out = causal_conv1d_varlen(qkv, conv, weight, cu, accepted)
                    for row, count in enumerate(limits):
                        begin, end = offsets[row], offsets[row] + count
                        expected_state = initial[row:row + 1].clone()
                        if count:
                            expected = gated_delta_rule_prefill(
                                q[begin:end].unsqueeze(0), k[begin:end].unsqueeze(0),
                                v[begin:end].unsqueeze(0), decay[begin:end].unsqueeze(0),
                                beta[begin:end].unsqueeze(0), expected_state)
                            torch.testing.assert_close(out[begin:end], expected[0])
                            expected_conv, conv_final = causal_depthwise_conv1d_reference(
                                qkv[begin:end].unsqueeze(0), weight,
                                conv_initial[row:row + 1])
                            torch.testing.assert_close(conv_out[begin:end], expected_conv[0],
                                                       atol=3e-2, rtol=3e-2)
                            torch.testing.assert_close(conv[row], conv_final[0])
                        else:
                            torch.testing.assert_close(conv[row], conv_initial[row])
                        torch.testing.assert_close(state[row], expected_state[0])

    def test_replay_mixed_acceptance_matches_each_prefix_and_ignores_other_rows(self):
        for counts in ((4, 1), (2, 3), (2, 1), (4, 3)):
            with self.subTest(counts=counts):
                self._check_replay(counts)

    def _check_replay(self, counts):
        torch.manual_seed(342)
        spec = GatedDeltaNetStateSpec(1, 2, 3, 4, 3)
        cache = HybridCache(("linear_attention",), {}, spec, 3, "cuda")
        slots = cache.acquire([10, 20, 30])
        state = cache._read_state(0, slots)
        state.conv_state.normal_()
        state.recurrent_state.normal_()
        cache._write_state(0, slots, state)
        snapshot = cache.snapshot([20, 10])
        metadata = SimpleNamespace(prompt_seq_ids=[10, 30, 20], prompt_lens=[3, 7, 4],
                                   slot_mapping=torch.arange(14, device="cuda"))
        replay = GatedDeltaNetReplay(metadata, snapshot)
        qkv = torch.randn(14, spec.conv_dim, device="cuda")
        weight = torch.randn(spec.conv_dim, 3, device="cuda")
        q, k = prepare_gated_delta_qk(torch.randn(14, 2, 3, device="cuda"),
                                     torch.randn(14, 2, 3, device="cuda"))
        v, beta = torch.randn(14, 2, 4, device="cuda"), torch.rand(14, 2, device="cuda")
        decay = -torch.rand(14, 2, device="cuda")
        replay.record(0, qkv, weight, k, v, decay, beta)
        expected = []
        for index, (start, count) in enumerate(zip((10, 0), counts)):
            initial = snapshot.layer_states[0]
            expected_rec = initial.recurrent_state[index:index + 1].clone()
            gated_delta_rule_prefill(q[start:start + count].unsqueeze(0),
                                    k[start:start + count].unsqueeze(0),
                                    v[start:start + count].unsqueeze(0),
                                    decay[start:start + count].unsqueeze(0),
                                    beta[start:start + count].unsqueeze(0), expected_rec)
            _, expected_conv = causal_depthwise_conv1d_reference(
                qkv[start:start + count].unsqueeze(0), weight,
                initial.conv_state[index:index + 1])
            expected.append((expected_conv[0], expected_rec[0]))
        # Simulate verification advancing all rows, including an unrelated one.
        changed = cache._read_state(0, slots)
        changed.conv_state.fill_(19)
        changed.recurrent_state.fill_(23)
        # Fully accepted rows have already reached their final state during
        # verification. They must remain bitwise untouched by commit().
        for index, (count, limit, slot) in enumerate(zip(counts, (4, 3), (1, 0))):
            if count == limit:
                changed.conv_state[slot].copy_(expected[index][0])
                changed.recurrent_state[slot].copy_(expected[index][1])
        cache._write_state(0, slots, changed)
        rejected = sum(n < limit for n, limit in zip(counts, (4, 3)))
        with patch.object(cache, "_normalize_slot_ids", side_effect=AssertionError), \
                patch("minivllm.worker.gdn_replay.causal_conv1d_varlen",
                      wraps=causal_conv1d_varlen) as conv_call, \
                patch("minivllm.worker.gdn_replay.gated_delta_rule_varlen",
                      wraps=gated_delta_rule_varlen) as gdn_call:
            self.assertEqual(replay.commit(cache, dict(zip((20, 10), counts))), rejected)
            self.assertEqual(conv_call.call_count, int(rejected > 0))
            self.assertEqual(gdn_call.call_count, int(rejected > 0))
            if rejected:
                self.assertEqual(conv_call.call_args.args[1].shape[0], rejected)
                self.assertEqual(gdn_call.call_args.args[5].shape[0], rejected)
        actual = cache._read_state(0, cache.acquire([20, 10, 30]))
        for index, (conv, rec) in enumerate(expected):
            torch.testing.assert_close(actual.conv_state[index], conv)
            torch.testing.assert_close(actual.recurrent_state[index], rec)
            if counts[index] == (4, 3)[index]:
                self.assertTrue(torch.equal(actual.conv_state[index], conv))
                self.assertTrue(torch.equal(actual.recurrent_state[index], rec))
        self.assertTrue(torch.all(actual.conv_state[2] == 19))
        self.assertTrue(torch.all(actual.recurrent_state[2] == 23))


if __name__ == "__main__":
    unittest.main()
