import unittest
from types import SimpleNamespace

import torch

from minivllm.configs.model_architecture import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from minivllm.model_executor.layers.gated_delta_net import GatedDeltaNetState
from minivllm.worker.hybrid_cache import (
    GatedDeltaNetStateSpec,
    HybridCache,
    RequestStateSlotAllocator,
)


def _tiny_text_config(**overrides):
    values = {
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 3,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _state_spec():
    return GatedDeltaNetStateSpec.from_text_config(_tiny_text_config())


def _full_attention_caches():
    return {
        1: (torch.empty(2, 3), torch.empty(2, 4)),
        3: (torch.empty(5, 6), torch.empty(5, 7)),
    }


def _make_cache(max_num_seqs=3):
    return HybridCache(
        layer_types=(
            LINEAR_ATTENTION,
            FULL_ATTENTION,
            LINEAR_ATTENTION,
            FULL_ATTENTION,
        ),
        full_attention_caches=_full_attention_caches(),
        state_spec=_state_spec(),
        max_num_seqs=max_num_seqs,
        device="cpu",
    )


def _filled_state(spec, batch_size, value):
    return GatedDeltaNetState(
        conv_state=torch.full(
            (batch_size, *spec.conv_state_shape),
            value,
            dtype=torch.float32,
        ),
        recurrent_state=torch.full(
            (batch_size, *spec.recurrent_state_shape),
            value,
            dtype=torch.float32,
        ),
    )


class GatedDeltaNetStateSpecTest(unittest.TestCase):
    def test_config_fields_define_state_shapes(self):
        spec = _state_spec()

        self.assertEqual(spec.conv_dim, 20)
        self.assertEqual(spec.conv_state_shape, (20, 3))
        self.assertEqual(spec.recurrent_state_shape, (4, 3, 2))

    def test_config_dimensions_must_be_positive_integers(self):
        with self.assertRaisesRegex(ValueError, "linear_key_head_dim"):
            GatedDeltaNetStateSpec.from_text_config(
                _tiny_text_config(linear_key_head_dim=0)
            )
        with self.assertRaisesRegex(ValueError, "linear_conv_kernel_dim"):
            GatedDeltaNetStateSpec.from_text_config(
                _tiny_text_config(linear_conv_kernel_dim=1.5)
            )

    def test_value_heads_must_be_grouped_by_key_heads(self):
        with self.assertRaisesRegex(ValueError, "value_heads"):
            GatedDeltaNetStateSpec.from_text_config(
                _tiny_text_config(
                    linear_num_key_heads=3,
                    linear_num_value_heads=4,
                )
            )


class RequestStateSlotAllocatorTest(unittest.TestCase):
    def test_acquire_is_stable_and_capacity_is_bounded(self):
        allocator = RequestStateSlotAllocator(capacity=2)

        self.assertEqual(allocator.acquire(10), (0, True))
        self.assertEqual(allocator.acquire(20), (1, True))
        self.assertEqual(allocator.acquire(10), (0, False))
        self.assertEqual(allocator.lookup(20), 1)
        self.assertEqual(allocator.num_active_slots, 2)
        self.assertEqual(allocator.num_free_slots, 0)

        with self.assertRaisesRegex(ValueError, "free state slots"):
            allocator.acquire(30)

    def test_release_makes_the_same_slot_reusable(self):
        allocator = RequestStateSlotAllocator(capacity=2)
        allocator.acquire(10)
        allocator.acquire(20)

        self.assertEqual(allocator.release(10), 0)
        self.assertEqual(allocator.acquire(30), (0, True))

    def test_unknown_sequence_errors_are_explicit(self):
        allocator = RequestStateSlotAllocator(capacity=1)

        with self.assertRaisesRegex(ValueError, "not active"):
            allocator.lookup(7)
        with self.assertRaisesRegex(ValueError, "not active"):
            allocator.release(7)

    def test_reset_restores_deterministic_slot_order(self):
        allocator = RequestStateSlotAllocator(capacity=2)
        allocator.acquire(10)
        allocator.acquire(20)

        allocator.reset()

        self.assertEqual(allocator.num_active_slots, 0)
        self.assertEqual(allocator.num_free_slots, 2)
        self.assertEqual(allocator.acquire(30), (0, True))


class HybridCacheLayoutTest(unittest.TestCase):
    def test_acquire_allows_a_batch_that_exactly_fills_the_pool(self):
        cache = _make_cache(max_num_seqs=2)

        self.assertEqual(cache.acquire([10, 20]).tolist(), [0, 1])

    def test_failed_batch_acquire_does_not_consume_slots(self):
        cache = _make_cache(max_num_seqs=2)

        with self.assertRaisesRegex(ValueError, "free state slots"):
            cache.acquire([10, 20, 30])

        self.assertEqual(cache.acquire([40, 50]).tolist(), [0, 1])

    def test_full_attention_cache_keeps_global_layer_index(self):
        full_caches = _full_attention_caches()
        cache = HybridCache(
            layer_types=(
                LINEAR_ATTENTION,
                FULL_ATTENTION,
                LINEAR_ATTENTION,
                FULL_ATTENTION,
            ),
            full_attention_caches=full_caches,
            state_spec=_state_spec(),
            max_num_seqs=2,
            device="cpu",
        )

        self.assertIs(cache.get_kv_cache(1), full_caches[1])
        self.assertIs(cache.get_kv_cache(3), full_caches[3])
        with self.assertRaisesRegex(ValueError, "linear_attention"):
            cache.get_kv_cache(0)

    def test_full_attention_cache_keys_must_match_layer_types(self):
        with self.assertRaisesRegex(ValueError, r"expected \[1\]"):
            HybridCache(
                layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
                full_attention_caches={},
                state_spec=_state_spec(),
                max_num_seqs=1,
                device="cpu",
            )

    def test_unknown_layer_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sliding_attention"):
            HybridCache(
                layer_types=("sliding_attention",),
                full_attention_caches={},
                state_spec=_state_spec(),
                max_num_seqs=1,
                device="cpu",
            )

    def test_linear_state_pool_is_fp32_and_zero_initialized(self):
        cache = _make_cache(max_num_seqs=2)
        slots = cache.acquire([10, 20])

        state = cache.read_state(0, slots)

        self.assertEqual(state.conv_state.shape, (2, 20, 3))
        self.assertEqual(state.recurrent_state.shape, (2, 4, 3, 2))
        self.assertEqual(state.conv_state.dtype, torch.float32)
        self.assertEqual(state.recurrent_state.dtype, torch.float32)
        self.assertTrue(torch.count_nonzero(state.conv_state) == 0)
        self.assertTrue(torch.count_nonzero(state.recurrent_state) == 0)


class HybridCacheLifecycleTest(unittest.TestCase):
    def test_acquire_clears_new_rows_without_touching_existing_requests(self):
        cache = _make_cache(max_num_seqs=3)
        # Dirty free rows expose missing clears that constructor zeros would hide.
        for pool in cache._linear_state_pools.values():
            pool.conv_state.fill_(9.0)
            pool.recurrent_state.fill_(9.0)
        first = cache.acquire([10])
        for layer in (0, 2):
            fresh = cache.read_state(layer, first)
            self.assertEqual(torch.count_nonzero(fresh.conv_state).item(), 0)
            self.assertEqual(torch.count_nonzero(fresh.recurrent_state).item(), 0)
            cache.write_state(layer, first, _filled_state(cache.state_spec, 1, 3.0))

        self.assertEqual(cache.acquire([10, 20]).tolist(), [0, 1])
        for layer in (0, 2):
            state = cache.read_state(layer, [0, 1, 2])
            self.assertEqual(state.conv_state[:, 0, 0].tolist(), [3.0, 0.0, 9.0])
            self.assertEqual(
                state.recurrent_state[:, 0, 0, 0].tolist(), [3.0, 0.0, 9.0]
            )

    def test_dynamic_batch_reordering_preserves_request_state(self):
        cache = _make_cache(max_num_seqs=3)
        slots = cache.acquire([10, 20])
        cache.write_state(0, slots, _filled_state(cache.state_spec, 2, 1.0))

        reordered_slots = cache.acquire([20, 10])
        reordered = cache.read_state(0, reordered_slots)

        self.assertEqual(slots.tolist(), [0, 1])
        self.assertEqual(reordered_slots.tolist(), [1, 0])
        torch.testing.assert_close(
            reordered.conv_state,
            torch.ones_like(reordered.conv_state),
        )

    def test_write_state_scatter_is_visible_to_later_read(self):
        cache = _make_cache(max_num_seqs=3)
        slots = cache.acquire([10, 20])
        state = _filled_state(cache.state_spec, 2, 3.0)
        state.conv_state[1].fill_(7.0)
        state.recurrent_state[1].fill_(7.0)

        cache.write_state(2, slots, state)

        first = cache.read_state(2, cache.acquire([10]))
        second = cache.read_state(2, cache.acquire([20]))
        torch.testing.assert_close(
            first.conv_state, torch.full_like(first.conv_state, 3.0)
        )
        torch.testing.assert_close(
            second.recurrent_state,
            torch.full_like(second.recurrent_state, 7.0),
        )

    def test_release_clears_state_before_slot_reuse(self):
        cache = _make_cache(max_num_seqs=2)
        old_slot = cache.acquire([10])
        for layer in (0, 2):
            cache.write_state(
                layer, old_slot, _filled_state(cache.state_spec, 1, 9.0)
            )

        cache.release([10])
        self.assertEqual(cache.num_active_slots, 0)
        self.assertEqual(cache.num_free_slots, 2)
        for layer in (0, 2):
            cleared = cache.read_state(layer, old_slot)
            self.assertEqual(torch.count_nonzero(cleared.conv_state).item(), 0)
            self.assertEqual(torch.count_nonzero(cleared.recurrent_state).item(), 0)
        new_slot = cache.acquire([30])
        fresh_state = cache.read_state(0, new_slot)

        self.assertEqual(old_slot.tolist(), new_slot.tolist())
        self.assertTrue(torch.count_nonzero(fresh_state.conv_state) == 0)
        self.assertTrue(torch.count_nonzero(fresh_state.recurrent_state) == 0)

    def test_release_keeps_layer_caches_and_other_requests(self):
        cache = _make_cache(max_num_seqs=2)
        kv = cache.get_kv_cache(1)
        slots = cache.acquire([1, 20])
        cache.write_state(0, slots, _filled_state(cache.state_spec, 2, 5.0))

        cache.release([1])

        self.assertIs(cache.get_kv_cache(1), kv)
        self.assertEqual(cache.num_active_slots, 1)
        self.assertEqual(cache.acquire([20, 30]).tolist(), [1, 0])
        remaining = cache.read_state(0, [1])
        self.assertTrue(torch.all(remaining.conv_state == 5.0))
        self.assertTrue(torch.all(remaining.recurrent_state == 5.0))

    def test_release_checks_the_whole_batch_before_mutating(self):
        cache = _make_cache(max_num_seqs=2)
        slots = cache.acquire([10, 20])
        cache.write_state(0, slots, _filled_state(cache.state_spec, 2, 5.0))
        for seq_ids, message in (
            ([10, 99], "not active"),
            ([10, 10], "duplicates"),
            ([10, -1], "non-negative"),
        ):
            with self.subTest(seq_ids=seq_ids):
                with self.assertRaisesRegex(ValueError, message):
                    cache.release(seq_ids)
                self.assertEqual(cache.num_active_slots, 2)
                state = cache.read_state(0, slots)
                self.assertTrue(torch.all(state.conv_state == 5.0))
                self.assertTrue(torch.all(state.recurrent_state == 5.0))

    def test_fork_copies_every_linear_layer_without_aliasing(self):
        cache = _make_cache(max_num_seqs=3)
        parent_slot = cache.acquire([10])
        cache.write_state(
            0,
            parent_slot,
            _filled_state(cache.state_spec, 1, 2.0),
        )
        cache.write_state(
            2,
            parent_slot,
            _filled_state(cache.state_spec, 1, 5.0),
        )

        child_slot = cache.fork(10, 20)

        self.assertIsInstance(child_slot, int)
        self.assertEqual(child_slot, 1)
        child_layer_0 = cache.read_state(0, [child_slot])
        child_layer_2 = cache.read_state(2, [child_slot])
        torch.testing.assert_close(
            child_layer_0.conv_state,
            torch.full_like(child_layer_0.conv_state, 2.0),
        )
        torch.testing.assert_close(
            child_layer_2.recurrent_state,
            torch.full_like(child_layer_2.recurrent_state, 5.0),
        )

        cache.write_state(
            0,
            [child_slot],
            _filled_state(cache.state_spec, 1, 11.0),
        )
        parent_after = cache.read_state(0, parent_slot)
        torch.testing.assert_close(
            parent_after.conv_state,
            torch.full_like(parent_after.conv_state, 2.0),
        )

    def test_fork_rejects_invalid_ownership_without_overwriting_state(self):
        cache = _make_cache(max_num_seqs=2)
        slots = cache.acquire([10, 20])
        cache.write_state(0, slots, _filled_state(cache.state_spec, 2, 5.0))
        for parent, child, message in (
            (10, 20, "already active"),
            (10, 10, "duplicates"),
            (99, 30, "not active"),
            (10, -1, "non-negative"),
            (10, 30, "free state slots"),
        ):
            with self.subTest(parent=parent, child=child):
                with self.assertRaisesRegex(ValueError, message):
                    cache.fork(parent, child)
                self.assertEqual(cache.num_active_slots, 2)
                state = cache.read_state(0, slots)
                self.assertTrue(torch.all(state.conv_state == 5.0))
                self.assertTrue(torch.all(state.recurrent_state == 5.0))

    def test_write_state_leaves_tensor_metadata_errors_to_pytorch(self):
        cache = _make_cache(max_num_seqs=2)
        slots = cache.acquire([10])
        valid = _filled_state(cache.state_spec, 1, 1.0)

        wrong_shape = GatedDeltaNetState(
            conv_state=valid.conv_state[:, :-1],
            recurrent_state=valid.recurrent_state,
        )
        # There is no Python state-validation pass on the per-layer hot path.
        with self.assertRaises(RuntimeError):
            cache.write_state(0, slots, wrong_shape)

        wrong_dtype = GatedDeltaNetState(
            conv_state=valid.conv_state.half(),
            recurrent_state=valid.recurrent_state.half(),
        )
        with self.assertRaises(RuntimeError):
            cache.write_state(0, slots, wrong_dtype)

    def test_acquire_rejects_duplicate_requests_before_allocating(self):
        cache = _make_cache(max_num_seqs=2)

        with self.assertRaisesRegex(ValueError, "duplicates"):
            cache.acquire([10, 10])
        self.assertEqual(cache.num_active_slots, 0)
        self.assertEqual(cache.acquire([10, 20]).tolist(), [0, 1])

    def test_read_state_is_independent_and_keeps_slot_order(self):
        cache = _make_cache(max_num_seqs=3)
        cache.acquire([10, 20, 30])
        for slot, value in enumerate((1.0, 2.0, 3.0)):
            cache.write_state(0, [slot], _filled_state(cache.state_spec, 1, value))

        state = cache.read_state(0, [2, 0])
        self.assertEqual(state.conv_state[:, 0, 0].tolist(), [3.0, 1.0])
        self.assertEqual(state.recurrent_state[:, 0, 0, 0].tolist(), [3.0, 1.0])
        self.assertTrue(state.conv_state.is_contiguous())
        self.assertTrue(state.recurrent_state.is_contiguous())
        state.conv_state.zero_()
        state.recurrent_state.zero_()

        original = cache.read_state(0, [0, 1, 2])
        self.assertEqual(original.conv_state[:, 0, 0].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(
            original.recurrent_state[:, 0, 0, 0].tolist(), [1.0, 2.0, 3.0]
        )

    def test_empty_batches_preserve_existing_state(self):
        cache = _make_cache(max_num_seqs=1)
        slots = cache.acquire([10])
        cache.write_state(0, slots, _filled_state(cache.state_spec, 1, 5.0))
        self.assertEqual(cache.acquire([]).shape, (0,))
        empty = cache.read_state(0, [])
        self.assertEqual(empty.conv_state.shape, (0, 20, 3))
        self.assertEqual(empty.recurrent_state.shape, (0, 4, 3, 2))
        cache.write_state(0, [], empty)
        cache.release([])
        self.assertEqual(cache.num_active_slots, 1)
        self.assertTrue(torch.all(cache.read_state(0, slots).conv_state == 5.0))

    def test_linear_state_api_rejects_full_attention_layer(self):
        cache = _make_cache(max_num_seqs=1)
        slot = cache.acquire([10])

        with self.assertRaisesRegex(ValueError, "full_attention"):
            cache.read_state(1, slot)
        with self.assertRaisesRegex(ValueError, "full_attention"):
            cache.write_state(
                1,
                slot,
                _filled_state(cache.state_spec, 1, 1.0),
            )

    def test_reset_clears_ownership_and_all_linear_state(self):
        cache = _make_cache(max_num_seqs=2)
        pools = dict(cache._linear_state_pools)
        kv = {layer: cache.get_kv_cache(layer) for layer in (1, 3)}
        slots = cache.acquire([10, 20])
        cache.write_state(0, slots, _filled_state(cache.state_spec, 2, 4.0))
        cache.write_state(2, slots, _filled_state(cache.state_spec, 2, 8.0))

        cache.reset()
        self.assertEqual(cache.num_active_slots, 0)
        self.assertEqual(cache.num_free_slots, 2)
        for layer, original in pools.items():
            self.assertIs(cache._linear_state_pools[layer], original)
            self.assertEqual(torch.count_nonzero(original.conv_state).item(), 0)
            self.assertEqual(torch.count_nonzero(original.recurrent_state).item(), 0)
        for layer, original in kv.items():
            self.assertIs(cache.get_kv_cache(layer), original)
        fresh_slot = cache.acquire([30])

        self.assertEqual(fresh_slot.tolist(), [0])
        self.assertEqual(cache.num_active_slots, 1)
        self.assertEqual(cache.num_free_slots, 1)
        for layer_idx in (0, 2):
            state = cache.read_state(layer_idx, fresh_slot)
            self.assertTrue(torch.count_nonzero(state.conv_state) == 0)
            self.assertTrue(torch.count_nonzero(state.recurrent_state) == 0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA graph capture")
class HybridCacheCudaTest(unittest.TestCase):
    def test_read_write_capture_without_host_value_checks(self):
        cache = HybridCache(
            layer_types=(LINEAR_ATTENTION,),
            full_attention_caches={},
            state_spec=_state_spec(),
            max_num_seqs=3,
            device="cuda",
        )
        slots = cache.acquire([10, 20]).flip(0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                warmup = cache.read_state(0, slots)
                warmup.conv_state.add_(1.0)
                warmup.recurrent_state.add_(2.0)
                cache.write_state(0, slots, warmup)
        torch.cuda.current_stream().wait_stream(stream)

        # A Python bool/item conversion of a CUDA value would fail capture.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            state = cache.read_state(0, slots)
            state.conv_state.add_(1.0)
            state.recurrent_state.add_(2.0)
            cache.write_state(0, slots, state)

        before = cache.read_state(0, slots)
        for step in (1, 2):
            graph.replay()
            after = cache.read_state(0, slots)
            torch.testing.assert_close(after.conv_state, before.conv_state + step)
            torch.testing.assert_close(
                after.recurrent_state, before.recurrent_state + 2 * step
            )
        untouched = cache.read_state(0, [2])
        self.assertEqual(torch.count_nonzero(untouched.conv_state).item(), 0)
        self.assertEqual(torch.count_nonzero(untouched.recurrent_state).item(), 0)


if __name__ == "__main__":
    unittest.main()
