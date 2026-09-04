import unittest
from unittest.mock import patch
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


def _state_spec():
    config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=3,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=3,
    )
    return GatedDeltaNetStateSpec.from_text_config(config)


def _make_cache(max_num_seqs=3):
    return HybridCache(
        layer_types=(
            LINEAR_ATTENTION,
            FULL_ATTENTION,
            LINEAR_ATTENTION,
        ),
        full_attention_caches={
            1: (torch.empty(2, 3), torch.empty(2, 4))
        },
        state_spec=_state_spec(),
        max_num_seqs=max_num_seqs,
        device="cpu",
    )


def _filled_state(spec, batch_size, value):
    return GatedDeltaNetState(
        conv_state=torch.full(
            (batch_size, *spec.conv_state_shape), value, dtype=torch.float32
        ),
        recurrent_state=torch.full(
            (batch_size, *spec.recurrent_state_shape),
            value,
            dtype=torch.float32,
        ),
    )


class RequestStateSlotAllocatorTest(unittest.TestCase):
    def test_slot_is_stable_bounded_and_reusable(self):
        allocator = RequestStateSlotAllocator(capacity=2)

        self.assertEqual(allocator.acquire(10), (0, True))
        self.assertEqual(allocator.acquire(20), (1, True))
        self.assertEqual(allocator.acquire(10), (0, False))
        with self.assertRaisesRegex(ValueError, "free state slots"):
            allocator.acquire(30)

        self.assertEqual(allocator.release(10), 0)
        self.assertEqual(allocator.acquire(30), (0, True))

    def test_unknown_sequence_and_reset(self):
        allocator = RequestStateSlotAllocator(capacity=2)
        with self.assertRaisesRegex(ValueError, "not active"):
            allocator.lookup(7)

        allocator.acquire(10)
        allocator.acquire(20)
        allocator.reset()
        self.assertEqual(allocator.acquire(30), (0, True))


class HybridCacheTest(unittest.TestCase):
    def test_internal_batch_io_does_not_revalidate_cuda_values(self):
        cache = _make_cache()
        slots = cache.acquire([10, 20])
        with patch.object(cache, "_normalize_slot_ids", side_effect=AssertionError):
            state = cache._read_state(0, slots)
            state.conv_state.fill_(3)
            cache._write_state(0, slots, state)
        torch.testing.assert_close(cache.read_state(0, slots).conv_state,
                                   torch.full_like(state.conv_state, 3))

    def test_snapshot_and_pool_remain_independent_in_both_directions(self):
        cache = _make_cache()
        slots = cache.acquire([10])
        cache.write_state(0, slots, _filled_state(cache.state_spec, 1, 2))
        snapshot = cache.snapshot([10])
        cache.write_state(0, slots, _filled_state(cache.state_spec, 1, 5))
        self.assertTrue(torch.all(snapshot.layer_states[0].conv_state == 2))
        snapshot.layer_states[0].recurrent_state.fill_(9)
        self.assertTrue(torch.all(cache.read_state(0, slots).recurrent_state == 5))

    def test_state_shapes_and_global_kv_layer_index(self):
        cache = _make_cache(max_num_seqs=2)
        slots = cache.acquire([10, 20])
        state = cache.read_state(0, slots)

        self.assertEqual(cache.state_spec.conv_state_shape, (20, 3))
        self.assertEqual(state.conv_state.shape, (2, 20, 3))
        self.assertEqual(state.recurrent_state.shape, (2, 4, 3, 2))
        self.assertEqual(state.conv_state.dtype, torch.float32)
        self.assertEqual(cache.get_kv_cache(1)[0].shape, (2, 3))
        with self.assertRaisesRegex(ValueError, "linear_attention"):
            cache.get_kv_cache(0)

    def test_reordered_batch_keeps_sequence_state(self):
        cache = _make_cache()
        slots = cache.acquire([10, 20])
        state = _filled_state(cache.state_spec, 2, 1.0)
        state.conv_state[1].fill_(7.0)
        state.recurrent_state[1].fill_(7.0)
        cache.write_state(0, slots, state)

        reordered = cache.read_state(0, cache.acquire([20, 10]))
        torch.testing.assert_close(
            reordered.conv_state[0], torch.full_like(reordered.conv_state[0], 7.0)
        )
        torch.testing.assert_close(
            reordered.conv_state[1], torch.ones_like(reordered.conv_state[1])
        )

    def test_release_clears_state_before_reuse(self):
        cache = _make_cache(max_num_seqs=1)
        old_slot = cache.acquire([10])
        cache.write_state(
            0, old_slot, _filled_state(cache.state_spec, 1, 9.0)
        )

        cache.release([10])
        new_slot = cache.acquire([20])
        fresh = cache.read_state(0, new_slot)

        self.assertEqual(old_slot.tolist(), new_slot.tolist())
        self.assertEqual(torch.count_nonzero(fresh.conv_state).item(), 0)
        self.assertEqual(torch.count_nonzero(fresh.recurrent_state).item(), 0)

    def test_fork_copies_all_linear_layers_without_aliasing(self):
        cache = _make_cache()
        parent_slot = cache.acquire([10])
        cache.write_state(
            0, parent_slot, _filled_state(cache.state_spec, 1, 2.0)
        )
        cache.write_state(
            2, parent_slot, _filled_state(cache.state_spec, 1, 5.0)
        )

        child_slot = cache.fork(10, 20)
        child_state = cache.read_state(2, [child_slot])
        torch.testing.assert_close(
            child_state.recurrent_state,
            torch.full_like(child_state.recurrent_state, 5.0),
        )

        cache.write_state(
            0, [child_slot], _filled_state(cache.state_spec, 1, 11.0)
        )
        parent_after = cache.read_state(0, parent_slot)
        torch.testing.assert_close(
            parent_after.conv_state,
            torch.full_like(parent_after.conv_state, 2.0),
        )

    def test_active_beam_state_can_be_overwritten_and_released(self):
        cache = _make_cache(max_num_seqs=2)
        parent_slot, child_slot = cache.acquire([10, 20]).tolist()
        cache.write_state(
            0, [parent_slot], _filled_state(cache.state_spec, 1, 3.0)
        )
        cache.write_state(
            0, [child_slot], _filled_state(cache.state_spec, 1, 8.0)
        )

        cache.copy(10, 20)
        copied = cache.read_state(0, [child_slot])
        torch.testing.assert_close(
            copied.conv_state, torch.full_like(copied.conv_state, 3.0)
        )

        cache.release_existing([20, 999])
        self.assertEqual(cache.num_active_slots, 1)
        self.assertEqual(cache.num_free_slots, 1)

    def test_initial_best_of_copy_allocates_the_child_slot(self):
        cache = _make_cache(max_num_seqs=2)
        parent_slot = cache.acquire([10])
        cache.write_state(
            0, parent_slot, _filled_state(cache.state_spec, 1, 6.0)
        )

        cache.copy(10, 20)

        child_slot = cache.acquire([20])
        child_state = cache.read_state(0, child_slot)
        torch.testing.assert_close(
            child_state.recurrent_state,
            torch.full_like(child_state.recurrent_state, 6.0),
        )

    def test_snapshot_restores_only_rejected_sequences(self):
        cache = _make_cache(max_num_seqs=2)
        slots = cache.acquire([10, 20])
        original = _filled_state(cache.state_spec, 2, 2.0)
        original.conv_state[1].fill_(4.0)
        original.recurrent_state[1].fill_(4.0)
        cache.write_state(0, slots, original)
        snapshot = cache.snapshot([10, 20])

        changed = _filled_state(cache.state_spec, 2, 9.0)
        cache.write_state(0, slots, changed)
        cache.restore(snapshot, [20])

        restored = cache.read_state(0, slots)
        torch.testing.assert_close(
            restored.conv_state[0],
            torch.full_like(restored.conv_state[0], 9.0),
        )
        torch.testing.assert_close(
            restored.conv_state[1],
            torch.full_like(restored.conv_state[1], 4.0),
        )

    def test_write_validation_and_reset(self):
        cache = _make_cache(max_num_seqs=2)
        slots = cache.acquire([10])
        valid = _filled_state(cache.state_spec, 1, 1.0)
        with self.assertRaisesRegex(ValueError, "conv_state"):
            cache.write_state(
                0,
                slots,
                GatedDeltaNetState(
                    valid.conv_state[:, :-1], valid.recurrent_state
                ),
            )
        with self.assertRaisesRegex(ValueError, "float32"):
            cache.write_state(
                0,
                slots,
                GatedDeltaNetState(
                    valid.conv_state.half(), valid.recurrent_state.half()
                ),
            )

        cache.write_state(0, slots, valid)
        cache.reset()
        fresh_slot = cache.acquire([20])
        self.assertEqual(fresh_slot.tolist(), [0])
        self.assertEqual(
            torch.count_nonzero(cache.read_state(0, fresh_slot).conv_state).item(),
            0,
        )

    def test_layout_and_slot_validation(self):
        with self.assertRaisesRegex(ValueError, r"expected \[1\]"):
            HybridCache(
                layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
                full_attention_caches={},
                state_spec=_state_spec(),
                max_num_seqs=1,
                device="cpu",
            )

        cache = _make_cache(max_num_seqs=2)
        with self.assertRaisesRegex(ValueError, "duplicates"):
            cache.read_state(0, [0, 0])
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            cache.read_state(0, [2])


if __name__ == "__main__":
    unittest.main()
