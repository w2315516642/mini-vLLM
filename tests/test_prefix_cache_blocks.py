import unittest

from prefix_cache_test_utils import (
    block_manager,
    hasher,
    make_group,
    make_seq,
)


class PrefixCacheBlockTest(unittest.TestCase):

    def test_disabled_cache_does_not_hold_blocks(self):
        manager = block_manager.BlockSpaceManager(
            4, 2, 1, watermark=0, enable_prefix_caching=False)
        seq = make_seq(0, list(range(4)))
        manager.allocate(make_group("disabled", [seq]))
        seq.num_computed_tokens = seq.get_len()
        manager.cache_blocks(seq)
        manager.free(seq)

        self.assertEqual(len(manager.hash_manager), 0)
        self.assertEqual(manager.get_num_free_gpu_blocks(), 2)

    def test_hashes_form_a_parent_chain(self):
        seq = make_seq(0, list(range(8)))

        first = hasher.hash_block_tokens(
            hasher.sha256, None, list(range(4)))
        second = hasher.hash_block_tokens(
            hasher.sha256, first, list(range(4, 8)))

        self.assertEqual(seq.block_hashes, [first, second])

    def test_new_blocks_have_one_reference_per_sequence(self):
        manager = block_manager.BlockSpaceManager(4, 4, 2, watermark=0)
        seqs = [make_seq(0, list(range(8))), make_seq(1, list(range(8)))]
        manager.allocate(make_group("group", seqs))

        blocks = manager.block_tables[seqs[0].seq_id]
        self.assertTrue(all(block.ref_count == 2 for block in blocks))

        for seq in seqs:
            manager.free(seq)
        self.assertEqual(manager.get_num_free_gpu_blocks(), 4)

    def test_cached_blocks_survive_sequence_free_and_are_reused(self):
        manager = block_manager.BlockSpaceManager(4, 4, 2, watermark=0)
        first_seq = make_seq(0, list(range(8)))
        manager.allocate(make_group("first", [first_seq]))
        first_seq.num_computed_tokens = first_seq.get_len()
        manager.cache_blocks(first_seq)

        cached_blocks = manager.block_tables[first_seq.seq_id].copy()
        manager.free(first_seq)
        self.assertTrue(all(block.ref_count == 1 for block in cached_blocks))

        second_seq = make_seq(1, list(range(8)))
        manager.allocate(make_group("second", [second_seq]))
        self.assertEqual(second_seq.num_computed_tokens, 4)
        self.assertEqual(second_seq.num_cached_blocks, 1)
        self.assertIs(
            manager.block_tables[second_seq.seq_id][0], cached_blocks[0])

    def test_lru_evicts_cache_only_block_when_gpu_is_full(self):
        manager = block_manager.BlockSpaceManager(4, 2, 1, watermark=0)
        cached_seq = make_seq(0, list(range(8)))
        manager.allocate(make_group("cached", [cached_seq]))
        cached_seq.num_computed_tokens = cached_seq.get_len()
        manager.cache_blocks(cached_seq)
        oldest_hash = cached_seq.block_hashes[0]
        manager.free(cached_seq)

        new_seq = make_seq(1, [10, 11, 12, 13])
        self.assertTrue(manager.can_allocate(make_group("new", [new_seq])))
        manager.allocate(make_group("new", [new_seq]))

        self.assertNotIn(oldest_hash, manager.hash_manager.hash_to_block)
        self.assertEqual(len(manager.hash_manager), 1)

        manager.free(new_seq)
        manager.reset()
        self.assertEqual(manager.get_num_free_gpu_blocks(), 2)


if __name__ == "__main__":
    unittest.main()
