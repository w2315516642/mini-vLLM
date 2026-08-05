import importlib.util
import sys
import types
import unittest
from enum import Enum, auto
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_prefix_cache_modules():
    """Load the CPU-only cache modules without importing CUDA extensions."""
    minivllm = types.ModuleType("minivllm")
    minivllm.__path__ = [str(ROOT / "minivllm")]
    sys.modules["minivllm"] = minivllm

    utils = types.ModuleType("minivllm.utils")
    utils.__path__ = [str(ROOT / "minivllm" / "utils")]
    sys.modules["minivllm.utils"] = utils

    kv_cache = types.ModuleType("minivllm.kv_cache")
    kv_cache.__path__ = [str(ROOT / "minivllm" / "kv_cache")]
    sys.modules["minivllm.kv_cache"] = kv_cache

    hasher = _load_module(
        "minivllm.utils.hasher", "minivllm/utils/hasher.py")

    class Device(Enum):
        CPU = auto()
        GPU = auto()

    utils.Device = Device
    utils.BlockHash = hasher.BlockHash
    utils.BlockHasher = hasher.BlockHasher

    sampling_params = types.ModuleType("minivllm.sampling_params")
    sampling_params.SamplingParams = object
    sys.modules["minivllm.sampling_params"] = sampling_params

    block = _load_module(
        "minivllm.kv_cache.block", "minivllm/kv_cache/block.py")
    sequence = _load_module("minivllm.sequence", "minivllm/sequence.py")
    block_manager = _load_module(
        "minivllm.kv_cache.block_manager",
        "minivllm/kv_cache/block_manager.py",
    )
    return hasher, sequence, block_manager


hasher, sequence, block_manager = _load_prefix_cache_modules()
hasher.init_none_hash(hasher.sha256)


def make_seq(seq_id, token_ids, block_size=4):
    return sequence.Sequence(
        seq_id=seq_id,
        prompt="",
        prompt_token_ids=token_ids,
        block_size=block_size,
        block_hasher=hasher.get_seq_block_hasher(hasher.sha256),
    )


def make_group(request_id, seqs):
    return sequence.SequenceGroup(request_id, seqs, None, 0.0)


class PrefixCacheBlockTest(unittest.TestCase):

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
