import unittest

import torch

from prefix_cache_test_utils import load_worker_module, sequence


worker_module = load_worker_module()


def make_metadata(
    request_id,
    seq_id,
    token_ids,
    block_table,
    is_prompt,
    num_computed_tokens,
    num_scheduled_tokens,
    do_sample=True,
):
    seq_data = sequence.SequenceData(token_ids)
    if not is_prompt:
        seq_data.append_token_id(99, 0.0)
    return sequence.SequenceGroupMetadata(
        request_id=request_id,
        is_prompt=is_prompt,
        seq_data={seq_id: seq_data},
        sampling_params=None,
        block_tables={seq_id: block_table},
        num_computed_tokens={seq_id: num_computed_tokens},
        num_scheduled_tokens={seq_id: num_scheduled_tokens},
        do_sample=do_sample,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA tensors")
class PrefixCacheWorkerTest(unittest.TestCase):

    def test_prepare_inputs_separates_fresh_cached_and_decode_tokens(self):
        fresh = make_metadata(
            "fresh", 0, list(range(8)), [10, 11], True, 0, 8)
        cached = make_metadata(
            "cached", 1, list(range(8)), [20, 21], True, 4, 4)
        cached_short = make_metadata(
            "cached-short", 3, list(range(6)), [40, 41], True, 4, 2)
        decode = make_metadata(
            "decode", 2, list(range(4)), [30, 31], False, 4, 1)

        worker = object.__new__(worker_module.Worker)
        worker.block_size = 4
        tokens, positions, metadata = worker._prepare_inputs(
            [cached, decode, fresh, cached_short])

        self.assertEqual(
            tokens[:15].cpu().tolist(),
            list(range(8)) + list(range(4, 8)) + [4, 5, 99],
        )
        self.assertEqual(
            positions[:15].cpu().tolist(),
            list(range(8)) + list(range(4, 8)) + [4, 5, 4],
        )
        self.assertEqual(
            metadata.slot_mapping.cpu().tolist(),
            list(range(40, 48)) + list(range(84, 88)) + [164, 165, 124],
        )

        self.assertEqual(metadata.prompt_lens, [8, 4, 2])
        self.assertEqual(metadata.num_fresh_prompt_tokens, 8)
        self.assertEqual(metadata.num_cached_prompt_tokens, 6)
        self.assertEqual(
            metadata.cached_prompt_cu_seqlens.cpu().tolist(), [0, 4, 6])
        self.assertEqual(
            metadata.cached_prompt_context_lens.cpu().tolist(), [8, 6])
        self.assertEqual(
            metadata.cached_prompt_block_tables.cpu().tolist(),
            [[20, 21], [40, 41]],
        )
        self.assertEqual(metadata.block_tables.cpu().tolist(), [[30, 31]])
        self.assertEqual(
            [seq_ids for seq_ids, _ in metadata.seq_groups],
            [[0], [1], [3], [2]],
        )

    def test_intermediate_prompt_chunk_is_excluded_from_sampling(self):
        chunk = make_metadata(
            "chunk", 0, list(range(8)), [10, 11], True, 0, 4,
            do_sample=False,
        )
        worker = object.__new__(worker_module.Worker)
        worker.block_size = 4

        tokens, _, metadata = worker._prepare_inputs([chunk])

        self.assertEqual(tokens[:4].cpu().tolist(), [0, 1, 2, 3])
        self.assertEqual(metadata.prompt_lens, [4])
        self.assertEqual(metadata.prompt_sample_indices, [])
        self.assertEqual(metadata.seq_groups, [])


if __name__ == "__main__":
    unittest.main()
