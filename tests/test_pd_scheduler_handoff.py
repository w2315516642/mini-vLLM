import unittest

from minivllm.configs import CacheConfig, SchedulerConfig
from minivllm.kv_cache.scheduler import Scheduler
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus


def make_scheduler():
    cache_config = CacheConfig(
        block_size=2,
        gpu_memory_utilization=0.9,
        swap_space=0,
    )
    cache_config.num_gpu_blocks = 12
    cache_config.num_cpu_blocks = 0
    return Scheduler(
        SchedulerConfig(max_num_batched_tokens=16, max_num_seqs=4),
        cache_config,
        log_stats=False,
    )


def make_group(request_id="request"):
    sequence = Sequence(7, "hello", [1, 2, 3], block_size=2)
    sequence.append_token_id(4, {4: -0.2})
    return SequenceGroup(
        request_id,
        [sequence],
        SamplingParams(temperature=0, max_tokens=4),
        arrival_time=0.0,
    )


class PDSchedulerHandoffTest(unittest.TestCase):
    def test_decode_reservation_is_not_runnable_before_activation(self):
        scheduler = make_scheduler()
        group = make_group()

        block_tables = scheduler.reserve_transferred_seq_group(group, 3)

        self.assertEqual(len(block_tables[7]), 2)
        self.assertEqual(group.seqs[0].num_computed_tokens, 3)
        self.assertEqual(group.seqs[0].status, SequenceStatus.WAITING)
        metadata, outputs = scheduler.scheduler()
        self.assertEqual(metadata, [])
        self.assertTrue(outputs.is_empty())
        self.assertTrue(scheduler.has_unfinished_seqs())

        scheduler.activate_transferred_seq_group("request")
        metadata, outputs = scheduler.scheduler()
        self.assertEqual(len(metadata), 1)
        self.assertFalse(metadata[0].is_prompt)
        self.assertEqual(outputs.num_scheduled_tokens, {7: 1})

    def test_prefill_blocks_remain_pinned_until_release(self):
        scheduler = make_scheduler()
        group = make_group()
        scheduler.block_manager.allocate(group)
        group.seqs[0].num_computed_tokens = 3
        group.seqs[0].status = SequenceStatus.RUNNING
        scheduler.running.append(group)
        before = scheduler.block_manager.get_num_free_gpu_blocks()

        scheduler.seal_prefilled_seq_group(group)

        self.assertNotIn(group, scheduler.running)
        self.assertEqual(
            scheduler.block_manager.get_num_free_gpu_blocks(), before
        )
        scheduler.release_sealed_seq_group("request")
        self.assertEqual(
            scheduler.block_manager.get_num_free_gpu_blocks(), before + 2
        )


if __name__ == "__main__":
    unittest.main()
