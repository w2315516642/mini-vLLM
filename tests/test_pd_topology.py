import unittest

import torch

from minivllm.distributed.kv_transfer import (
    InMemoryTransferBackend,
    InMemoryTransferRegistry,
    PDTransferTopology,
    TransferEndpoint,
    register_cache_layout,
)


def make_layout(registry, role, rank, port, block_size=2):
    backend = InMemoryTransferBackend(
        TransferEndpoint(
            f"{role}/rank-{rank}", f"127.0.0.1:{port}", rank=rank
        ),
        registry,
    )
    layout = register_cache_layout(
        backend,
        block_size,
        {0: (torch.zeros(2, 4), torch.zeros(2, 4))},
    )
    return backend, layout


class PDTransferTopologyTest(unittest.TestCase):
    def test_pairs_equal_tp_by_rank_not_input_order(self):
        registry = InMemoryTransferRegistry()
        p0, p0_layout = make_layout(registry, "p", 0, 12000)
        p1, p1_layout = make_layout(registry, "p", 1, 12001)
        d0, d0_layout = make_layout(registry, "d", 0, 13000)
        d1, d1_layout = make_layout(registry, "d", 1, 13001)

        topology = PDTransferTopology.build(
            [p1_layout, p0_layout], [d0_layout, d1_layout]
        )

        self.assertEqual([pair.rank for pair in topology.pairs], [0, 1])
        self.assertEqual(topology.pairs[0].source.endpoint.rank, 0)
        self.assertEqual(topology.pairs[1].target.endpoint.rank, 1)
        for backend in (p0, p1, d0, d1):
            backend.close()

    def test_rejects_heterogeneous_tp(self):
        registry = InMemoryTransferRegistry()
        p0, p0_layout = make_layout(registry, "p", 0, 12000)
        p1, p1_layout = make_layout(registry, "p", 1, 12001)
        d0, d0_layout = make_layout(registry, "d", 0, 13000)
        with self.assertRaisesRegex(ValueError, "equal tensor parallel"):
            PDTransferTopology.build([p0_layout, p1_layout], [d0_layout])
        for backend in (p0, p1, d0):
            backend.close()

    def test_rejects_noncontiguous_ranks(self):
        registry = InMemoryTransferRegistry()
        p0, p0_layout = make_layout(registry, "p", 0, 12000)
        p2, p2_layout = make_layout(registry, "p", 2, 12002)
        d0, d0_layout = make_layout(registry, "d", 0, 13000)
        d1, d1_layout = make_layout(registry, "d", 1, 13001)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            PDTransferTopology.build(
                [p0_layout, p2_layout], [d0_layout, d1_layout]
            )
        for backend in (p0, p2, d0, d1):
            backend.close()


if __name__ == "__main__":
    unittest.main()
