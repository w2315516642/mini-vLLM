import unittest

import torch

from minivllm.configs import KVTransferBackend, PDConfig, PDRole
from minivllm.distributed.kv_transfer import (
    BufferSlice,
    InMemoryTransferBackend,
    InMemoryTransferRegistry,
    TransferEndpoint,
    TransferPlan,
    TransferSlice,
    TransferStatus,
)


class PDConfigTest(unittest.TestCase):
    def test_unified_is_the_backward_compatible_default(self) -> None:
        config = PDConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.role, PDRole.UNIFIED)
        self.assertEqual(config.backend, KVTransferBackend.NONE)

    def test_pd_role_requires_complete_peer_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "endpoint_id"):
            PDConfig(
                role=PDRole.PREFILL,
                backend=KVTransferBackend.MEMORY,
            )

    def test_string_enums_are_normalized(self) -> None:
        config = PDConfig(
            role="decode",
            backend="memory",
            endpoint_id="decode-0",
            hostname="127.0.0.1:12001",
            peer_endpoint_id="prefill-0",
            peer_hostname="127.0.0.1:12000",
        )
        self.assertEqual(config.role, PDRole.DECODE)
        self.assertEqual(config.backend, KVTransferBackend.MEMORY)


class TransferContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_endpoint = TransferEndpoint(
            "prefill-0", "127.0.0.1:12000"
        )
        self.target_endpoint = TransferEndpoint(
            "decode-0", "127.0.0.1:12001"
        )
        self.registry = InMemoryTransferRegistry()
        self.source_backend = InMemoryTransferBackend(
            self.source_endpoint, self.registry
        )
        self.target_backend = InMemoryTransferBackend(
            self.target_endpoint, self.registry
        )

    def tearDown(self) -> None:
        self.source_backend.close()
        self.target_backend.close()

    def test_plan_round_trip_and_partial_copy(self) -> None:
        source = torch.arange(16, dtype=torch.int32)
        target = torch.full((16,), -1, dtype=torch.int32)
        source_buffer = self.source_backend.register_tensor("kv", source)
        target_buffer = self.target_backend.register_tensor("kv", target)
        item = TransferSlice(
            BufferSlice(source_buffer, offset=4 * 4, length=6 * 4),
            BufferSlice(target_buffer, offset=2 * 4, length=6 * 4),
        )
        plan = TransferPlan(
            transfer_id="request-7/0",
            request_id="request-7",
            source_endpoint=self.source_endpoint,
            target_endpoint=self.target_endpoint,
            slices=(item,),
            metadata={"num_tokens": 6},
        )

        restored = TransferPlan.from_dict(plan.to_dict())
        handle = self.source_backend.submit(restored)

        self.assertEqual(handle.status, TransferStatus.COMPLETED)
        self.assertEqual(restored.total_bytes, 24)
        torch.testing.assert_close(target[2:8], source[4:10])
        torch.testing.assert_close(
            target[:2], torch.full((2,), -1, dtype=torch.int32)
        )

    def test_plan_rejects_endpoint_mismatch(self) -> None:
        source = self.source_backend.register_tensor(
            "source", torch.zeros(4)
        )
        target = self.target_backend.register_tensor(
            "target", torch.zeros(4)
        )
        item = TransferSlice(
            BufferSlice(source, 0, source.nbytes),
            BufferSlice(target, 0, target.nbytes),
        )
        with self.assertRaisesRegex(ValueError, "source slice"):
            TransferPlan(
                transfer_id="bad",
                request_id="request",
                source_endpoint=self.target_endpoint,
                target_endpoint=self.source_endpoint,
                slices=(item,),
            )

    def test_non_contiguous_tensor_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.source_backend.register_tensor(
                "bad", torch.zeros(2, 3).transpose(0, 1)
            )


if __name__ == "__main__":
    unittest.main()
