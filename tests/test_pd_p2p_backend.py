import socket
import unittest

import torch

from minivllm.distributed.kv_transfer import (
    BufferSlice,
    P2PTransferBackend,
    RegisteredBuffer,
    TransferEndpoint,
    TransferPlan,
    TransferSlice,
    TransferStatus,
)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as value:
        value.bind(("127.0.0.1", 0))
        return value.getsockname()[1]


class P2PTransferBackendTest(unittest.TestCase):
    def setUp(self):
        self.source_endpoint = TransferEndpoint(
            "prefill-0", f"127.0.0.1:{free_port()}"
        )
        self.target_endpoint = TransferEndpoint(
            "decode-0", f"127.0.0.1:{free_port()}"
        )
        self.source = P2PTransferBackend(self.source_endpoint, timeout_s=2)
        self.target = P2PTransferBackend(self.target_endpoint, timeout_s=2)

    def tearDown(self):
        for backend in (self.source, self.target):
            for handle in backend._handles.values():
                if not handle.status.is_terminal:
                    backend.wait(handle, timeout_s=3)
            backend.close()

    def test_batch_pushes_multiple_partial_tensor_slices(self):
        source_key = torch.arange(16, dtype=torch.float32)
        source_value = torch.arange(100, 116, dtype=torch.float32)
        target_key = torch.full((16,), -1.0)
        target_value = torch.full((16,), -2.0)
        source_key_buffer = self.source.register_tensor("key", source_key)
        source_value_buffer = self.source.register_tensor(
            "value", source_value
        )
        target_key_buffer = self.target.register_tensor("key", target_key)
        target_value_buffer = self.target.register_tensor(
            "value", target_value
        )
        plan = TransferPlan(
            transfer_id="request/0",
            request_id="request",
            source_endpoint=self.source_endpoint,
            target_endpoint=self.target_endpoint,
            slices=(
                TransferSlice(
                    BufferSlice(source_key_buffer, 4 * 4, 5 * 4),
                    BufferSlice(target_key_buffer, 2 * 4, 5 * 4),
                ),
                TransferSlice(
                    BufferSlice(source_value_buffer, 6 * 4, 4 * 4),
                    BufferSlice(target_value_buffer, 10 * 4, 4 * 4),
                ),
            ),
        )

        handle = self.source.submit(plan)
        status = self.source.wait(handle, timeout_s=3)

        self.assertEqual(status, TransferStatus.COMPLETED)
        torch.testing.assert_close(target_key[2:7], source_key[4:9])
        torch.testing.assert_close(target_value[10:14], source_value[6:10])
        self.assertTrue(torch.all(target_key[:2] == -1))
        self.assertTrue(torch.all(target_value[:10] == -2))

    def test_remote_rejects_unregistered_target(self):
        source_tensor = torch.arange(4, dtype=torch.int32)
        source = self.source.register_tensor("source", source_tensor)
        missing = RegisteredBuffer(
            endpoint=self.target_endpoint,
            name="missing",
            address=123_456,
            nbytes=source.nbytes,
            dtype=source.dtype,
            shape=source.shape,
            device=source.device,
        )
        handle = self.source.submit(
            TransferPlan(
                "request/1",
                "request",
                self.source_endpoint,
                self.target_endpoint,
                (
                    TransferSlice(
                        BufferSlice(source, 0, source.nbytes),
                        BufferSlice(missing, 0, missing.nbytes),
                    ),
                ),
            )
        )

        status = self.source.wait(handle, timeout_s=3)

        self.assertEqual(status, TransferStatus.FAILED)
        self.assertIn("not registered", handle.error)

    def test_source_descriptor_must_be_locally_registered(self):
        tensor = torch.zeros(4)
        actual = self.source.register_tensor("actual", tensor)
        target = self.target.register_tensor("target", torch.zeros(4))
        forged = RegisteredBuffer(
            endpoint=actual.endpoint,
            name=actual.name,
            address=actual.address + 4,
            nbytes=actual.nbytes,
            dtype=actual.dtype,
            shape=actual.shape,
            device=actual.device,
        )
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.source.submit(
                TransferPlan(
                    "request/2",
                    "request",
                    self.source_endpoint,
                    self.target_endpoint,
                    (
                        TransferSlice(
                            BufferSlice(forged, 0, forged.nbytes),
                            BufferSlice(target, 0, target.nbytes),
                        ),
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
