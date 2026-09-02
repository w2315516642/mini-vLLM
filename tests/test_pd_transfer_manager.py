import time
import unittest

from minivllm.distributed.kv_transfer import (
    BufferSlice,
    RegisteredBuffer,
    TransferBackend,
    TransferEndpoint,
    TransferHandle,
    TransferManager,
    TransferPlan,
    TransferResourceLease,
    TransferSlice,
    TransferStatus,
)


class ControlledBackend(TransferBackend):
    def __init__(self, endpoint, cancellable=True):
        super().__init__(endpoint)
        self.handles = {}
        self.cancellable = cancellable

    def register_tensor(self, name, tensor):
        raise NotImplementedError

    def unregister_buffer(self, name):
        raise NotImplementedError

    def submit(self, plan):
        handle = TransferHandle(plan.transfer_id)
        handle.transition(TransferStatus.RUNNING)
        self.handles[plan.transfer_id] = handle
        return handle

    def poll(self, handle):
        return handle.status

    def abort(self, handle):
        if self.cancellable and not handle.status.is_terminal:
            handle.transition(TransferStatus.CANCELLED)

    def complete(self, transfer_id, status=TransferStatus.COMPLETED):
        self.handles[transfer_id].transition(status)

    def close(self):
        pass


def make_plan(transfer_id):
    source_endpoint = TransferEndpoint("p", "p:1")
    target_endpoint = TransferEndpoint("d", "d:2")
    source = RegisteredBuffer(
        source_endpoint, "kv", 1000, 16, "torch.float32", (4,), "cpu"
    )
    target = RegisteredBuffer(
        target_endpoint, "kv", 2000, 16, "torch.float32", (4,), "cpu"
    )
    return TransferPlan(
        transfer_id,
        "request",
        source_endpoint,
        target_endpoint,
        (
            TransferSlice(
                BufferSlice(source, 0, 16),
                BufferSlice(target, 0, 16),
            ),
        ),
    )


class TransferManagerTest(unittest.TestCase):
    def test_backpressure_and_completion_release_resources(self):
        backend = ControlledBackend(TransferEndpoint("p", "p:1"))
        manager = TransferManager(backend, max_inflight=1, timeout_s=1)
        releases = []
        task = manager.submit(
            make_plan("one"),
            TransferResourceLease(lambda: releases.append("one")),
        )
        with self.assertRaisesRegex(RuntimeError, "backpressure"):
            manager.submit(make_plan("two"))

        backend.complete("one")
        completed = manager.poll()

        self.assertEqual(completed, [task])
        self.assertEqual(releases, ["one"])
        self.assertEqual(manager.num_inflight, 0)
        self.assertEqual(manager.metrics.completed, 1)
        self.assertEqual(manager.metrics.bytes_completed, 16)

    def test_duplicate_submission_is_idempotent(self):
        backend = ControlledBackend(TransferEndpoint("p", "p:1"))
        manager = TransferManager(backend, max_inflight=2, timeout_s=1)
        first = manager.submit(make_plan("same"))
        second = manager.submit(make_plan("same"))
        self.assertIs(first, second)
        self.assertEqual(manager.metrics.submitted, 1)

    def test_noncancellable_io_keeps_lease_until_native_completion(self):
        backend = ControlledBackend(
            TransferEndpoint("p", "p:1"), cancellable=False
        )
        manager = TransferManager(backend, max_inflight=1, timeout_s=1)
        releases = []
        manager.submit(
            make_plan("one"),
            TransferResourceLease(lambda: releases.append("released")),
        )

        self.assertEqual(manager.cancel("one"), TransferStatus.RUNNING)
        self.assertEqual(releases, [])
        backend.complete("one")
        manager.poll()

        self.assertEqual(releases, ["released"])
        self.assertEqual(manager.metrics.cancelled, 1)

    def test_timeout_requests_abort_without_early_release(self):
        backend = ControlledBackend(
            TransferEndpoint("p", "p:1"), cancellable=False
        )
        manager = TransferManager(backend, max_inflight=1, timeout_s=0.01)
        releases = []
        task = manager.submit(
            make_plan("slow"),
            TransferResourceLease(lambda: releases.append("released")),
        )
        time.sleep(0.02)

        manager.poll()

        self.assertTrue(task.timeout_requested)
        self.assertFalse(task.finalized)
        self.assertEqual(releases, [])
        backend.complete("slow")
        manager.poll()
        self.assertEqual(releases, ["released"])
        self.assertEqual(manager.metrics.timed_out, 1)


if __name__ == "__main__":
    unittest.main()
