import socket
import threading
import unittest

from minivllm.engine.pd_rpc import (
    PDControlServer,
    RemoteEngineClient,
    RemoteEngineError,
)


def free_address():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as value:
        value.bind(("127.0.0.1", 0))
        host, port = value.getsockname()
        return f"{host}:{port}"


class FakeEngine:
    def __init__(self):
        self.count = 2

    def get_num_unfinished_requests(self):
        return self.count

    def has_unfinished_requests(self):
        return self.count > 0

    def step(self):
        self.count -= 1
        return []

    def abort_decode_handoff(self, request_id):
        raise ValueError(f"cannot abort {request_id}")


class PDRPCTest(unittest.TestCase):
    def setUp(self):
        self.address = free_address()
        self.server = PDControlServer(
            FakeEngine(), self.address, authkey=b"test-secret"
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()
        self.client = RemoteEngineClient(
            self.address, authkey=b"test-secret"
        )

    def tearDown(self):
        self.client.close()
        self.server.close()

    def test_remote_engine_proxy_round_trip(self):
        self.assertEqual(self.client.get_num_unfinished_requests(), 2)
        self.assertEqual(self.client.step(), [])
        self.assertEqual(self.client.get_num_unfinished_requests(), 1)

    def test_remote_exception_keeps_type_and_message(self):
        with self.assertRaisesRegex(
            RemoteEngineError, "ValueError: cannot abort request-5"
        ):
            self.client.abort_decode_handoff("request-5")

    def test_client_rejects_non_pd_methods(self):
        with self.assertRaisesRegex(ValueError, "not allowed"):
            self.client.call("__getattribute__", "workers")


if __name__ == "__main__":
    unittest.main()
