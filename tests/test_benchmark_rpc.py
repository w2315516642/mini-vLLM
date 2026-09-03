import socket
import threading
import unittest
from unittest.mock import Mock

from benchmarks.benchmark_utils import WorkItem
from benchmarks.serving_runner import run_pd, run_unified
from minivllm.engine.pd_rpc import PDControlServer, RemoteEngineClient
from tests.test_benchmark_runner import Engine, fake_bridge


class BenchmarkRPCTest(unittest.TestCase):
    def setUp(self):
        self.servers = []
        self.clients = []

    def client(self, engine):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            address = f"127.0.0.1:{sock.getsockname()[1]}"
        server = PDControlServer(engine, address, b"benchmark-test")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        client = RemoteEngineClient(address, b"benchmark-test", timeout_s=5.)
        self.servers.append(server)
        self.clients.append(client)
        return client

    def tearDown(self):
        for client in self.clients:
            client.close()
        for server in self.servers:
            server.close()

    def test_unified_multi_request_rpc(self):
        clients = [self.client(Engine()), self.client(Engine())]
        traces, _ = run_unified(clients, [WorkItem(str(i), [1], 4) for i in range(4)], int)
        self.assertEqual([t.output_tokens for t in traces], [4] * 4)

    def test_pd_multi_request_rpc(self):
        p, d = Engine(prefill=True), Engine()
        traces, _ = run_pd(self.client(p), self.client(d),
                           [WorkItem(str(i), [1], 4) for i in range(4)], int, fake_bridge(p, d))
        self.assertEqual([t.output_tokens for t in traces], [4] * 4)
        self.assertTrue(all(t.handoff_s is not None for t in traces))

    def test_runtime_statistics_are_read_over_rpc(self):
        engine = Engine()
        engine.get_runtime_stats = lambda: {"speculative": {"verification_rounds": 4}}
        self.assertEqual(self.client(engine).get_runtime_stats()["speculative"]["verification_rounds"], 4)

    def test_timeout_closes_connection_instead_of_reading_a_late_response(self):
        client = object.__new__(RemoteEngineClient)
        client.connection = Mock()
        client.connection.poll.return_value = False
        client._lock = threading.RLock()
        client.timeout_s = .1
        with self.assertRaisesRegex(TimeoutError, "RPC step exceeded"):
            client.step()
        client.connection.close.assert_called_once()
        client.connection.recv.assert_not_called()


if __name__ == "__main__":
    unittest.main()
