import json
import socket
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer

from benchmarks.mini_http_server import StreamingHandler
from benchmarks.run_framework_matrix import check_port_available


class TransportTest(unittest.TestCase):
    def test_first_event_arrives_before_generation_finishes(self):
        import requests

        received_first = threading.Event()
        server_saw_ack = threading.Event()

        class Handler(StreamingHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.start_stream()
                self.send_event({"choices": [{"token_ids": [11]}]})
                # Completion is withheld until the client consumes the first
                # event. A read-until-EOF client fails this handshake.
                if received_first.wait(3):
                    server_saw_ack.set()
                self.send_event({"choices": [{"token_ids": [12], "finish_reason": "length"}]})
                self.send_event(None)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with requests.get(f"http://127.0.0.1:{server.server_port}", stream=True, timeout=5) as response:
                response.raise_for_status()
                self.assertEqual(response.headers["Transfer-Encoding"], "chunked")
                lines = response.iter_lines(chunk_size=None, decode_unicode=True)
                first = next(lines)
                self.assertEqual(json.loads(first[5:])["choices"][0]["token_ids"], [11])
                received_first.set()
                remaining = list(lines)
                self.assertIn("data: [DONE]", remaining)
                self.assertTrue(server_saw_ack.is_set(), "First event was buffered until stream completion")
        finally:
            received_first.set()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_live_listener_is_not_reusable(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with self.assertRaisesRegex(OSError, "live listeners"):
                check_port_available(listener.getsockname()[1])

    @unittest.skipUnless(sys.platform.startswith("linux"), "AutoDL/Linux TIME_WAIT semantics")
    def test_server_time_wait_does_not_block_next_cell(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                connection, _ = listener.accept()
                # Server actively closes first, leaving its endpoint in TIME_WAIT.
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
                client.shutdown(socket.SHUT_WR)
                self.assertEqual(connection.recv(1), b"")
                connection.close()
        check_port_available(port)
        with socket.socket() as next_server:
            next_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            next_server.bind(("127.0.0.1", port))
            next_server.listen(1)


if __name__ == "__main__":
    unittest.main()
