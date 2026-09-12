import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.benchmark_vllm_serving import main, run_closed_loop
from benchmarks.mini_http_server import validate_request
from benchmarks.summarize_framework_matrix import compare_outputs


class MatrixTest(unittest.TestCase):
    def test_refill_before_slow_request_finishes(self):
        gate = threading.Event()
        def generate(ids):
            if ids == [0]:
                self.assertTrue(gate.wait(5), "No refill while slow request active")
            elif ids == [2]:
                gate.set()
            return {"token_ids": ids}
        rows, elapsed, events = run_closed_loop([[i] for i in range(5)], 2, generate)
        self.assertEqual([r["request_id"] for r in rows], list(map(str, range(5))))
        self.assertEqual(events[-1]["inflight"], 0)
        self.assertTrue(all(e["inflight"] <= 2 for e in events))
        self.assertGreater(elapsed, 0)

    def test_worker_failure_propagates(self):
        def fail(ids):
            raise ValueError("request failed")
        with self.assertRaisesRegex(ValueError, "request failed"):
            run_closed_loop([[1]], 1, fail)

    def test_fixed_adapter_contract(self):
        data = dict(prompt=[1] * 512, model="target", temperature=0, top_p=1, top_k=-1,
                    max_tokens=128, ignore_eos=True, stream=True, return_token_ids=True)
        self.assertEqual(validate_request(data), [1] * 512)
        with self.assertRaises(ValueError):
            validate_request(dict(data, temperature=1))
        with self.assertRaises(ValueError):
            validate_request(dict(data, prompt=[True] * 512))

    def test_token_comparison(self):
        a = {"workload": {"sha256": "a"}, "requests": [{"request_id": "0", "token_ids": [1, 2]}]}
        b = {"workload": a["workload"], "requests": [{"request_id": "0", "token_ids": [1, 3]}]}
        self.assertEqual(compare_outputs(a, b)["differences"][0]["first_difference"], 1)
        with self.assertRaises(ValueError):
            compare_outputs(a, dict(b, workload={"sha256": "different"}))

    def test_http_client_end_to_end(self):
        """Real socket/SSE path, fake inference. Does not claim GPU performance."""
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"" if self.path == "/metrics" else b'{"version":"fake"}')
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                chunks = [{"choices": [{"token_ids": [9] * 128, "finish_reason": "length"}]},
                          {"usage": {"prompt_tokens": len(data["prompt"]), "completion_tokens": 128}}]
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "result.json"
                tokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: object())
                argv = ["bench", "--mode", "target", "--tokenizer", "fake", "--dataset", "fake",
                        "--output", str(output), "--count", "2", "--metrics-settle-seconds", "0",
                        "--url", f"http://127.0.0.1:{server.server_port}"]
                def prompts(tok, count, length, **kw):
                    return [[i] * length for i in range(count)]
                with patch("sys.argv", argv), patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=tokenizer)}), \
                        patch("benchmarks.benchmark_vllm_serving.prepare_prompts", side_effect=prompts), \
                        contextlib.redirect_stdout(io.StringIO()):
                    main()
                result = json.loads(output.read_text())
                self.assertEqual(result["metrics"]["output_tokens"], 256)
                self.assertEqual(len(result["requests"]), 2)
                self.assertEqual(result["workload"]["unique_prompts"], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
