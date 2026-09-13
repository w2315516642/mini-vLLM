"""Loopback-only benchmark adapter. The main thread exclusively owns the engine.

This is deliberately not a general OpenAI API implementation. It accepts only
the fixed token-ID workload used by benchmark_vllm_serving.
"""
import argparse
import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from benchmarks.benchmark_vllm_serving import COUNTERS


class StreamingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def start_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def send_event(self, data):
        payload = "[DONE]" if data is None else json.dumps(data)
        raw = ("data: " + payload + "\n\n").encode()
        # Frame each event so the HTTP reader can deliver it without waiting
        # for the full response or a fixed-size buffer to fill.
        self.wfile.write(f"{len(raw):x}\r\n".encode() + raw + b"\r\n")
        if data is None:
            self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def validate_request(data):
    ids = data.get("prompt")
    if (not isinstance(ids, list) or len(ids) != 512
            or any(type(x) is not int or x < 0 for x in ids)):
        raise ValueError("Expected 512 nonnegative token IDs")
    required = dict(model="target", temperature=0, top_p=1, top_k=-1,
                    max_tokens=128, ignore_eos=True, stream=True, return_token_ids=True)
    if any(data.get(k) != v for k, v in required.items()):
        raise ValueError("Only the fixed greedy benchmark request is supported")
    return ids


def main():
    from minivllm.engine.arg_utils import EngineArgs
    from minivllm.engine.llm_engine import LLMEngine
    from minivllm.sampling_params import SamplingParams
    from benchmarks.benchmark_utils import environment_info

    parser = argparse.ArgumentParser(description=__doc__)
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(args))
    params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=128)
    commands = queue.Queue()
    active = {}
    version = {"version": "mini-vllm-benchmark", "environment": environment_info(),
               "config": engine.get_runtime_stats()["config"]}

    class Handler(StreamingHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, content_type="application/json"):
            raw = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, "{}")
            elif self.path == "/version":
                self.reply(200, json.dumps(version))
            elif self.path == "/metrics":
                answer = queue.Queue()
                commands.put(("stats", answer))
                stats = answer.get(timeout=600)
                body = "\n".join(f'{name}{{engine="0",model_name="target"}} {stats[k]}'
                                 for k, name in COUNTERS.items()) + "\n"
                self.reply(200, body, "text/plain; version=0.0.4")
            else:
                self.reply(404, "{}")

        def do_POST(self):
            if self.path in ("/start_profile", "/stop_profile"):
                answer = queue.Queue()
                commands.put(("profile", self.path == "/start_profile", answer))
                answer.get(timeout=600)
                self.reply(200, "{}")
                return
            if self.path != "/v1/completions":
                self.reply(404, "{}")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request size")
                ids = validate_request(json.loads(self.rfile.read(size)))
            except (ValueError, TypeError) as exc:
                self.reply(400, json.dumps({"error": str(exc)}))
                return
            request_id, answer = uuid.uuid4().hex, queue.Queue()
            commands.put(("add", request_id, ids, answer))
            self.start_stream()
            try:
                while True:
                    data = answer.get(timeout=600)
                    self.send_event(data)
                    if data is None:
                        break
            except (BrokenPipeError, ConnectionResetError, queue.Empty):
                pass
            finally:
                commands.put(("abort", request_id))

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps(version), flush=True)
    try:
        while True:
            try:
                command = commands.get(timeout=0 if active else 0.1)
            except queue.Empty:
                command = None
            # Admit all currently queued requests before the next model step.
            while command is not None:
                kind, *rest = command
                if kind == "add":
                    rid, ids, answer = rest
                    engine.add_request(rid, None, params, ids)
                    active[rid] = [answer, 0]
                elif kind == "abort":
                    rid = rest[0]
                    if rid in active:
                        engine.abort_request(rid)
                        del active[rid]
                elif kind == "stats":
                    rest[0].put(engine.get_runtime_stats()["speculative"])
                elif kind == "profile":
                    import torch
                    if rest[0]:
                        torch.cuda.cudart().cudaProfilerStart()
                    else:
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                    rest[1].put(True)
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    command = None
            if not active:
                continue
            for output in engine.step():
                answer, offset = active[output.request_id]
                completion = output.outputs[0]
                ids = list(completion.token_ids)
                finished = output.is_finished()
                answer.put({"choices": [{"index": 0, "text": "", "token_ids": ids[offset:],
                                         "finish_reason": "length" if finished and len(ids) == 128 else
                                         ("stop" if finished else None)}]})
                if finished:
                    answer.put({"usage": {"prompt_tokens": len(output.prompt_token_ids),
                                          "completion_tokens": len(ids)}})
                    answer.put(None)
                    del active[output.request_id]
                else:
                    active[output.request_id][1] = len(ids)
    except BaseException as exc:
        for answer, _ in active.values():
            answer.put({"error": str(exc)})
            answer.put(None)
        raise
    finally:
        server.shutdown()
        server.server_close()
        engine._run_workers("close_transfer_engine")


if __name__ == "__main__":
    main()
