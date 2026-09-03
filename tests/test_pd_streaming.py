import socket
import threading
import unittest
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

from minivllm.engine.pd_rpc import PDClient, PDControlServer, RemoteEngineError
from minivllm.outputs import CompletionOutput, RequestOutput, RequestOutputKind
from minivllm.sampling_params import SamplingParams


def output(request_id, text, tokens, finished=False):
    return RequestOutput(request_id, "prompt", [9], [CompletionOutput(
        0, text, tokens, 0.0, None, "length" if finished else None,
    )])


class PrefillEngine:
    def __init__(self):
        self.request_id = None
        self.handoffs = []
        self.aborted = []
        self.finish_on_prefill = False

    def add_request(self, request_id, *args, **kwargs):
        self.request_id = request_id

    def step(self):
        if not self.finish_on_prefill:
            self.handoffs.append(SimpleNamespace(request_id=self.request_id))
        return [output(self.request_id, "A", [1], self.finish_on_prefill)]

    def pop_prefill_handoffs(self):
        handoffs, self.handoffs = self.handoffs, []
        return handoffs

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.request_id = None
        self.handoffs = []


class DecodeEngine:
    def __init__(self):
        self.request_id = None
        self.steps = 0
        self.aborted = []
        self.error = False

    def has_unfinished_requests(self):
        return self.request_id is not None

    def step(self):
        if self.error:
            raise ValueError("decode failed")
        self.steps += 1
        if self.steps == 1:
            # Cumulative D output includes the first token already sent by P.
            return [output(self.request_id, "ABC", [1, 2, 3])]
        result = [output(self.request_id, "ABCD", [1, 2, 3, 4], True)]
        self.request_id = None
        return result

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.request_id = None


class PDStreamingTest(unittest.TestCase):
    def setUp(self):
        self.prefill = PrefillEngine()
        self.decode = DecodeEngine()
        self.servers = []
        addresses = []
        for engine in (self.prefill, self.decode):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            address = f"127.0.0.1:{port}"
            server = PDControlServer(engine, address, b"test-key")
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers.append(server)
            addresses.append(address)
        self.client = PDClient(*addresses, b"test-key")
        self.client.bridge = Mock()

        def transfer(handoff):
            self.decode.request_id = handoff.request_id
            self.prefill.request_id = None

        self.client.bridge.transfer.side_effect = transfer
        self.params = SamplingParams(temperature=0.0, max_tokens=4)

    def tearDown(self):
        self.client.close()
        for server in self.servers:
            server.close()

    def test_rpc_stream_emits_first_token_before_transfer_without_repeating_it(self):
        with closing(self.client.generate_stream(
            "prompt", self.params, request_id="request",
        )) as stream:
            first = next(stream)
            self.assertEqual(first.outputs[0].text, "A")
            self.client.bridge.transfer.assert_not_called()
            self.assertEqual(self.decode.steps, 0)
            rest = list(stream)
        self.assertEqual([x.outputs[0].text for x in rest], ["BC", "D"])
        self.assertEqual([x.outputs[0].token_ids for x in rest], [[2, 3], [4]])
        self.assertTrue(rest[-1].is_finished())
        self.client.bridge.coordinator.finish.assert_called_once_with("request")
        self.assertEqual(self.prefill.aborted + self.decode.aborted, [])

    def test_close_before_transfer_releases_sealed_prefill_request(self):
        stream = self.client.generate_stream("prompt", self.params, request_id="request")
        next(stream)
        stream.close()
        self.assertEqual(self.prefill.aborted, ["request"])
        self.assertEqual(self.prefill.handoffs, [])
        self.client.bridge.transfer.assert_not_called()
        self.assertFalse(self.client._generating)

    def test_close_during_decode_cancels_only_decode_owner(self):
        stream = self.client.generate_stream("prompt", self.params, request_id="request")
        next(stream)
        next(stream)
        stream.close()
        self.assertEqual(self.prefill.aborted, [])
        self.assertEqual(self.decode.aborted, ["request"])
        self.client.bridge.coordinator.cancel.assert_called_once_with("request")

    def test_prefill_can_finish_without_handoff(self):
        self.prefill.finish_on_prefill = True
        chunks = list(self.client.generate_stream("prompt", self.params))
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].is_finished())
        self.client.bridge.transfer.assert_not_called()

    def test_non_streaming_metrics_api_keeps_final_snapshot(self):
        final, metrics = self.client.generate_with_metrics("prompt", self.params)
        self.assertEqual(final.outputs[0].text, "ABCD")
        self.assertEqual(final.outputs[0].token_ids, [1, 2, 3, 4])
        for name in ("prefill_s", "transfer_s", "decode_s", "ttft_s", "tpot_s"):
            self.assertGreaterEqual(getattr(metrics, name), 0.0)

    def test_final_only_stream_and_original_generate_match(self):
        chunks = list(self.client.generate_stream(
            "prompt", self.params, output_kind=RequestOutputKind.FINAL_ONLY,
        ))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].outputs[0].text, "ABCD")
        self.decode.steps = 0
        final = self.client.generate("prompt", self.params)
        self.assertEqual(final.outputs[0].text, chunks[0].outputs[0].text)

    def test_remote_error_propagates_and_releases_decode_request(self):
        self.decode.error = True
        stream = self.client.generate_stream("prompt", self.params, request_id="request")
        next(stream)
        with self.assertRaisesRegex(RemoteEngineError, "ValueError: decode failed"):
            next(stream)
        self.assertEqual(self.decode.aborted, ["request"])

    def test_active_client_rejects_a_second_generation(self):
        with closing(self.client.generate_stream("prompt", self.params)) as stream:
            next(stream)
            with self.assertRaisesRegex(RuntimeError, "active PD generation"):
                self.client.generate("other", self.params)


if __name__ == "__main__":
    unittest.main()
