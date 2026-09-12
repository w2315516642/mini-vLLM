import json
import unittest

from benchmarks.benchmark_vllm_serving import consume_stream, counter_delta, parse_counters


def stream_event(**values):
    return "data: " + json.dumps(values)


class ServingBenchmarkTest(unittest.TestCase):
    def test_stream_bursts_and_usage(self):
        lines = [stream_event(choices=[{"token_ids": [3, 4], "text": "hi"}]),
                 stream_event(choices=[{"token_ids": [5], "finish_reason": "length"}]),
                 stream_event(usage={"prompt_tokens": 512, "completion_tokens": 3}),
                 "data: [DONE]"]
        times = iter([11., 12.])
        result = consume_stream(lines, 512, 3, 10., clock=lambda: next(times))
        self.assertEqual(result["token_ids"], [3, 4, 5])
        self.assertEqual(result["ttft_ms"], 1000)
        self.assertEqual(result["tpot_ms"], 500)
        self.assertEqual([x["tokens"] for x in result["updates"]], [2, 1])

    def test_truncated_and_error_streams_fail(self):
        for lines in ([], [stream_event(error="failed")],
                      [stream_event(choices=[{"text": "missing IDs"}])]):
            with self.assertRaises((ValueError, RuntimeError)):
                consume_stream(lines, 512, 128, 0)

    def test_usage_mismatch(self):
        lines = [stream_event(choices=[{"token_ids": [1], "finish_reason": "length"}]),
                 stream_event(usage={"prompt_tokens": 513, "completion_tokens": 128}),
                 "data: [DONE]"]
        with self.assertRaises(ValueError):
            consume_stream(lines, 512, 128, 0)

    def test_counter_labels_and_created_exclusion(self):
        body = '\n'.join([
            'vllm:spec_decode_num_drafts_total{engine="0",model_name="target"} 12',
            'vllm:spec_decode_num_drafts_total{engine="0",model_name="other"} 99',
            'vllm:spec_decode_num_drafts_created{model_name="target"} 100000',
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="target",position="0"} 9',
        ])
        self.assertEqual(parse_counters(body, "target"),
                         {"verification_rounds": 12, "position_0": 9})

    def test_counter_delta(self):
        before = dict(verification_rounds=12, verified_draft_tokens=36, accepted_draft_tokens=22)
        after = dict(verification_rounds=24, verified_draft_tokens=72, accepted_draft_tokens=44)
        result = counter_delta(before, after, "dspark")
        self.assertAlmostEqual(result["accepted_tokens_per_round"], 22 / 12)
        self.assertAlmostEqual(result["acceptance_rate"], 22 / 36)
        for a, b, mode in [(after, before, "dspark"), (before, before, "dspark"),
                           (before, after, "target")]:
            with self.assertRaises(ValueError):
                counter_delta(a, b, mode)
        self.assertIsNone(counter_delta({}, {}, "target")["acceptance_rate"])


if __name__ == "__main__":
    unittest.main()
