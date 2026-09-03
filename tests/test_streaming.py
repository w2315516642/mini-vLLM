import io
import json
import runpy
import sys
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from minivllm.engine.llm_engine import LLMEngine
from minivllm.engine.output_processor import OutputProcessor
from minivllm.entrypoints.llm import LLM
from minivllm.outputs import CompletionOutput, RequestOutput, RequestOutputKind
from minivllm.sampling_params import SamplingParams
from minivllm.sequence import Sequence, SequenceGroup, SequenceStatus
from minivllm.utils import Counter


def snapshot(text, tokens, reason=None, request_id="0", index=0):
    return RequestOutput(request_id, "prompt", [9], [CompletionOutput(
        index, text, list(tokens), -float(len(tokens)),
        [{token: -1.0} for token in tokens], reason,
    )])


class OutputProcessorTest(unittest.TestCase):
    def test_delta_tracks_text_tokens_and_logprobs_independently(self):
        processor = OutputProcessor(SamplingParams())
        first = processor.process(snapshot("A", [1]))
        block = processor.process(snapshot("ABC", [1, 2, 3]))
        final = processor.process(snapshot("ABC", [1, 2, 3], "length"))
        self.assertEqual(first.outputs[0].text, "A")
        self.assertEqual(block.outputs[0].text, "BC")
        self.assertEqual(block.outputs[0].token_ids, [2, 3])
        self.assertEqual(block.outputs[0].logprobs, [{2: -1.0}, {3: -1.0}])
        self.assertEqual(block.outputs[0].cumulative_logprob, -3.0)
        self.assertEqual(final.outputs[0].token_ids, [])
        self.assertEqual(final.outputs[0].text, "")
        self.assertEqual(final.outputs[0].finish_reason, "length")
        self.assertTrue(final.is_finished())
        self.assertEqual(processor._offsets, {})

    def test_cumulative_and_final_only(self):
        for kind in (RequestOutputKind.CUMULATIVE, RequestOutputKind.FINAL_ONLY):
            with self.subTest(kind=kind):
                processor = OutputProcessor(SamplingParams(), kind)
                first = processor.process(snapshot("A", [1]))
                final = processor.process(snapshot("AB", [1, 2], "stop"))
                if kind == RequestOutputKind.FINAL_ONLY:
                    self.assertIsNone(first)
                else:
                    self.assertEqual(first.outputs[0].text, "A")
                self.assertEqual(final.outputs[0].text, "AB")
                self.assertEqual(final.outputs[0].token_ids, [1, 2])

    def test_no_stop_prefix_leaks_before_cross_token_match(self):
        processor = OutputProcessor(SamplingParams(stop=["<end>"]))
        parts = []
        for text, tokens, reason in (
            ("hello", [1], None), ("hello<en", [1, 2], None),
            ("hello", [1, 2, 3], "stop"),
        ):
            update = processor.process(snapshot(text, tokens, reason))
            parts.append(update.outputs[0].text)
        self.assertEqual("".join(parts), "hello")
        self.assertNotIn("<", "".join(parts))

    def test_incomplete_stop_prefix_is_flushed_on_length_limit(self):
        processor = OutputProcessor(SamplingParams(stop=["<end>"]))
        first = processor.process(snapshot("hello<en", [1, 2]))
        final = processor.process(snapshot("hello<en", [1, 2], "length"))
        self.assertEqual(first.outputs[0].text + final.outputs[0].text, "hello<en")

    def test_incomplete_utf8_is_not_printed_but_tokens_are_delivered(self):
        processor = OutputProcessor(SamplingParams())
        first = processor.process(snapshot("\ufffd", [1]))
        second = processor.process(snapshot("\u4e2d", [1, 2]))
        self.assertEqual(first.outputs[0].text, "")
        self.assertEqual(first.outputs[0].token_ids, [1])
        self.assertEqual(second.outputs[0].text, "\u4e2d")
        self.assertEqual(second.outputs[0].token_ids, [2])

    def test_offsets_use_completion_index_not_current_rank(self):
        processor = OutputProcessor(SamplingParams(n=2))
        first = snapshot("A", [1], "stop", index=0)
        first.outputs.append(snapshot("B", [2], index=1).outputs[0])
        first.finished = False
        processor.process(first)
        second = snapshot("BC", [2, 3], "length", index=1)
        second.outputs.append(first.outputs[0])
        final = processor.process(second)
        self.assertEqual(len(final.outputs), 1)
        self.assertEqual(final.outputs[0].index, 1)
        self.assertEqual(final.outputs[0].text, "C")
        self.assertTrue(final.is_finished())

    def test_finishing_one_completion_does_not_finish_whole_request(self):
        processor = OutputProcessor(SamplingParams(n=2))
        output = snapshot("A", [1], "stop")
        output.finished = False
        self.assertFalse(processor.process(output).is_finished())

    def test_requests_have_independent_offsets_and_empty_steps_are_suppressed(self):
        processor = OutputProcessor(SamplingParams())
        processor.process(snapshot("A", [1], request_id="a"))
        other = processor.process(snapshot("B", [2], request_id="b"))
        self.assertEqual(other.outputs[0].text, "B")
        self.assertIsNone(
            processor.process(snapshot("A", [1], request_id="a"))
        )

    def test_unstable_candidate_selection_is_rejected_before_streaming(self):
        for params in (SamplingParams(best_of=2), SamplingParams(use_beam_search=True)):
            with self.assertRaisesRegex(ValueError, "best_of == n"):
                OutputProcessor(params)
            OutputProcessor(params, RequestOutputKind.FINAL_ONLY)


class ByteTokenizer:
    all_special_tokens = ["<eos>"]
    added_tokens_encoder = {"<eos>": 0}
    eos_token_id = 0
    tokens = {0: "<eos>", 1: "A", 2: "B", 3: "ENDtail", 4: "\xe4", 5: "\xb8\xad"}

    def convert_ids_to_tokens(self, token_id):
        return self.tokens[token_id]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens).encode("latin-1").decode("utf-8", errors="replace")


class DetokenizationTest(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(LLMEngine)
        self.engine.tokenizer = ByteTokenizer()
        self.engine.scheduler = Mock()
        self.engine.scheduler.free_seq.side_effect = (
            lambda seq, status: setattr(seq, "status", status)
        )
        self.seq = Sequence(0, "prompt", [9], 16)
        self.seq.status = SequenceStatus.RUNNING
        self.group = SequenceGroup("0", [self.seq], SamplingParams(), 0.0)

    def append(self, token_id):
        self.seq.append_token_id(token_id, {token_id: -1.0})
        self.engine._decode_sequences([self.group])
        self.engine._stop_sequences([self.group])
        return RequestOutput.from_seq_group(self.group)

    def test_cumulative_text_is_not_repeated_and_eos_is_not_rendered(self):
        first = self.append(1)
        self.append(2)
        final = self.append(0)
        self.assertEqual(first.outputs[0].text, "A")
        self.assertEqual(first.outputs[0].token_ids, [1])
        self.assertEqual(final.outputs[0].text, "AB")
        self.assertEqual(final.outputs[0].finish_reason, "stop")
        self.assertEqual(len(self.seq.output_tokens), 3)

    def test_real_decoding_of_split_utf8_matches_stream_concatenation(self):
        processor = OutputProcessor(self.group.sampling_params)
        parts = [
            processor.process(self.append(token)).outputs[0].text
            for token in (4, 5, 0)
        ]
        self.assertEqual("".join(parts), "\u4e2d")
        self.assertEqual(self.seq.output_text, "\u4e2d")

    def test_stop_inside_speculative_block_excludes_trailing_text(self):
        self.group.sampling_params.stop = ["END"]
        self.seq.append_token_id(1, {1: -1.0})
        self.seq.append_token_id(3, {3: -1.0})
        self.engine._decode_sequences([self.group])
        self.engine._stop_sequences([self.group])
        self.assertEqual(self.seq.output_text, "A")
        self.assertTrue(self.seq.is_finished())

    def test_snapshot_owns_lists_and_group_finished_is_not_top_n_finished(self):
        self.group.sampling_params.logprobs = 1
        first = self.append(1)
        self.seq.output_logprobs[0][1] = -99.0
        self.assertEqual(first.outputs[0].logprobs, [{1: -1.0}])
        self.group.sampling_params.best_of = 2
        self.seq.status = SequenceStatus.FINISHED_STOPPED
        other = Sequence(1, "prompt", [9], 16)
        other.data.cumulative_logprob = -2.0
        self.group.seqs.append(other)
        output = RequestOutput.from_seq_group(self.group)
        self.assertTrue(output.outputs[0].finished())
        self.assertFalse(output.is_finished())

    def test_abort_removes_pending_pd_handoff_notification(self):
        self.engine._sealed_handoffs = [
            SimpleNamespace(request_id="0"), SimpleNamespace(request_id="1")
        ]
        self.engine._flush_pending_state_operations = Mock()
        self.engine.abort_request("0")
        self.assertEqual(
            [x.request_id for x in self.engine._sealed_handoffs], ["1"]
        )
        self.engine.scheduler.abort_seq_group.assert_called_once_with("0")


class ScriptedEngine:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.pending = set()
        self.calls = 0
        self.aborted = []
        self.added = []

    def add_request(self, request_id, *args, **kwargs):
        self.pending.add(request_id)
        self.added.append((request_id, args, kwargs))

    def has_unfinished_requests(self):
        return bool(self.pending)

    def get_num_unfinished_requests(self):
        return len(self.pending)

    def step(self):
        self.calls += 1
        result = next(self.steps)
        if isinstance(result, Exception):
            raise result
        for output in result:
            if output.is_finished():
                self.pending.discard(output.request_id)
        return result

    def abort_request(self, request_id):
        self.aborted.append(request_id)
        self.pending.discard(request_id)


def make_llm(steps):
    llm = object.__new__(LLM)
    llm.llm_engine = ScriptedEngine(steps)
    llm.request_conuter = Counter()
    llm._generation_active = False
    return llm


class LLMStreamingTest(unittest.TestCase):
    def test_stream_is_lazy_and_delivers_before_generation_completes(self):
        llm = make_llm([
            [], [snapshot("A", [1])], [snapshot("ABC", [1, 2, 3], "length")]
        ])
        with closing(llm.generate_stream("prompt")) as stream:
            self.assertEqual(llm.llm_engine.calls, 0)
            first = next(stream)
            self.assertEqual(first.outputs[0].text, "A")
            self.assertEqual(llm.llm_engine.calls, 2)
            final = next(stream)
            self.assertEqual(final.outputs[0].text, "BC")
            self.assertTrue(final.is_finished())
        self.assertEqual(llm.llm_engine.aborted, [])

    def test_generate_keeps_final_results_in_prompt_order(self):
        llm = make_llm([
            [snapshot("B", [2], "stop", request_id="1")],
            [snapshot("A", [1], "length")],
        ])
        outputs = llm.generate(["first", "second"], use_tqdm=False)
        self.assertEqual([o.request_id for o in outputs], ["0", "1"])
        self.assertEqual([o.outputs[0].text for o in outputs], ["A", "B"])

    def test_early_close_aborts_batch_and_releases_single_consumer_guard(self):
        llm = make_llm([[snapshot("A", [1])]])
        stream = llm.generate_stream(["one", "two"])
        next(stream)
        with self.assertRaisesRegex(RuntimeError, "active generation"):
            llm.generate("other", use_tqdm=False)
        stream.close()
        self.assertCountEqual(llm.llm_engine.aborted, ["0", "1"])
        self.assertFalse(llm._generation_active)
        self.assertFalse(llm.llm_engine.pending)

    def test_step_exception_cleans_up_request(self):
        llm = make_llm([RuntimeError("failed step")])
        with self.assertRaisesRegex(RuntimeError, "failed step"):
            list(llm.generate_stream("prompt"))
        self.assertEqual(llm.llm_engine.aborted, ["0"])
        self.assertFalse(llm._generation_active)

    def test_empty_batch_does_not_step_engine(self):
        llm = make_llm([])
        self.assertEqual(list(llm.generate_stream([])), [])
        self.assertEqual(llm.llm_engine.calls, 0)

    def test_chat_stream_uses_existing_processor_inputs(self):
        llm = make_llm([])
        llm._prepare_chat = Mock(return_value=[{"processed": True}])
        llm.generate_stream = Mock(return_value=iter(()))
        llm.chat_stream([{"role": "user", "content": "image"}], enable_thinking=False)
        llm.generate_stream.assert_called_once_with(
            sampling_params=None, multi_modal_inputs=[{"processed": True}],
            output_kind=RequestOutputKind.DELTA,
        )


class StreamingCLITest(unittest.TestCase):
    def test_generation_script_streams_once_and_counts_final_tokens(self):
        llm = Mock()
        llm.generate.return_value = [snapshot("warmup", [7], "length")]
        llm.generate_stream.return_value = (item for item in (
            snapshot("A", [1]), snapshot("AB", [1, 2], "length"),
        ))
        script = Path(__file__).resolve().parents[1] / "scripts/autodl/run_generation.py"
        stdout = io.StringIO()
        argv = [str(script), "--model", "test", "--stream", "--warmup", "1"]
        with patch("minivllm.LLM", return_value=llm), \
                patch.object(sys, "argv", argv), redirect_stdout(stdout):
            runpy.run_path(str(script), run_name="__main__")
        text, metrics_json = stdout.getvalue().split("\n", 1)
        self.assertEqual(text, "AB")
        self.assertEqual(json.loads(metrics_json)["generated_tokens"], 2)
        llm.generate.assert_called_once()
        self.assertEqual(
            llm.generate_stream.call_args.kwargs["output_kind"],
            RequestOutputKind.CUMULATIVE,
        )

    def test_pd_cli_prints_delta_and_closes_client(self):
        from minivllm.entrypoints import pd_generate

        client = Mock()
        client.generate_stream.return_value = (item for item in (
            snapshot("A", [1]), snapshot("B", [2], "length"),
        ))
        argv = [
            "pd_generate", "--prefill-control", "localhost:15000",
            "--decode-control", "localhost:15100", "--control-authkey", "key",
            "--prompt", "test", "--stream",
        ]
        stdout = io.StringIO()
        with patch.object(pd_generate, "PDClient", return_value=client), \
                patch.object(sys, "argv", argv), redirect_stdout(stdout):
            pd_generate.main()
        self.assertEqual(stdout.getvalue(), "AB\n")
        client.close.assert_called_once()
        client.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
