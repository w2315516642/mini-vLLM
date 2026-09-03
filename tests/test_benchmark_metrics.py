import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from benchmarks.benchmark_utils import WorkItem, prepare_prompts, selected_gpus, workload_info
from benchmarks.compare_results import compare_results
from benchmarks.streaming_metrics import RequestTrace, distribution, speculative_delta, summarize


def output(request_id, tokens, finished=False):
    return SimpleNamespace(request_id=request_id,
                           outputs=[SimpleNamespace(token_ids=tokens)],
                           is_finished=lambda: finished)


class MetricsTest(unittest.TestCase):
    def test_ttft_itl_and_decode_rate_exclude_prefill(self):
        trace = RequestTrace("a", 2048, 10.)
        trace.observe(output("a", [1]), 11., cumulative=False)
        trace.observe(output("a", [2]), 11.25, cumulative=False)
        trace.observe(output("a", [3], True), 11.5, cumulative=False)
        metrics, records = summarize([trace], [(10., 11.5)])
        self.assertEqual(records[0]["ttft_ms"], 1000.)
        self.assertEqual(records[0]["decode_tokens_per_s"], 4.)
        self.assertEqual(metrics["output_tokens_per_s"], 2.)
        self.assertEqual(metrics["itl_ms"]["p99"], 250.)

    def test_pd_cumulative_snapshot_does_not_duplicate_first_token(self):
        trace = RequestTrace("a", 128, 0.)
        trace.observe(output("a", [5]), 1., cumulative=True)
        trace.handoff_s = .5
        trace.observe(output("a", [5, 6]), 1.75, cumulative=True)
        trace.observe(output("a", [5, 6, 7], True), 2., cumulative=True)
        record = trace.record(0.)
        self.assertEqual(record["output_tokens"], 3)
        self.assertEqual(record["tpot_ms"], 500.)  # Includes handoff.
        self.assertEqual(record["handoff_ms"], 500.)

    def test_speculative_bursts_have_no_fabricated_itl(self):
        trace = RequestTrace("a", 128, 0.)
        trace.observe(output("a", [1]), 1., cumulative=False)
        trace.observe(output("a", [2, 3, 4, 5], True), 2., cumulative=False)
        metrics, records = summarize([trace], [(0., 2.)])
        self.assertIsNone(metrics["itl_ms"])
        self.assertEqual(metrics["inter_update_ms"]["count"], 1)
        self.assertEqual(records[0]["decode_tokens_per_s"], 4.)

    def test_empty_final_update_does_not_extend_decode_duration(self):
        trace = RequestTrace("a", 2, 0.)
        trace.observe(output("a", [1]), 1., cumulative=False)
        trace.observe(output("a", [2]), 2., cumulative=False)
        trace.observe(output("a", [], True), 3., cumulative=False)
        record = trace.record(0.)
        self.assertEqual(record["tpot_ms"], 1000.)
        self.assertEqual(record["e2e_ms"], 3000.)

    def test_single_token_has_no_tpot(self):
        trace = RequestTrace("a", 2, 0.)
        trace.observe(output("a", [1], True), 1., cumulative=False)
        self.assertIsNone(trace.record(0.)["tpot_ms"])

    def test_itl_percentile_is_pooled_not_mean_of_request_percentiles(self):
        traces = []
        for name, count, interval in (("a", 101, .001), ("b", 3, .1)):
            trace = RequestTrace(name, 1, 0.)
            for i in range(count):
                trace.observe(output(name, [i], i == count - 1), 1 + i * interval, cumulative=False)
            traces.append(trace)
        metrics, _ = summarize(traces, [(0., 2.)])
        self.assertEqual(metrics["itl_ms"]["count"], 102)
        self.assertAlmostEqual(metrics["itl_ms"]["p99"], 100.)

    def test_disjoint_windows_exclude_warmup_and_priming(self):
        traces = []
        for i, start in enumerate((0., 10.)):
            trace = RequestTrace(str(i), 1, start)
            trace.observe(output(str(i), [1], True), start + 1, cumulative=False)
            traces.append(trace)
        metrics, _ = summarize(traces, [(0., 1.), (10., 11.)])
        self.assertEqual(metrics["output_tokens_per_s"], 1.)

    def test_statistics_subtract_warmup_and_use_verified_width(self):
        def snapshot(rounds, verified, accepted):
            return {"speculative": dict(verification_rounds=rounds,
                    verified_draft_tokens=verified, accepted_draft_tokens=accepted)}
        delta = speculative_delta([snapshot(5, 35, 10)], [snapshot(7, 42, 13)])
        self.assertEqual(delta["accepted_tokens_per_round"], 1.5)
        self.assertEqual(delta["verified_tokens_per_round"], 3.5)
        self.assertAlmostEqual(delta["acceptance_rate"], 3 / 7)

    def test_percentile_empty_and_singleton(self):
        self.assertIsNone(distribution([])["p99"])
        self.assertEqual(distribution([2])["p99"], 2)


class DatasetTest(unittest.TestCase):
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: list(range(len(text))),
        all_special_ids=[0], get_vocab=lambda: {str(i): i for i in range(20)})

    def test_natural_inputs_are_truncated_never_padded(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "prompts.jsonl"
            path.write_text(json.dumps({"prompt": "abc"}) + "\n" + json.dumps({"prompt": "abcdefgh"}))
            prompts = prepare_prompts(self.tokenizer, 2, 5, dataset=str(path))
            self.assertEqual(prompts, [[0, 1, 2, 3, 4]] * 2)
            with self.assertRaisesRegex(ValueError, "No dataset prompt"):
                prepare_prompts(self.tokenizer, 1, 10, dataset=str(path))

    def test_synthetic_is_seeded_and_excludes_special_tokens(self):
        first = prepare_prompts(self.tokenizer, 2, 32, synthetic=True, seed=3)
        self.assertEqual(first, prepare_prompts(self.tokenizer, 2, 32, synthetic=True, seed=3))
        self.assertNotIn(0, first[0])

    def test_workload_hash_includes_timing(self):
        a = workload_info([WorkItem("a", [1, 2], 3, arrival_s=0.)])
        b = workload_info([WorkItem("a", [1, 2], 3, arrival_s=1.)])
        self.assertNotEqual(a["sha256"], b["sha256"])


def example_result():
    environment = {"cuda_visible_devices": "0", "cuda_device_order": "PCI_BUS_ID",
                   "torch": "test", "torch_cuda": "test",
                   "gpu_inventory": "0, GPU-a, TestGPU, 80000 MiB, 1\n1, GPU-b, TestGPU, 80000 MiB, 1"}
    config = {"model": "qwen", "dtype": "bf16", "tensor_parallel_size": 1, "pd_role": "unified"}
    metrics = {name: distribution([value]) for name, value in (
        ("ttft_ms", 100.), ("decode_tokens_per_s", 10.), ("tpot_ms", 100.),
        ("itl_ms", 200.), ("handoff_ms", 0.))}
    metrics.update(requests=2, completed_requests=2, output_tokens_per_s=20.)
    second_env = {**environment, "cuda_visible_devices": "1"}
    return {"schema_version": 1, "config": {"benchmark": "serving", "long_requests": 1},
            "workload": {"sha256": "same"}, "environment": environment,
            "servers": [{"config": config, "environment": environment},
                        {"config": dict(config), "environment": second_env}],
            "metrics": metrics, "cohorts": {"decode": dict(metrics)},
            "speculative": {}, "warnings": [], "decode_requests_overlapping_prefill_load": 1}


class ComparisonTest(unittest.TestCase):
    def test_equal_gpu_pd_comparison(self):
        baseline = example_result()
        candidate = copy.deepcopy(baseline)
        candidate["servers"][0]["config"]["pd_role"] = "prefill"
        candidate["servers"][1]["config"]["pd_role"] = "decode"
        candidate["cohorts"]["decode"]["itl_ms"]["p99"] = 120.
        result = compare_results(baseline, candidate)
        self.assertEqual(result["metrics"]["itl_ms.p99"]["improvement_percent"], 40.)

    def test_rejects_different_hardware_workload_and_unreported_config(self):
        for change in ("hardware", "workload", "configuration", "overlap"):
            a = example_result()
            b = copy.deepcopy(a)
            if change == "hardware":
                b["servers"][1]["environment"]["cuda_visible_devices"] = "0"
            elif change == "workload":
                b["workload"]["sha256"] = "different"
            elif change == "configuration":
                b["servers"][1]["config"]["max_num_seqs"] = 8
            else:
                b["decode_requests_overlapping_prefill_load"] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_results(a, b)

    def test_uuid_resolution(self):
        env = example_result()["environment"]
        self.assertEqual(selected_gpus({**env, "cuda_visible_devices": "GPU-a"}, 1)[0][0], "GPU-a")


if __name__ == "__main__":
    unittest.main()
