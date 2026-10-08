"""Offline tests. All records and judge responses below are synthetic."""

import asyncio
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from benchmark.utils import api_client, pointq_paths, price_config
from evaluation import merge_judge_shards as merger
from evaluation import run_main_judge as perception
from evaluation import run_reasoning_ssfrq5d as reasoning
from evaluation.input_validation import validate_perception_inputs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def synthetic_inputs(root):
    csv_root = root / "csv"
    csv_root.mkdir()
    for qtype, filename in perception.CSV_FILES.items():
        with (csv_root / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "question", "gt", "is_boundary"])
            writer.writeheader()
            for number in range(1, 4):
                label = "ABC"[number - 1] if qtype == "how" else "I" if qtype == "what" else "A"
                writer.writerow({"index": f"toy-{number:03d}", "question": f"Synthetic {qtype}?",
                                 "gt": label, "is_boundary": "true" if number == 1 else "false"})
    records = []
    reference_root = root / "references"
    for number in range(1, 4):
        sid = f"synthetic::object_{number}"
        records.append({
            "pcqa_index": f"toy-{number:03d}", "sample_id": sid, "dataset": "synthetic",
            "perception": {"yesno": {"answer": "A"}, "what": {"answer": "I"},
                           "how": {"answer": "ABC"[number - 1]}},
            "reasoning": {"answer": "Synthetic geometry has uniform coverage. Quality is good."},
        })
        write_json(reference_root / f"final-toy-{number}.json", {
            "sample_id": sid, "ai_summary": {"summary_text": "Synthetic uniform geometry. Quality is good."},
            "stats": {"final_level": "good"},
        })
    bundle = {"meta": {"model": "synthetic-model", "view_setting": "synthetic-view"}, "results": records}
    predictions = root / "predictions.json"
    write_json(predictions, bundle)
    return csv_root, reference_root, predictions, bundle


class TemporaryInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv_root, self.references, self.predictions, self.bundle = synthetic_inputs(self.root)
        # Do not inherit credentials or runtime knobs, and fail if any test opens a socket.
        platform_env = {name: os.environ[name] for name in ("SystemRoot", "WINDIR", "TEMP", "TMP")
                        if name in os.environ}
        self.env = mock.patch.dict(os.environ, platform_env, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        network = mock.patch("socket.create_connection", side_effect=AssertionError("Offline test attempted network"))
        network.start()
        self.addCleanup(network.stop)


class PromptFidelityTests(unittest.TestCase):
    def test_perception_prompts_match_original_digest(self):
        names = ["JUDGE_SYSTEM", "JUDGE_USER_YESNO", "JUDGE_USER_WHAT", "JUDGE_USER_HOW"]
        value = json.dumps({name: getattr(perception, name) for name in names}, sort_keys=True).encode()
        self.assertEqual(hashlib.sha256(value).hexdigest(),
                         "8c6c33f346ba48b66d4d5db7da49402be4eb8efffc5db93c97a7557c9f137e95")

    def test_reasoning_prompts_match_original_digest(self):
        names = ["SSFRQ5D_JUDGE_SYSTEM", "COMMON_CONTEXT_PREFIX_TEMPLATE", "DIMENSION_PROMPTS"]
        value = json.dumps({name: getattr(reasoning, name) for name in names}, sort_keys=True).encode()
        self.assertEqual(hashlib.sha256(value).hexdigest(),
                         "6416a4029fe94de9b4fe45014e9e8aa7973b0f738a8a30774e69a661aebd8d35")


class PerceptionTests(TemporaryInputs):
    def test_label_parsers(self):
        self.assertEqual(perception._parse_judge_yesno(" YES "), "yes")
        self.assertEqual(perception._parse_judge_yesno("maybe"), perception.INVALID)
        self.assertEqual(perception._parse_judge_how(" usable "), "usable")
        self.assertEqual(perception._parse_judge_how("A"), perception.INVALID)
        self.assertEqual(perception._parse_judge_what("NONE"), set())
        self.assertEqual(perception._parse_judge_what("S1,NONE,S1,S3"), {"S1", "S3"})
        self.assertEqual(perception._parse_judge_what("S9"), perception.INVALID)

    def test_canonicalization_and_local_fallback(self):
        self.assertEqual(perception._canonicalize_judge_prediction("yesno", "Prediction: B"), "no")
        self.assertEqual(perception._canonicalize_judge_prediction("what", "Prediction: S3,S1"), "S1,S3")
        self.assertEqual(perception._local_parse_raw_answer("yesno", "Final answer: A"), "yes")
        self.assertEqual(perception._local_parse_raw_answer("how", "Final answer: C"), "bad")
        self.assertEqual(perception._local_parse_raw_answer("what", "Final answer: A,C"), {"S1", "S3"})
        self.assertIsNone(perception._local_parse_raw_answer("yesno", ""))

    def test_gt_mapping_and_boundary(self):
        gt = perception.load_gt(self.csv_root)
        self.assertEqual(gt["yesno"]["toy-001"]["gt"], "yes")
        self.assertEqual(gt["what"]["toy-001"]["gt"], set())
        self.assertEqual(gt["how"]["toy-003"]["gt"], "bad")
        self.assertTrue(gt["how"]["toy-001"]["is_boundary"])

    def test_valid_input(self):
        validate_perception_inputs(self.bundle, self.csv_root, perception.CSV_FILES)

    def test_bad_prediction_shape(self):
        for data in ([], {}, {"results": []}, {"results": [None]}, {"results": self.bundle["results"], "meta": []}):
            with self.subTest(data_type=type(data).__name__), self.assertRaises(ValueError):
                validate_perception_inputs(data, self.csv_root, perception.CSV_FILES)

    def test_duplicate_prediction_indices(self):
        self.bundle["results"].append(dict(self.bundle["results"][0]))
        with self.assertRaisesRegex(ValueError, "pcqa_index"):
            validate_perception_inputs(self.bundle, self.csv_root, perception.CSV_FILES)

    def test_unmatched_gt(self):
        self.bundle["results"][0]["pcqa_index"] = "unmatched"
        with self.assertRaisesRegex(ValueError, "missing from"):
            validate_perception_inputs(self.bundle, self.csv_root, perception.CSV_FILES)

    def test_missing_gt_file(self):
        (self.csv_root / "how_questions.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            validate_perception_inputs(self.bundle, self.csv_root, perception.CSV_FILES)

    def test_bad_gt_header_and_label(self):
        path = self.csv_root / "how_questions.csv"
        path.write_text("index,gt\ntoy-001,A\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "columns"):
            validate_perception_inputs(self.bundle, self.csv_root, perception.CSV_FILES)
        path.write_text("index,gt,question\ntoy-001,Z,Synthetic?\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "label"):
            validate_perception_inputs(self.bundle, self.csv_root, perception.CSV_FILES)

    def test_metric_invalid_and_empty_sets(self):
        records = [
            {"yesno": {"gt": "yes", "pred": "yes"}, "what": {"gt": [], "pred": []},
             "how": {"gt": "good", "pred": "good"}},
            {"yesno": {"gt": "no", "pred": "INVALID"}, "what": {"gt": ["S1"], "pred": "INVALID"},
             "how": {"gt": "usable", "pred": "INVALID"}},
            {"what": {"gt": ["S1", "S3"], "pred": ["S1"]}, "how": {"gt": "bad", "pred": "bad"}},
        ]
        result = perception.compute_main_table(records)
        self.assertEqual(result["main_table"], {"yesno_accuracy": 50.0, "what_sample_f1": 55.56,
                                                 "how_macro_f1": 66.67})
        self.assertEqual(result["diagnostics"]["yesno"]["invalid"], 1)

    def test_cache_compatibility(self):
        path = self.root / "cache.json"
        meta = {"cache_schema_version": perception.CACHE_SCHEMA_VERSION, "result_json": "synthetic.json",
                "judge_model": "synthetic-judge", "enable_thinking": False, "thinking_budget": None}
        perception._save_cache(path, [{"pcqa_index": "toy-001"}], meta)
        kwargs = dict(result_json="synthetic.json", judge_model="synthetic-judge",
                      enable_thinking=False, thinking_budget=None)
        self.assertIn("toy-001", perception._load_cache(path, **kwargs))
        kwargs["judge_model"] = "different-judge"
        self.assertEqual(perception._load_cache(path, **kwargs), {})

    def test_mocked_evaluation_pipeline(self):
        argv = ["judge", "--result-json", str(self.predictions), "--csv-root", str(self.csv_root),
                "--output", str(self.root / "scores.json"), "--cache-path", str(self.root / "cache.json"),
                "--cost-summary", str(self.root / "cost.json")]
        with mock.patch.object(sys, "argv", argv):
            args = perception.parse_args()
        args.api_key = str(mock.sentinel.offline_api_key)
        replies = [(f"Prediction: {answer}", {}, 0.0) for _ in range(3) for answer in ("yes", "NONE", "good")]
        replies[5] = ("Prediction: usable", {}, 0.0)
        replies[8] = ("Prediction: bad", {}, 0.0)
        with mock.patch.object(api_client.VLMClient, "call_async", new=mock.AsyncMock(side_effect=replies)) as call:
            with mock.patch.object(api_client, "_LOG_DIR", self.root / "logs"), contextlib.redirect_stdout(io.StringIO()):
                asyncio.run(perception.run_judge(args))
        self.assertEqual(call.await_count, 9)
        scores = json.loads((self.root / "scores.json").read_text(encoding="utf-8"))
        self.assertEqual(scores["main_table"], {"yesno_accuracy": 100.0, "what_sample_f1": 100.0,
                                                 "how_macro_f1": 100.0})
        self.assertEqual(len(scores["per_sample"]), 3)


class ReasoningTests(TemporaryInputs):
    def test_score_parser_legacy_clamping(self):
        for text, expected in [("Score: 2", 2), ('```json\n{"score": 1}\n```', 1),
                               ('{"score": 99}', 2), ('{"score": -9}', 0),
                               ('{"score": true}', None), ("not a score", None), ("0", 0)]:
            with self.subTest(text=text):
                self.assertEqual(reasoning.parse_dimension_score(text), expected)

    def test_windows_relative_id(self):
        self.assertEqual(reasoning.derive_sample_id({"rel_path": "synthetic\\folder\\cloud.ply"}),
                         "synthetic::folder/cloud")
        self.assertEqual(reasoning.derive_sample_id({"sample_id": "explicit", "rel_path": "x/y.ply"}), "explicit")
        self.assertEqual(reasoning.derive_sample_id({}), "")

    def test_extract_candidates_and_cleaner(self):
        record = {"reasoning": {"answer": "Quality is good."}, "reasoning_runs": [{"answer": "Quality is good."}],
                  "reasoning_text": "Coverage is uniform."}
        self.assertEqual(reasoning.extract_reasoning_candidates(record), ["Quality is good.", "Coverage is uniform."])
        cleaned = reasoning.clean_reasoning_text("- Coverage is uniform. Coverage is uniform.\nConclusion: quality is good.")
        self.assertEqual(cleaned.count("Coverage is uniform."), 1)
        self.assertTrue(cleaned.endswith("Overall, the quality of this point cloud is good."))
        self.assertLessEqual(len(reasoning.clean_reasoning_text("x" * 300, max_chars=40, append_canonical_tail=False)), 40)

    def test_prediction_bundle_formats(self):
        record = self.bundle["results"][0]
        for value in ([record], record, {"predictions": [record]}, {"items": [record]}, {"data": [record]}):
            write_json(self.predictions, value)
            self.assertEqual(reasoning.load_prediction_bundle(self.predictions)[0], [record])
        path = self.root / "synthetic.jsonl"
        path.write_text(json.dumps(record) + "\nmalformed\n[]\n", encoding="utf-8")
        self.assertEqual(reasoning.load_prediction_bundle(path), ([record], {}))

    def test_reference_loader_and_pairing(self):
        gt = reasoning.load_gt_map(self.references)
        self.assertEqual(len(gt), 3)
        self.assertEqual(gt["synthetic::object_1"]["final_level"], "good")
        records, _ = reasoning.load_prediction_bundle(self.predictions)
        pairs, stats = reasoning._collect_pairs(records, gt, max_samples=1, use_all_reasoning_runs=False,
                                               max_clean_chars=1200, append_canonical_tail=True)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(stats["pairs_built"], 1)
        self.assertEqual(pairs[0]["sample_id"], "synthetic::object_1")

    def test_retry_budgets_and_truncation(self):
        self.assertEqual(reasoning.parse_retry_budgets("512, 1024 2048"), [512, 1024, 2048])
        self.assertEqual(reasoning.normalize_retry_budgets(initial_budget=512, retry_budgets=[512, 1024, 1024, 2048],
                                                          max_extra_attempts=2), [1024, 2048])
        flag, reasons = reasoning.detect_truncation_suspected(raw_response="Score:", finish_reason="length",
                                                              usage={"output_tokens": 512}, max_completion_tokens=512)
        self.assertTrue(flag)
        self.assertIn("finish_reason_length", reasons)
        self.assertIn("output_tokens_hit_cap", reasons)

    def test_derived_metric_scaling(self):
        scores = reasoning._attach_derived_metrics(dict.fromkeys(reasoning.DIMENSIONS, 1))
        self.assertEqual(scores["ssfrq5d_total"], 5.0)
        self.assertEqual(scores["ssfrq5d_norm100"], 50.0)

    def run_options(self, **overrides):
        kwargs = dict(predictions_path=self.predictions, final_protocol_dir=self.references,
                      judge_model="synthetic-judge", api_key="", api_base="https://example.invalid/v1",
                      output_dir=self.root / "reasoning-output", num_judge_runs=1, max_concurrent=2,
                      max_samples=None, use_all_reasoning_runs=False, max_clean_chars=1200,
                      append_canonical_tail=True, reasoning_effort="low", max_completion_tokens=1024,
                      adaptive_completion_retry_budgets=[512, 1024], truncate_retry_max_attempts=2,
                      timeout=1.0, max_retries=1, retry_backoff=1.0, dry_run=True, force_rerun=False)
        kwargs.update(overrides)
        return kwargs

    def test_dry_run_no_api_and_no_scores(self):
        with mock.patch.object(api_client.VLMClient, "_build_async_client", side_effect=AssertionError("No API allowed")):
            with contextlib.redirect_stdout(io.StringIO()):
                summary = asyncio.run(reasoning.run_ssfrq5d_eval(**self.run_options()))
        self.assertEqual(summary["build_stats"]["pairs_built"], 3)
        self.assertTrue((self.root / "reasoning-output" / reasoning.RUN_MANIFEST_NAME).is_file())
        self.assertFalse((self.root / "reasoning-output" / "ssfrq5d_judge_scores.jsonl").exists())

    def test_zero_pairs_report_no_scores(self):
        write_json(self.predictions, [{"sample_id": "missing", "reasoning_text": "Synthetic unmatched reasoning."}])
        with contextlib.redirect_stdout(io.StringIO()):
            summary = asyncio.run(reasoning.run_ssfrq5d_eval(**self.run_options()))
        self.assertEqual(summary["build_stats"]["pairs_built"], 0)
        self.assertEqual(summary["build_stats"]["records_missing_gt"], 1)
        self.assertNotIn("ssfrq5d_total_mean", summary)

    def test_mocked_scoring_pipeline_and_resume(self):
        details = {"text": "Score: 2", "usage": {"output_tokens": 5}, "finish_reason": "stop"}
        call = mock.AsyncMock(return_value=(details, 0.0))
        with mock.patch.object(api_client.VLMClient, "call_async_detailed", new=call):
            with mock.patch.object(api_client, "_LOG_DIR", self.root / "logs"), contextlib.redirect_stdout(io.StringIO()):
                summary = asyncio.run(reasoning.run_ssfrq5d_eval(**self.run_options(dry_run=False)))
                again = asyncio.run(reasoning.run_ssfrq5d_eval(**self.run_options(dry_run=False)))
        self.assertEqual(call.await_count, 15)
        self.assertEqual(summary["ssfrq5d_total_mean"], 10.0)
        self.assertEqual(again["ssfrq5d_norm100_mean"], 100.0)

    def test_incomplete_dimension_excluded_from_aggregate(self):
        valid = {"text": "Score: 2", "usage": {"output_tokens": 5}, "finish_reason": "stop"}
        invalid = {"text": "synthetic unparseable response", "usage": {}, "finish_reason": "stop"}
        replies = [(invalid, 0.0)] + [(valid, 0.0)] * 14
        with mock.patch.object(api_client.VLMClient, "call_async_detailed", new=mock.AsyncMock(side_effect=replies)):
            with mock.patch.object(api_client, "_LOG_DIR", self.root / "logs"), contextlib.redirect_stdout(io.StringIO()):
                summary = asyncio.run(reasoning.run_ssfrq5d_eval(**self.run_options(dry_run=False)))
        self.assertEqual(summary["meta"]["scored_pairs"], 2)
        self.assertEqual(summary["meta"]["total_pairs"], 3)
        self.assertTrue((self.root / "reasoning-output" / reasoning.INVALID_RESPONSES_NAME).is_file())


class PathAndClientTests(TemporaryInputs):
    def test_repository_relative_defaults(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(pointq_paths.project_root(), root)
        self.assertEqual(pointq_paths.csv_root_default(), root / "data" / "csv")
        self.assertEqual(pointq_paths.final_protocol_dir(), root / "data" / "final_protocol")
        self.assertEqual(pointq_paths.screenshot_root(), root / "data" / "screenshots")

    def test_environment_overrides(self):
        pairs = {"POINTQ_PCQA_ROOT": pointq_paths.pcqa_inner_root,
                 "POINTQ_FINAL_PROTOCOL_DIR": pointq_paths.final_protocol_dir,
                 "POINTQ_CSV_ROOT": pointq_paths.csv_root_default, "POINTQ_WEBAPP_DIR": pointq_paths.webapp_dir,
                 "POINTQ_SCREENSHOT_ROOT": pointq_paths.screenshot_root,
                 "POINTQ_DATASETS_ROOT": pointq_paths.datasets_root}
        for name, function in pairs.items():
            with mock.patch.dict(os.environ, {name: str(self.root)}):
                self.assertEqual(function(), self.root.resolve())

    def test_price_overlay_and_unknown_cost(self):
        self.assertIsNone(price_config.estimate_cost(provider="openai", model="unknown", input_tokens=10,
                                                     output_tokens=20, cached_tokens=0, price_config={}))
        write_json(self.root / "token_price_config.json", {"openai": {"synthetic": {
            "input_per_1m": 1, "cached_input_per_1m": 0.5, "output_per_1m": 2}}})
        with mock.patch.object(price_config, "_BENCH_CFG", self.root / "nonexistent.json"):
            prices = price_config.load_merged_price_config(webapp_dir=self.root)
        cost = price_config.estimate_cost(provider="openai", model="synthetic", input_tokens=100,
                                          output_tokens=50, cached_tokens=40, price_config=prices)
        self.assertAlmostEqual(cost, 0.00018)

    def test_response_diagnostics(self):
        response = {"choices": [{"message": {"content": [{"text": "Score: 2"}]}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}
        details = api_client.extract_response_details(response, requested_max_completion_tokens=64)
        self.assertEqual(details["text"], "Score: 2")
        self.assertEqual(details["usage"]["total_tokens"], 7)
        self.assertFalse(details["content_was_empty"])

    def test_client_dry_run_bypasses_transport(self):
        client = api_client.VLMClient(api_key="", dry_run=True)
        with mock.patch.object(client, "_build_sync_client", side_effect=AssertionError("No transport")):
            details, latency = client.call_sync_detailed("synthetic", [], max_completion_tokens=12)
        self.assertEqual(details["response_id"], "__dry_run__")
        self.assertEqual(latency, 0.0)

    def test_sync_transport_closed_on_success_and_failure(self):
        client = api_client.VLMClient(api_key="", max_retries=1)
        transport = mock.MagicMock()
        transport.chat.completions.create.return_value = {"choices": []}
        with mock.patch.object(client, "_build_sync_client", return_value=transport):
            client.call_sync_detailed("synthetic", [])
        transport.close.assert_called_once()
        transport.close.reset_mock()
        transport.chat.completions.create.side_effect = RuntimeError("synthetic failure")
        with mock.patch.object(client, "_build_sync_client", return_value=transport), self.assertRaises(RuntimeError):
            client.call_sync_detailed("synthetic", [])
        transport.close.assert_called_once()

    def test_async_transport_closed(self):
        client = api_client.VLMClient(api_key="", max_retries=1)
        transport = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=mock.AsyncMock(return_value={"choices": []}))), close=mock.AsyncMock())
        with mock.patch.object(client, "_build_async_client", return_value=transport):
            asyncio.run(client.call_async_detailed("synthetic", []))
        transport.close.assert_awaited_once()

    def test_api_error_classification(self):
        error = RuntimeError("Error code: 401 unauthorized synthetic response")
        self.assertEqual(api_client.describe_api_error(error)["error_kind"], "auth_error")
        self.assertEqual(api_client.describe_api_error(RuntimeError("rate limit"))["error_kind"], "rate_limit_error")

    @unittest.skipIf(api_client.openai is None or api_client.httpx is None, "Install evaluation/requirements.txt for SDK checks")
    def test_sdk_client_construction_without_requests(self):
        client = api_client.VLMClient(api_key=str(mock.sentinel.offline_api_key),
                                      api_base="https://example.invalid/v1")
        sync_client = client._build_sync_client()
        sync_client.close()
        async_client = client._build_async_client()
        asyncio.run(async_client.close())

    @unittest.skipIf(api_client.openai is None or api_client.httpx is None, "Install evaluation/requirements.txt for SDK checks")
    def test_actual_sdk_over_mock_http_transport(self):
        seen = []

        def handle(request):
            seen.append(json.loads(request.content))
            return api_client.httpx.Response(200, json={
                "id": "synthetic-response", "object": "chat.completion", "created": 0,
                "model": "synthetic-judge", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Score: 2"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            })

        client = api_client.VLMClient(api_key=str(mock.sentinel.offline_api_key),
                                      api_base="https://example.invalid/v1", max_retries=1)
        sync_sdk = api_client.openai.OpenAI(api_key=client.api_key, base_url=client.api_base,
            http_client=api_client.httpx.Client(transport=api_client.httpx.MockTransport(handle)))
        with mock.patch.object(client, "_build_sync_client", return_value=sync_sdk):
            text, usage, _ = client.call_sync("synthetic-judge", [{"role": "user", "content": "Synthetic prompt"}],
                                            max_completion_tokens=32)
        self.assertEqual(text, "Score: 2")
        self.assertEqual(usage["total_tokens"], 5)
        async_sdk = api_client.openai.AsyncOpenAI(api_key=client.api_key, base_url=client.api_base,
            http_client=api_client.httpx.AsyncClient(transport=api_client.httpx.MockTransport(handle)))
        with mock.patch.object(client, "_build_async_client", return_value=async_sdk):
            text, _, _ = asyncio.run(client.call_async("synthetic-judge", [{"role": "user", "content": "Synthetic prompt"}],
                                                      max_completion_tokens=32))
        self.assertEqual(text, "Score: 2")
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(body["max_completion_tokens"] == 32 for body in seen))
        self.assertTrue(all("api_key" not in body for body in seen))

    def test_cli_uses_standard_environment_without_fallback_key(self):
        with mock.patch.object(sys, "argv", ["judge", "--result-json", "synthetic.json"]):
            args = perception.parse_args()
        self.assertEqual(args.api_key, "")
        self.assertEqual(args.api_base, "https://api.openai.com/v1")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": str(mock.sentinel.offline_key),
                                          "OPENAI_BASE_URL": "https://example.invalid/v1"}):
            with mock.patch.object(sys, "argv", ["judge", "--result-json", "synthetic.json"]):
                args = perception.parse_args()
            self.assertEqual(args.api_key, str(mock.sentinel.offline_key))
            self.assertEqual(args.api_base, "https://example.invalid/v1")

    def test_cli_help_module_and_direct(self):
        repo = Path(__file__).resolve().parents[1]
        for name in ("run_main_judge", "run_reasoning_ssfrq5d", "merge_judge_shards"):
            for argv in (["-m", f"evaluation.{name}"], [str(repo / "evaluation" / f"{name}.py")]):
                result = subprocess.run([sys.executable, "-B", *argv, "--help"], cwd=repo,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


class MergeTests(TemporaryInputs):
    def make_shards(self):
        ids = ["synthetic::object_1", "synthetic::object_2"]
        manifest = {"selected_sample_ids": ids, "num_shards": 2,
                    "shard_to_sample_ids": {"0": [ids[0]], "1": [ids[1]]}}
        path = self.root / "manifest.json"
        write_json(path, manifest)
        for number, sid in enumerate(ids):
            write_json(self.root / "shards" / f"shard_{number:02d}" / "judge_output.json", {
                "meta": {"judge_model": "synthetic-judge"},
                "per_sample": [{"sample_id": sid, "pcqa_index": f"toy-{number:03d}",
                                "yesno": {"gt": "yes", "pred": "yes"}}]})
        return path

    def test_complete_merge_recomputes_metrics(self):
        path = self.make_shards()
        output = self.root / "merged.json"
        with contextlib.redirect_stdout(io.StringIO()):
            summary = merger.merge_judge_shards(manifest_json=path, shards_root=self.root / "shards", output_json=output)
        merged = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(summary["merged_total"], 2)
        self.assertEqual(merged["main_table"]["yesno_accuracy"], 100.0)
        self.assertEqual(merged["per_sample"][0]["sample_id"], "synthetic::object_1")

    def test_missing_shard_fails(self):
        path = self.make_shards()
        (self.root / "shards" / "shard_01" / "judge_output.json").unlink()
        with self.assertRaises(FileNotFoundError):
            merger.merge_judge_shards(manifest_json=path, shards_root=self.root / "shards", output_json=self.root / "merged.json")

    def test_wrong_shard_ids_fail(self):
        path = self.make_shards()
        write_json(self.root / "shards" / "shard_01" / "judge_output.json", {"per_sample": [{"sample_id": "wrong"}]})
        with self.assertRaises(ValueError):
            merger.merge_judge_shards(manifest_json=path, shards_root=self.root / "shards", output_json=self.root / "merged.json")


if __name__ == "__main__":
    unittest.main()
