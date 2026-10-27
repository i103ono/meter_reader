"""Batch preparation, fixed denominator, resume and tabular exports; no live API."""
from contextlib import redirect_stdout, redirect_stderr
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from batch_prepare import load_source, prepare_batch, load_prepared_batch
from benchmark import run_benchmark
from evaluation import evaluate_experiment, main as evaluate_main
from evaluation_csv import safe_text, truth_columns
from provider import HTTPResult, OpenAIProvider, ProviderConfig, TransportFailure
from reader import MockReader
from schemas import ObservationInput, ReaderOutput
from vlm_reader import VLMReader


class Batch20Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "image.png"
        with Image.new("RGB", (80, 60), "gray") as image:
            image.save(self.image)
        self.source = self.root / "inputs.jsonl"
        self.source.write_text(json.dumps({"filename": "image.png", "question_id": "q1", "question": "Read upper display"}) + "\n")
        self.workspace = self.root / "batch"
        self.labels = self.root / "labels.jsonl"
        self.labels.write_text(json.dumps({"question_id": "q1", "evaluator": "interval_matching", "image_type": "test_category",
                                           "evaluator_kwargs": {"interval": [1, 1], "units": ["mm"]}}) + "\n")
        self.sent = []
        environment = patch.dict(os.environ, {"METER_READER_API_KEY": "fixture-not-real"})
        environment.start()
        self.addCleanup(environment.stop)

    def reader(self, outcomes):
        queue = list(outcomes)
        def transport(request, timeout):
            self.sent.append(json.loads(request.data))
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            body = {"status": "completed", "model": "test-not-real", "usage": {"input_tokens": 1, "output_tokens": 1},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(item)}]}]}
            return HTTPResult(json.dumps(body).encode(), 200)
        return VLMReader(OpenAIProvider(ProviderConfig(model="test-not-real"), transport=transport))

    def proposal(self, status="located"):
        return {"status": status, "roi": [10, 10, 70, 50] if status == "located" else None,
                "target_description": "Test rectangle" if status == "located" else None,
                "reason": "Test ambiguity" if status != "located" else None, "issue_codes": []}

    def test_original_input_conversion_reads_provenance_without_labels(self):
        (self.root / "provenance.json").write_text(json.dumps({"dataset_id": "test-set", "dataset_revision": "rev", "seed": 42}))
        requests = load_source(self.source)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].image_path, self.image)
        self.assertEqual(requests[0].question, "Read upper display")
        self.assertEqual(requests[0].dataset.revision, "rev")
        self.assertEqual(requests[0].dataset.seed, 42)

    def test_label_shaped_or_escaping_input_is_rejected(self):
        for row in ({"filename": "image.png", "question_id": "q1", "question": "read", "truth_lower": 1},
                    {"filename": "../outside.png", "question_id": "q1", "question": "read"}):
            self.source.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                load_source(self.source)

    def test_count_check_precedes_workspace_write_and_api(self):
        reader = self.reader([])
        with self.assertRaises(ValueError):
            prepare_batch(load_source(self.source), reader, self.workspace, expected_count=20)
        self.assertFalse(self.workspace.exists())
        self.assertEqual(self.sent, [])

    def test_roi_proposals_are_cached_and_not_requested_again(self):
        reader = self.reader([self.proposal()])
        state = prepare_batch(load_source(self.source), reader, self.workspace)
        first = state.read_bytes()
        prepare_batch(load_source(self.source), reader, self.workspace)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(state.read_bytes(), first)
        requests, unresolved = load_prepared_batch(state)
        self.assertEqual(requests[0].roi, (10, 10, 70, 50))
        self.assertEqual(unresolved, {})
        self.assertTrue((self.workspace / "roi_contact_sheet.png").exists())
        payload = json.dumps(self.sent)
        self.assertNotIn("interval", payload)
        self.assertNotIn("evaluator", payload)

    def test_unresolved_roi_is_kept_until_explicit_retry_or_manual_override(self):
        reader = self.reader([self.proposal("ambiguous"), self.proposal()])
        requests = load_source(self.source)
        state = prepare_batch(requests, reader, self.workspace)
        prepare_batch(requests, reader, self.workspace)
        self.assertEqual(len(self.sent), 1)
        _, unresolved = load_prepared_batch(state)
        self.assertEqual(unresolved, {"q1": "ambiguous"})
        prepare_batch(requests, reader, self.workspace, retry_unresolved=True)
        self.assertEqual(len(self.sent), 2)
        prepared, unresolved = load_prepared_batch(state)
        self.assertEqual(unresolved, {})
        prepare_batch(requests, reader, self.workspace, overrides={"q1": [0, 0, 80, 60]})
        self.assertEqual(load_prepared_batch(state)[0][0].roi, (0, 0, 80, 60))
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(len(json.loads(state.read_text())["cases"][0]["attempts"]), 2)

    def test_bad_override_or_changed_image_is_rejected(self):
        requests = load_source(self.source)
        with self.assertRaises(ValueError):
            prepare_batch(requests, MockReader(), self.workspace, overrides={"q1": [0, 0, 81, 60]})
        self.assertFalse(self.workspace.exists())
        state = prepare_batch(requests, MockReader(), self.workspace)
        with Image.new("RGB", (80, 60), "red") as image:
            image.save(self.image)
        with self.assertRaises(ValueError):
            load_prepared_batch(state)
        with self.assertRaises(ValueError):
            prepare_batch(requests, MockReader(), self.workspace)

    def test_bad_override_container_and_tampered_source_state_fail_before_api(self):
        requests = load_source(self.source)
        with self.assertRaises(ValueError):
            prepare_batch(requests, self.reader([]), self.workspace, overrides=[])
        state = prepare_batch(requests, MockReader(), self.workspace)
        document = json.loads(state.read_text())
        document["cases"][0]["request"]["question"] = "different display"
        state.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            prepare_batch(requests, self.reader([]), self.workspace)
        self.assertEqual(self.sent, [])

    def test_authentication_failure_stops_roi_batch_without_dropping_pending_cases(self):
        self.source.write_text(self.source.read_text() + json.dumps({"filename": "image.png", "question_id": "q2", "question": "read"}) + "\n")
        reader = self.reader([TransportFailure("authentication_error", status=401)])
        state = prepare_batch(load_source(self.source), reader, self.workspace)
        rows = json.loads(state.read_text())["cases"]
        self.assertEqual([r["status"] for r in rows], ["error", "pending"])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(load_prepared_batch(state)[0]), 2)

    def test_interrupted_roi_request_does_not_report_a_complete_known_call_count(self):
        requests = load_source(self.source)
        with self.assertRaises(KeyboardInterrupt):
            prepare_batch(requests, self.reader([KeyboardInterrupt()]), self.workspace)
        state = self.workspace / "batch.json"
        self.assertEqual(json.loads(state.read_text())["cases"][0]["status"], "running")
        prepare_batch(requests, self.reader([self.proposal()]), self.workspace)
        prepared, unresolved = load_prepared_batch(state)
        experiment = run_benchmark(prepared, MockReader(), self.root / "experiments", preparation_reference=state)
        metadata = json.loads(experiment.read_text())["roi_preparation"]
        self.assertIsNone(metadata["api_calls"])
        self.assertEqual(metadata["known_api_calls"], 1)

    def test_unresolved_bcd_are_abstentions_in_the_same_denominator(self):
        state = prepare_batch(load_source(self.source), MockReader(), self.workspace)
        requests, unresolved = load_prepared_batch(state)
        experiment = run_benchmark(requests, MockReader(), self.root / "experiments", unresolved_rois=unresolved, preparation_reference=state)
        report = evaluate_experiment(experiment, self.labels)
        self.assertEqual(report["summaries"]["A"]["n"], 1)
        for condition in "BCD":
            summary = report["summaries"][condition]
            self.assertEqual(summary["n"], 1)
            self.assertEqual(summary["roi_skipped_cases"], 1)
            self.assertEqual(summary["technical_error_rate"], 0)
            self.assertEqual(summary["needs_review_rate"], 1)
            self.assertEqual(summary["abstention_rate"], 1)
            self.assertEqual(summary["missing_results"], 0)
        self.assertEqual(report["roi_preparation"]["api_calls"], 0)

    def test_resume_completed_experiment_makes_no_more_calls(self):
        requests = [ObservationInput(image_path=self.image, roi=[0, 0, 80, 60], observation_id="obs1", dataset={"question_id": "q1"})]
        reader = MockReader()
        experiment = run_benchmark(requests, reader, self.root / "experiments")
        before = experiment.read_bytes()
        with patch.object(reader, "read", side_effect=AssertionError("Do not repeat completed reads")):
            again = run_benchmark(requests, reader, self.root / "experiments", resume=experiment)
        self.assertEqual(again, experiment)
        self.assertEqual(experiment.read_bytes(), before)

    def test_resume_refuses_changed_recipe_or_roi(self):
        request = ObservationInput(image_path=self.image, roi=[0, 0, 80, 60], observation_id="obs1", dataset={"question_id": "q1"})
        experiment = run_benchmark([request], MockReader(), self.root / "experiments")
        for changes in ({"fixed_recipe": "resize_2x"}, {"requests": [request.model_copy(update={"roi": (1, 1, 80, 60)})]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_benchmark(changes.get("requests", [request]), MockReader(), self.root / "experiments", resume=experiment,
                              fixed_recipe=changes.get("fixed_recipe", "gamma_mild"))

    def test_resume_keeps_generated_observation_ids(self):
        request = ObservationInput(image_path=self.image, roi=[0, 0, 80, 60], dataset={"question_id": "q1"})
        experiment = run_benchmark([request], MockReader(), self.root / "experiments")
        before = experiment.read_bytes()
        run_benchmark([request], MockReader(), self.root / "experiments", resume=experiment)
        self.assertEqual(experiment.read_bytes(), before)

    def test_interrupt_is_recorded_and_resume_does_not_invent_usage(self):
        requests = [ObservationInput(image_path=self.image, roi=[0, 0, 80, 60], observation_id="obs1", dataset={"question_id": "q1"})]
        reader = self.reader([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            run_benchmark(requests, reader, self.root / "experiments")
        experiment = next((self.root / "experiments").glob("*/experiment.json"))
        document = json.loads(experiment.read_text())
        self.assertEqual(document["conditions"]["A"][0]["state"], "running")
        outputs = [ReaderOutput().model_dump()] * 5 + [{"action": "stop", "recipe_id": None, "reason": "fixture", "reason_codes": []}]
        run_benchmark(requests, self.reader(outputs), self.root / "experiments", resume=experiment)
        report = evaluate_experiment(experiment, self.labels)
        self.assertEqual(report["conditions"]["A"][0]["interrupted_attempts"], 1)
        self.assertIsNone(report["summaries"]["A"]["api_calls"])
        self.assertIsNone(report["summaries"]["A"]["estimated_cost_usd"])

    def test_csv_exports_include_category_and_case_details_without_formulas(self):
        requests = [ObservationInput(image_path=self.image, roi=[0, 0, 80, 60], dataset={"question_id": "q1"})]
        experiment = run_benchmark(requests, MockReader(), self.root / "experiments")
        with redirect_stdout(io.StringIO()) as stdout:
            code = evaluate_main(["--experiment", str(experiment), "--labels", str(self.labels), "--output-dir", str(self.root / "evaluations")])
        self.assertEqual(code, 0)
        root = Path(stdout.getvalue().strip()).parent
        with (root / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["category"] for row in rows}, {"ALL", "test_category"})
        with (root / "cases.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual((row["truth_lower"], row["truth_upper"]), ("1", "1"))
            self.assertEqual(row["truth_value_unit"], "[1, 1] mm")
            self.assertIn("predicted_value", row)
            self.assertIn("predicted_unit", row)
        with (root / "diagnostics.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(rows)
        self.assertTrue(all(row["truth_unit"] == "mm" for row in rows))
        self.assertTrue(all(row["predicted_value"] == row["value"] for row in rows))
        self.assertTrue(all(row["predicted_unit"] == row["unit"] for row in rows))
        self.assertEqual(safe_text("=SUM(A1)"), "'=SUM(A1)")
        self.assertEqual(safe_text("-cmd"), "'-cmd")

    def test_csv_alternatives_keep_intervals_paired_with_units(self):
        row = {"truth_alternatives": [
            {"interval": ["0.15", "0.2"], "units": ["MPa"]},
            {"interval": ["1.5", "2"], "units": ["kgf/cm^2", "kgf/cm²"]}]}
        columns = truth_columns(row)
        self.assertIsNone(columns["truth_lower"])
        self.assertIsNone(columns["truth_upper"])
        self.assertEqual(columns["truth_value"], "[0.15, 0.2] | [1.5, 2]")
        self.assertEqual(columns["truth_unit"], "MPa | kgf/cm^2 / kgf/cm²")
        self.assertEqual(columns["truth_value_unit"], "[0.15, 0.2] MPa | [1.5, 2] kgf/cm^2 / kgf/cm²")


if __name__ == "__main__":
    unittest.main()
