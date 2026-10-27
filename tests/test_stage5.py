"""Persistence and evaluation correctness; synthetic data and simulated HTTP only."""
from contextlib import redirect_stdout, redirect_stderr
import csv
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from benchmark import run_benchmark, main as benchmark_main
from cli import main
from evaluation import evaluate_experiment, load_labels, number, score_reading, trace_totals
from history_cli import main as history_main
from history_models import MeterRecord, ReviewInput
from provider import HTTPResult, OpenAIProvider, ProviderConfig
from reader import MockReader
from recording import run_observation
from repository import Repository
from schemas import ObservationInput, ReaderOutput, RunResult, ScaleInfo
from vlm_reader import VLMReader


def fixture_reading(value=12.59, unit="mm"):
    return ReaderOutput(meter_type="digital", value=value, raw_value_text=str(value) if value is not None else None,
                        unit=unit, value_status="readable" if value is not None else "unreadable",
                        unit_status="readable" if unit is not None else "unreadable",
                        scale=ScaleInfo(status="not_applicable")).model_dump()


class Stage5Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "meter.png"
        with Image.new("RGB", (80, 60), (75, 100, 150)) as image:
            image.save(self.image)
        self.image_bytes = self.image.read_bytes()
        self.database = self.root / "history.sqlite3"
        self.labels = self.root / "labels.jsonl"
        self.label = {"question_id": "q1", "evaluator": "interval_matching",
                      "evaluator_kwargs": {"interval": [12.59, 12.59], "units": ["mm", "Millimeter"]},
                      "truth_lower": 12.59, "truth_upper": 12.59, "source_group": "synthetic-only",
                      "image_sha256": hashlib.sha256(self.image_bytes).hexdigest()}
        self.labels.write_text(json.dumps(self.label) + "\n")

    def request(self, **changes):
        return ObservationInput(image_path=self.image, roi=[10, 10, 70, 50], meter_id="meter1",
                                observation_id=changes.pop("observation_id", "obs1"), question="Read test display",
                                dataset={"question_id": "q1"}, **changes)

    def run_file(self, *, prediction=None, **changes):
        if prediction is None:
            reader = MockReader()
        else:
            class Fixture:
                mode = "inference"
                reader_id = "offline-fixture"
                def read(self, request):
                    return ReaderOutput.model_validate(prediction)
            reader = Fixture()
        return run_observation(self.request(**changes), reader, self.root / "runs")

    def fake_reader(self, outputs=None):
        queue = list(outputs or [fixture_reading(), fixture_reading(), fixture_reading(), fixture_reading(),
                                 fixture_reading(), {"action": "stop", "recipe_id": None, "reason": "Synthetic fixture", "reason_codes": ["fixture"]}])
        self.sent = []
        def transport(request, timeout):
            self.sent.append(json.loads(request.data))
            output = queue.pop(0)
            body = {"status": "completed", "model": "test-no-real-api", "usage": {"input_tokens": 11, "output_tokens": 7},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(output)}]}]}
            return HTTPResult(json.dumps(body).encode(), 200, "req_fixture")
        return VLMReader(OpenAIProvider(ProviderConfig(model="test-no-real-api"), transport=transport))

    def experiment(self, *, mock=False, outputs=None, **kwargs):
        with patch.dict(os.environ, {"METER_READER_API_KEY": "fixture-not-real"}), patch("socket.socket", side_effect=AssertionError("No network")):
            path = run_benchmark([self.request()], MockReader() if mock else self.fake_reader(outputs), self.root / "experiments", **kwargs)
        return path, json.loads(path.read_text())

    def test_managed_assets_and_runs_survive_reopen_and_original_deletion(self):
        path = self.run_file()
        raw = path.read_bytes()
        with Repository(self.database) as repository:
            run_id = repository.import_run(path)
            stored = repository.get_run(run_id)
            self.assertEqual(stored["result"], json.loads(raw))
            assets = stored["assets"]
            self.assertTrue(all(set(a) == {"asset_id", "name", "sha256", "size"} for a in assets))
        shutil.rmtree(path.parent)
        self.image.unlink()
        with Repository(self.database) as repository:
            self.assertEqual(repository.get_run(run_id)["result"], json.loads(raw))
            for asset in assets:
                data = repository.get_asset(asset["asset_id"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), asset["sha256"])

    def test_history_is_readable_from_a_new_process(self):
        path = self.run_file()
        with Repository(self.database) as repository:
            repository.import_run(path)
        process = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "src/history_cli.py"),
                                  "--database", str(self.database), "show", "obs1"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["review_status"], "unreviewed")

    def test_import_idempotence_and_changed_run_rejection(self):
        path = self.run_file()
        with Repository(self.database) as repository:
            first = repository.import_run(path)
            self.assertEqual(repository.import_run(path), first)
            self.assertEqual(len(repository.get_observation("obs1")["run_ids"]), 1)
            doc = json.loads(path.read_text())
            doc["reason_codes"].append("tampered")
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                repository.import_run(path)
            self.assertNotIn("tampered", repository.get_run(first)["result"]["reason_codes"])

    def test_reviews_append_without_changing_predictions_and_keep_zero(self):
        path = self.run_file(prediction=fixture_reading())
        raw = path.read_bytes()
        run_id = json.loads(raw)["run_id"]
        with Repository(self.database) as repository:
            repository.import_run(path)
            confirm = ReviewInput(run_id=run_id, action="confirm", candidate_id="c0", reviewer="test", reason="synthetic confirm")
            repository.add_review(confirm)
            repository.add_review(ReviewInput(run_id=run_id, action="correct", value="0", unit="mm", reviewer="test", reason="synthetic zero correction"))
            view = repository.get_observation("obs1")
            self.assertEqual(view["effective_value"], "0")
            self.assertEqual(view["review_status"], "confirmed")
            repository.add_review(ReviewInput(run_id=run_id, action="reject", reviewer="test", reason="synthetic rejection"))
            self.assertEqual(repository.get_observation("obs1")["review_status"], "rejected")
            self.assertIsNone(repository.get_observation("obs1")["effective_value"])
            history = repository.get_run(run_id)
            self.assertEqual(len(history["reviews"]), 3)
            self.assertEqual(history["result"]["reading"]["value"], 12.59)
            self.assertFalse(history["result"]["auto_accept"])
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(self.image.read_bytes(), self.image_bytes)

    def test_new_selected_run_does_not_inherit_human_review(self):
        first = self.run_file(prediction=fixture_reading())
        second = self.run_file()
        r1, r2 = [json.loads(path.read_text())["run_id"] for path in (first, second)]
        with Repository(self.database) as repository:
            repository.import_run(first)
            repository.add_review(ReviewInput(run_id=r1, action="confirm", candidate_id="c0", reviewer="test", reason="synthetic"))
            repository.import_run(second)
            view = repository.get_observation("obs1")
            self.assertEqual(view["run_ids"], [r1, r2])
            self.assertEqual(view["selected_run_id"], r2)
            self.assertEqual(view["review_status"], "unreviewed")
            self.assertIsNone(view["effective_value"])
            repository.select_run("obs1", r1)
            self.assertEqual(repository.get_observation("obs1")["review_status"], "confirmed")

    def test_confirm_unknown_and_invalid_correction_are_rejected(self):
        path = self.run_file()
        run_id = json.loads(path.read_text())["run_id"]
        with Repository(self.database) as repository:
            repository.import_run(path)
            with self.assertRaises(ValueError):
                repository.add_review(ReviewInput(run_id=run_id, action="confirm", candidate_id="c0", reviewer="test", reason="test"))
            with self.assertRaises(ValueError):
                repository.add_review(ReviewInput(run_id=run_id, action="reject", candidate_id="c99", reviewer="test", reason="test"))
        for fields in ({"value": "NaN", "unit": "mm"}, {"value": None, "unit": "mm"}, {"value": 0, "unit": ""}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                ReviewInput(run_id=run_id, action="correct", reviewer="test", reason="test", **fields)

    def test_review_id_is_idempotent_and_sql_cannot_overwrite_runs_or_reviews(self):
        path = self.run_file()
        with Repository(self.database) as repository:
            run_id = repository.import_run(path)
            request = ReviewInput(run_id=run_id, action="reject", reviewer="test", reason="test", review_id="stable-review")
            self.assertEqual(repository.add_review(request), repository.add_review(request))
            with self.assertRaises(ValueError):
                repository.add_review(request.model_copy(update={"reason": "different"}))
            for command in ("UPDATE read_runs SET status='x'", "DELETE FROM read_runs", "DELETE FROM reviews", "UPDATE reviews SET created_at='x'"):
                with self.subTest(command=command), self.assertRaises(sqlite3.IntegrityError):
                    repository.connection.execute(command)

    def test_meter_series_change_preserves_old_observation_series(self):
        with Repository(self.database) as repository:
            repository.register_meter(MeterRecord(meter_id="meter1", name="Gauge", registered_unit="psi"))
            repository.import_run(self.run_file())
            with self.assertRaises(ValueError):
                repository.register_meter(MeterRecord(meter_id="meter1", name="Gauge", registered_unit="bar"))
            repository.register_meter(MeterRecord(meter_id="meter1", name="Replacement", registered_unit="bar", series_version="2"))
            repository.import_run(self.run_file(observation_id="obs2"))
            self.assertEqual(repository.get_observation("obs1")["series_version"], "1")
            self.assertEqual(repository.get_observation("obs2")["series_version"], "2")

    def test_capture_time_is_utc_and_unknown_time_remains_null(self):
        with Repository(self.database) as repository:
            repository.import_run(self.run_file())
            self.assertIsNone(repository.get_observation("obs1")["captured_at"])
            repository.import_run(self.run_file(observation_id="obs2", captured_at="2026-10-27T10:00:00+09:00"))
            self.assertEqual(repository.get_observation("obs2")["captured_at"], "2026-10-27T01:00:00+00:00")

    def test_observation_collision_rolls_back_and_does_not_create_a_second_run(self):
        first = self.run_file()
        with Repository(self.database) as repository:
            repository.import_run(first)
            before = len(list(repository.asset_root.iterdir()))
            with Image.new("RGB", (80, 60), "red") as image:
                image.save(self.image)
            second = self.run_file()
            with self.assertRaises(ValueError):
                repository.import_run(second)
            self.assertEqual(len(repository.get_observation("obs1")["run_ids"]), 1)
            self.assertEqual(len(list(repository.asset_root.iterdir())), before)

    def test_cross_observation_run_selection_is_rejected(self):
        with Repository(self.database) as repository:
            repository.import_run(self.run_file())
            run_id = repository.import_run(self.run_file(observation_id="obs2"))
            with self.assertRaises(ValueError):
                repository.select_run("obs1", run_id)

    def test_path_escape_missing_or_tampered_assets_are_rejected(self):
        for kind in ("escape", "missing", "hash"):
            path = self.run_file()
            doc = json.loads(path.read_text())
            if kind == "escape":
                doc["source_image"]["stored_path"] = "../../outside.jpg"
            elif kind == "missing":
                (path.parent / "roi.png").unlink()
            else:
                (path.parent / "source.png").write_bytes(b"tampered")
            path.write_text(json.dumps(doc))
            with Repository(self.database) as repository, self.subTest(kind=kind), self.assertRaises(ValueError):
                repository.import_run(path)

    def test_managed_asset_integrity_is_checked_on_read(self):
        with Repository(self.database) as repository:
            run_id = repository.import_run(self.run_file())
            asset = repository.get_run(run_id)["assets"][0]
            path = repository.get_asset(asset["asset_id"])
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                repository.get_asset(asset["asset_id"])

    def test_cli_database_flag_and_history_commands(self):
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            status = main([str(self.image), "--meter-id", "meter1", "--observation-id", "obs1", "--database", str(self.database),
                           "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 0)
        path = Path(stdout.getvalue().strip())
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(history_main(["--database", str(self.database), "show", "obs1"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["selected_run_id"], json.loads(path.read_text())["run_id"])

    def test_closed_interval_boundaries_and_no_midpoint_substitution(self):
        label = {"evaluator_kwargs": {"interval": [30, 35], "units": ["psi"]}}
        for value, correct in ((30, True), (35, True), (33, True), (29.9, False), (35.1, False)):
            with self.subTest(value=value):
                score = score_reading(ReaderOutput.model_validate(fixture_reading(value, "psi")), label)
                self.assertEqual(score["value_correct"], correct)
                self.assertEqual(score["truth_interval"], ["30", "35"])
                self.assertIsNone(score["relative_error"])
        score = score_reading(ReaderOutput.model_validate(fixture_reading(29, "psi")), label)
        self.assertEqual(score["absolute_distance_to_interval"], "1.0")

    def test_decimal_sign_point_target_zero_and_unit_magnitude(self):
        label = {"evaluator_kwargs": {"interval": [0, 0], "units": ["Pa"]}}
        score = score_reading(ReaderOutput.model_validate(fixture_reading(0, "Pa")), label)
        self.assertTrue(score["both_correct"])
        self.assertIsNone(score["relative_error"])
        self.assertFalse(score_reading(ReaderOutput.model_validate(fixture_reading(0, "MPa")), label)["unit_correct"])
        signed = {"evaluator_kwargs": {"interval": [-1.2, -1.2], "units": ["mm"]}}
        self.assertFalse(score_reading(ReaderOutput.model_validate(fixture_reading(1.2)), signed)["value_correct"])
        for value in (True, float("nan"), float("inf"), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                number(value)

    def test_safe_unit_aliases_do_not_casefold_magnitude(self):
        label = {"evaluator_kwargs": {"interval": [1, 1], "units": ["kW·h"]}}
        self.assertTrue(score_reading(ReaderOutput.model_validate(fixture_reading(1, "kWh")), label)["unit_correct"])
        label["evaluator_kwargs"]["units"] = ["mW"]
        self.assertFalse(score_reading(ReaderOutput.model_validate(fixture_reading(1, "MW")), label)["unit_correct"])

    def test_label_csv_and_jsonl_preserve_original_evaluator_metadata(self):
        csv_path = self.root / "labels.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question_id", "evaluator", "evaluator_kwargs_json", "truth_lower", "truth_upper", "source_group"])
            writer.writeheader()
            writer.writerow({"question_id": "q1", "evaluator": "interval_matching", "evaluator_kwargs_json": json.dumps(self.label["evaluator_kwargs"]),
                             "truth_lower": 12.59, "truth_upper": 12.59, "source_group": "synthetic-only"})
        left, right = load_labels(self.labels)["q1"], load_labels(csv_path)["q1"]
        self.assertEqual(left["evaluator_kwargs"], right["evaluator_kwargs"])
        self.assertEqual(right["source_group"], "synthetic-only")

    def test_unknown_evaluator_duplicate_labels_or_inconsistent_bounds_fail(self):
        for label in ({**self.label, "evaluator": "unknown"}, {**self.label, "truth_lower": 9},
                      {**self.label, "evaluator_kwargs": {"interval": [2, 1], "units": ["mm"]}}):
            self.labels.write_text(json.dumps(label) + "\n")
            with self.assertRaises(ValueError):
                load_labels(self.labels)
        self.labels.write_text((json.dumps(self.label) + "\n") * 2)
        with self.assertRaises(ValueError):
            load_labels(self.labels)

    def test_benchmark_requires_roi_and_question_id_before_any_api(self):
        class Forbidden:
            mode = "inference"
            def read(self, request):
                raise AssertionError("No call")
        for request in (self.request().model_copy(update={"roi": None}), self.request().model_copy(update={"dataset": None})):
            with self.assertRaises(ValueError):
                run_benchmark([request], Forbidden(), self.root / "experiments")
        self.assertFalse((self.root / "experiments").exists())

    def test_abcd_shared_observation_roi_and_no_labels_in_requests(self):
        path, document = self.experiment()
        self.assertEqual(set(document["conditions"]), set("ABCD"))
        self.assertEqual(document["conditions"]["A"][0]["roi"], None)
        self.assertTrue(all(document["conditions"][c][0]["roi"] == [10, 10, 70, 50] for c in "BCD"))
        self.assertEqual({document["conditions"][c][0]["observation_id"] for c in "ABCD"}, {"obs1"})
        for payload in self.sent:
            text = json.dumps(payload)
            self.assertNotIn("truth_lower", text)
            self.assertNotIn("evaluator", text)
            self.assertNotIn("synthetic-only", text)
        report = evaluate_experiment(path, self.labels)
        self.assertTrue(report["accuracy_evaluated"])
        self.assertFalse(report["official_evaluator_executed"])
        for condition in "ABCD":
            self.assertEqual(report["summaries"][condition]["combined_accuracy"], 1)
            self.assertEqual(report["summaries"][condition]["auto_accept_rate"], 0)
            self.assertIsNone(report["summaries"][condition]["auto_accept_error_rate"])
        self.assertEqual(report["conditions"]["A"][0]["label"], self.label)

    def test_c_uses_fixed_corrected_candidate_not_the_better_baseline(self):
        path, _ = self.experiment(outputs=[fixture_reading(), fixture_reading(), fixture_reading(), fixture_reading(70),
                                          fixture_reading(), {"action": "stop", "recipe_id": None, "reason": "fixture", "reason_codes": []}])
        report = evaluate_experiment(path, self.labels)
        self.assertEqual(report["summaries"]["C"]["combined_accuracy"], 0)
        self.assertEqual(report["conditions"]["C"][0]["predicted_value"], "70.0")

    def test_d_abstention_does_not_choose_best_ground_truth_candidate(self):
        path, _ = self.experiment(outputs=[fixture_reading()] * 5 + [
            {"action": "apply", "recipe_id": "gamma_mild", "reason": "fixture", "reason_codes": []}, fixture_reading(70)])
        report = evaluate_experiment(path, self.labels)
        self.assertEqual(report["summaries"]["D"]["combined_accuracy"], 0)
        self.assertEqual(report["summaries"]["D"]["abstention_rate"], 1)
        self.assertIsNone(report["conditions"]["D"][0]["predicted_value"])

    def test_mock_benchmark_never_reports_numeric_accuracy(self):
        path, _ = self.experiment(mock=True)
        with patch("socket.socket", side_effect=AssertionError("Offline evaluator")):
            report = evaluate_experiment(path, self.labels)
        self.assertFalse(report["accuracy_evaluated"])
        for condition in "ABCD":
            self.assertIsNone(report["summaries"][condition]["value_accuracy"])
            self.assertIsNone(report["summaries"][condition]["combined_accuracy"])
            self.assertIsNone(report["summaries"][condition]["estimated_cost_usd"])
            self.assertEqual(report["summaries"][condition]["api_calls"], 0)

    def test_missing_result_stays_in_common_denominator(self):
        path, document = self.experiment()
        document["conditions"]["C"][0].update(result_path=None, error="fixture missing result")
        path.write_text(json.dumps(document))
        report = evaluate_experiment(path, self.labels)
        summary = report["summaries"]["C"]
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["missing_results"], 1)
        self.assertEqual(summary["combined_accuracy"], 0)
        self.assertEqual(summary["technical_error_rate"], 1)

    def test_incomparable_conditions_or_tampered_result_are_rejected(self):
        path, original = self.experiment()
        for kind in ("roi", "cases", "hash", "selection", "question"):
            document = json.loads(json.dumps(original))
            row = document["conditions"]["C"][0]
            if kind == "roi":
                row["roi"] = [0, 0, 1, 1]
            elif kind == "cases":
                document["conditions"]["C"] = []
            elif kind == "selection":
                row["evaluation_candidate_id"] = "c0"
            elif kind == "question":
                row["question"] = "wrong display"
            else:
                row["result_sha256"] = "0" * 64
            path.write_text(json.dumps(document))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                evaluate_experiment(path, self.labels)

    def test_wrong_label_image_hash_and_mixed_mock_results_are_rejected(self):
        path, document = self.experiment()
        self.label["image_sha256"] = "0" * 64
        self.labels.write_text(json.dumps(self.label))
        with self.assertRaises(ValueError):
            evaluate_experiment(path, self.labels)
        self.label["image_sha256"] = hashlib.sha256(self.image_bytes).hexdigest()
        self.labels.write_text(json.dumps(self.label))
        document["mode"] = "mock"
        path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            evaluate_experiment(path, self.labels)

    def test_benchmark_and_history_cli_run_in_subprocesses(self):
        repo_root = Path(__file__).resolve().parents[1]
        process = subprocess.run([sys.executable, "src/benchmark.py", str(self.image), "--question-id", "q1",
                                  "--roi", "10", "10", "70", "50", "--database", str(self.database),
                                  "--output-dir", str(self.root / "experiments")], cwd=repo_root, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        manifest = Path(process.stdout.strip())
        process = subprocess.run([sys.executable, "src/evaluation.py", "--experiment", str(manifest), "--labels", str(self.labels),
                                  "--output-dir", str(self.root / "evaluations")], cwd=repo_root, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertFalse(json.loads(Path(process.stdout.strip()).read_text())["accuracy_evaluated"])

    def test_asset_write_failure_rolls_back_database_and_owned_directory(self):
        path = self.run_file()
        original_open = Path.open
        def failing_open(target, mode="r", *args, **kwargs):
            if mode == "xb" and target.name == "roi.png":
                raise OSError("Synthetic asset write failure")
            return original_open(target, mode, *args, **kwargs)
        with Repository(self.database) as repository, patch.object(Path, "open", failing_open):
            with self.assertRaises(OSError):
                repository.import_run(path)
            self.assertEqual(repository.list_observations(), [])
            self.assertEqual(repository.list_meters(), [])
            self.assertEqual(list(repository.asset_root.iterdir()), [])

    def test_symlink_escape_is_rejected_before_copying(self):
        path = self.run_file()
        outside = self.root / "outside.png"
        outside.write_bytes((path.parent / "roi.png").read_bytes())
        (path.parent / "roi.png").unlink()
        (path.parent / "roi.png").symlink_to(outside)
        with Repository(self.database) as repository, self.assertRaises(ValueError):
            repository.import_run(path)

    def test_manual_scale_correction_keeps_original_scale_and_source(self):
        path = self.run_file(prediction=fixture_reading())
        scale = ScaleInfo(min_value=0, max_value=60, unit="psi", status="readable", source="manual")
        with Repository(self.database) as repository:
            run_id = repository.import_run(path)
            repository.add_review(ReviewInput(run_id=run_id, action="correct", value="1.00", unit="psi", scale=scale,
                                             reviewer="test", reason="synthetic scale correction"))
            run = repository.get_run(run_id)
            self.assertEqual(run["reviews"][-1]["resolution"]["scale"]["source"], "manual")
            self.assertEqual(run["result"]["reading"]["scale"]["source"], "none")
            self.assertEqual(run["result"]["reading"]["value"], 12.59)

    def test_keep_selected_preserves_existing_reviewed_run(self):
        with Repository(self.database) as repository:
            first = repository.import_run(self.run_file(prediction=fixture_reading()))
            repository.add_review(ReviewInput(run_id=first, action="confirm", candidate_id="c0", reviewer="test", reason="synthetic"))
            second = repository.import_run(self.run_file(), select=False)
            self.assertNotEqual(first, second)
            self.assertEqual(repository.get_observation("obs1")["selected_run_id"], first)
            self.assertEqual(repository.get_observation("obs1")["review_status"], "confirmed")

    def test_unsupported_database_version_is_not_overwritten(self):
        with sqlite3.connect(self.database) as connection:
            connection.execute("PRAGMA user_version=999")
        with self.assertRaises(ValueError):
            Repository(self.database)
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 999)

    def test_c_recipe_and_reader_settings_must_match_declared_experiment(self):
        path, original = self.experiment()
        document = json.loads(json.dumps(original))
        document["fixed_recipe"] = "clahe_mild"
        path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            evaluate_experiment(path, self.labels)
        document = original
        row = document["conditions"]["C"][0]
        run_path = path.parent / row["result_path"]
        result = json.loads(run_path.read_text())
        result["candidates"][1]["inference"]["image_detail"] = "low"
        raw = json.dumps(result).encode()
        run_path.write_bytes(raw)
        row["result_sha256"] = hashlib.sha256(raw).hexdigest()
        path.write_text(json.dumps(document))
        with self.assertRaises(ValueError):
            evaluate_experiment(path, self.labels)

    def test_benchmark_technical_reader_error_produces_nonzero_and_saved_manifest(self):
        reader = self.fake_reader([fixture_reading()] * 4 + [fixture_reading(), {}])
        with patch("benchmark.configured_reader", return_value=reader), patch.dict(os.environ, {"METER_READER_API_KEY": "fixture"}), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            code = benchmark_main([str(self.image), "--question-id", "q1", "--roi", "10", "10", "70", "50",
                                   "--mode", "inference", "--model", "test-no-real-api", "--output-dir", str(self.root / "experiments")])
        self.assertEqual(code, 1)
        document = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertTrue(document["conditions"]["D"][0]["technical_error"])

    def test_benchmark_database_failure_is_detected_before_network(self):
        bad = self.root / "directory.sqlite3"
        bad.mkdir()
        with self.assertRaises(sqlite3.Error):
            run_benchmark([self.request()], self.fake_reader(), self.root / "experiments", database=bad)
        self.assertEqual(self.sent, [])

    def test_old_registered_unit_is_preserved_and_series_versions_cannot_be_repurposed(self):
        with Repository(self.database) as repository:
            repository.register_meter(MeterRecord(meter_id="meter1", name="Original", registered_unit="psi"))
            repository.import_run(self.run_file())
            repository.register_meter(MeterRecord(meter_id="meter1", name="Replacement", registered_unit="bar", series_version="2"))
            repository.import_run(self.run_file(observation_id="obs2"))
            self.assertEqual(repository.get_observation("obs1")["meter"]["registered_unit"], "psi")
            self.assertEqual(repository.get_observation("obs2")["meter"]["registered_unit"], "bar")
            with self.assertRaises(ValueError):
                repository.register_meter(MeterRecord(meter_id="meter1", name="Invalid reused version", registered_unit="MPa", series_version="1"))

    def test_review_exposes_effective_scale_without_overwriting_read_status(self):
        path = self.run_file()
        with Repository(self.database) as repository:
            run_id = repository.import_run(path)
            repository.add_review(ReviewInput(run_id=run_id, action="correct", value="1", unit="psi", reviewer="test", reason="synthetic correction",
                                             scale=ScaleInfo(min_value=0, max_value=60, unit="psi", source="manual", status="readable")))
            view = repository.get_observation("obs1")
            self.assertEqual(view["read_status"], "needs_review")
            self.assertEqual(view["review_status"], "confirmed")
            self.assertEqual(view["effective_scale"]["source"], "manual")
            self.assertTrue(view["corrected"])

    def test_multi_interval_values_are_paired_with_their_own_units(self):
        label = {"evaluator": "multi_interval_matching", "evaluator_kwargs": {
            "intervals": [[1.5, 2], [20, 30]], "units": [["bar"], ["psi"]]}}
        for value, unit_name, correct in ((1.8, "bar", True), (25, "psi", True), (25, "bar", False), (1.8, "psi", False), (25, "MPa", False)):
            with self.subTest(value=value, unit=unit_name):
                score = score_reading(ReaderOutput.model_validate(fixture_reading(value, unit_name)), label)
                self.assertEqual(score["both_correct"], correct)
                self.assertIsNone(score["truth_interval"])
                self.assertEqual(len(score["truth_alternatives"]), 2)
        self.labels.write_text(json.dumps({"question_id": "q1", **label}) + "\n")
        self.assertEqual(load_labels(self.labels)["q1"]["evaluator_kwargs"], label["evaluator_kwargs"])

    def test_multi_interval_misaligned_groups_are_rejected(self):
        label = {"question_id": "q1", "evaluator": "multi_interval_matching",
                 "evaluator_kwargs": {"intervals": [[1, 2], [3, 4]], "units": [["bar"]]}}
        self.labels.write_text(json.dumps(label))
        with self.assertRaises(ValueError):
            load_labels(self.labels)
