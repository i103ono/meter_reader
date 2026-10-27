"""Synthetic landmarks and mocked HTTP only; no claim of visual accuracy."""
from contextlib import redirect_stdout
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analog_diagnostics import diagnose_analog, guard_analog_reading
from benchmark import run_benchmark
from evaluation import evaluate_experiment, main as evaluate_main
from provider import HTTPResult, OpenAIProvider, ProviderConfig
from schemas import AnalogTrace, ObservationInput, ReaderOutput, RunResult, ScaleInfo
from vlm_reader import VLMReader, parse_reading, structured_schema, PROMPT_VERSION


def trace(**changes):
    data = {
        **{stage: {"status": "visible", "evidence": "Fabricated visible test observation."}
           for stage in ("target", "geometry", "scale", "needle", "interpolation")},
        "view": "front_facing", "center": [0.5, 0.5], "tip": [0.5, 0.1],
        "counterweight": [0.5, 0.7],
        "anchors": [{"value": 20, "raw_text": "20", "tick_point": [0.3, 0.1]},
                    {"value": 40, "raw_text": "40", "tick_point": [0.7, 0.1]}],
        "scale_unit": "psi", "interpolation_mode": "linear", "fraction": 0.5,
    }
    data.update(changes)
    return AnalogTrace.model_validate(data)


def reading(value=30, **trace_changes):
    return ReaderOutput(meter_type="analog", value=value, unit="psi", value_status="readable",
                        unit_status="readable", scale=ScaleInfo(min_value=0, max_value=200,
                        minor_division=1, unit="psi", status="readable", source="image"),
                        analog_trace=trace(**trace_changes))


class AnalogDiagnosticsTests(unittest.TestCase):
    def test_arithmetic_is_a_check_not_a_silent_numeric_repair(self):
        good = guard_analog_reading(reading(), (100, 100))
        self.assertEqual(good.value_status, "readable")
        self.assertEqual(diagnose_analog(good, (100, 100))["interpolated_value"], "30.00")
        bad = guard_analog_reading(reading(165), (100, 100))
        self.assertEqual(bad.value, 165)
        self.assertEqual(bad.value_status, "uncertain")
        self.assertIn("analog_interpolation_value_conflict", bad.issue_codes)

    def test_reversed_tip_is_detected_without_a_ground_truth_reading(self):
        bad = guard_analog_reading(reading(tip=[0.5, 0.9]), (100, 100))
        self.assertIn("analog_tip_outside_anchor_arc", bad.issue_codes)
        self.assertIn("analog_pointer_counterweight_conflict", bad.issue_codes)
        self.assertEqual(bad.value_status, "uncertain")

    def test_oblique_or_unknown_geometry_does_not_use_image_angles(self):
        for view in ("oblique", "unknown"):
            candidate = reading(tip=[0.5, 0.9], counterweight=None, view=view,
                                geometry={"status": "uncertain", "evidence": "Synthetic oblique view."})
            diagnostic = diagnose_analog(candidate, (100, 100))
            self.assertEqual(diagnostic["reported_problem_stages"], ["geometry"])
            self.assertNotIn("analog_tip_outside_anchor_arc", diagnostic["consistency_findings"])
            self.assertEqual(guard_analog_reading(candidate, (100, 100)).value_status, "readable")

    def test_angle_check_restores_non_square_image_coordinates_and_wraps(self):
        def point(angle):
            angle = math.radians(angle)
            # A 400x100 Reader image; x/y normalization changes angles.
            return [0.5 + math.cos(angle) * 0.1, 0.5 + math.sin(angle) * 0.4]
        anchors = [{"value": 20, "raw_text": "20", "tick_point": point(170)},
                   {"value": 40, "raw_text": "40", "tick_point": point(-170)}]
        candidate = reading(tip=point(180), counterweight=None, anchors=anchors)
        self.assertEqual(diagnose_analog(candidate, (400, 100))["consistency_findings"], [])

    def test_degenerate_landmarks_and_equal_anchors_are_not_supported(self):
        candidate = reading(tip=[0.5, 0.5], anchors=[
            {"value": 20, "raw_text": "20", "tick_point": [0.3, 0.1]},
            {"value": 20, "raw_text": "20", "tick_point": [0.7, 0.1]}])
        checked = guard_analog_reading(candidate, (100, 100))
        self.assertIn("analog_degenerate_pointer", checked.issue_codes)
        self.assertIn("analog_degenerate_anchors", checked.issue_codes)

    def test_descending_negative_and_rounding_cases(self):
        anchors = [{"value": 40, "raw_text": "40", "tick_point": [0.3, 0.1]},
                   {"value": 20, "raw_text": "20", "tick_point": [0.7, 0.1]}]
        self.assertEqual(guard_analog_reading(reading(33, anchors=anchors, fraction=0.35)).value_status, "readable")
        anchors[0].update(value=-1, raw_text="-1")
        anchors[1].update(value=1, raw_text="1")
        self.assertEqual(guard_analog_reading(reading(-0.6, anchors=anchors, fraction=0.2)).value_status, "readable")
        self.assertEqual(guard_analog_reading(reading(30.5)).value_status, "readable")
        self.assertEqual(guard_analog_reading(reading(31)).value_status, "uncertain")

    def test_unknown_nonlinear_scales_are_not_forced_into_an_interpolator(self):
        candidate = reading(interpolation_mode="unknown", fraction=None, anchors=None)
        diagnostic = diagnose_analog(candidate)
        self.assertIsNone(diagnostic["interpolated_value"])
        self.assertEqual(diagnostic["consistency_findings"], [])
        candidate = reading(fraction=None, anchors=None)
        self.assertIn("analog_interpolation_evidence_missing", guard_analog_reading(candidate).issue_codes)

    def test_model_reported_problem_and_programmatic_conflict_remain_separate(self):
        candidate = reading(scale={"status": "uncertain", "evidence": "Two synthetic overlapping scales."})
        diagnostic = diagnose_analog(candidate)
        self.assertEqual(diagnostic["first_reported_problem_stage"], "scale")
        self.assertEqual(diagnostic["consistency_findings"], [])
        self.assertEqual(guard_analog_reading(candidate).value_status, "uncertain")
        self.assertIn("analog_scale_unresolved", guard_analog_reading(candidate).issue_codes)

    def test_unit_conflict_is_not_resolved_by_an_implicit_conversion(self):
        checked = guard_analog_reading(reading(scale_unit="bar"))
        self.assertIn("analog_scale_unit_conflict", checked.issue_codes)
        self.assertEqual((checked.value, checked.unit), (30, "psi"))
        self.assertEqual(checked.unit_status, "uncertain")

    def test_anchor_decimal_and_sign_transcription_mismatches_are_flagged(self):
        anchors = [{"value": 1, "raw_text": "0.1", "tick_point": [0.3, 0.1]},
                   {"value": 2, "raw_text": "2", "tick_point": [0.7, 0.1]}]
        checked = guard_analog_reading(reading(1.5, anchors=anchors))
        self.assertIn("analog_anchor_text_value_conflict", checked.issue_codes)
        self.assertEqual(checked.value_status, "uncertain")
        anchors[0].update(value=-1, raw_text="−1")
        self.assertNotIn("analog_anchor_text_value_conflict", guard_analog_reading(reading(0.5, anchors=anchors)).issue_codes)

    def test_legacy_records_load_but_new_missing_evidence_is_uncertain(self):
        legacy = reading().model_dump()
        legacy.pop("analog_trace")
        loaded = ReaderOutput.model_validate(legacy)
        self.assertIsNone(loaded.analog_trace)
        self.assertFalse(diagnose_analog(loaded)["trace_available"])
        legacy["analog_trace"] = None  # Explicit null from a new response.
        parsed = parse_reading(json.dumps(legacy))
        self.assertEqual(parsed.value_status, "uncertain")
        self.assertIn("analog_trace_missing", parsed.issue_codes)

    def test_schema_is_strict_and_rejects_bad_coordinates_and_digital_traces(self):
        schema = structured_schema()
        self.assertNotIn("prefixItems", json.dumps(schema))
        self.assertEqual(schema["$defs"]["AnalogTrace"]["additionalProperties"], False)
        for change in ({"tip": [1.1, 0.1]}, {"fraction": -0.1}, {"center": [True, 0.5]},
                       {"needle": {"status": "visible", "evidence": ""}}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                trace(**change)
        with self.assertRaises(ValidationError):
            ReaderOutput(meter_type="digital", analog_trace=trace())

    def test_mocked_abcd_exports_unselected_candidates_and_preserves_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "input.png"
            with Image.new("RGB", (120, 80), "white") as pixels:
                pixels.save(image)
            # A/B correct; C wrong; D keeps both conflicting candidates and
            # abstains. Diagnostics must not select using the label.
            outputs = [reading().model_dump()] * 3 + [reading(165).model_dump(), reading().model_dump(),
                       {"action": "apply", "recipe_id": "resize_2x", "reason": "Synthetic test choice.", "reason_codes": []},
                       reading(165).model_dump()]
            sent = []
            def transport(request, timeout):
                payload = json.loads(request.data)
                sent.append(payload)
                document = {"status": "completed", "model": "test-only",
                            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(outputs.pop(0))}]}]}
                return HTTPResult(json.dumps(document).encode(), 200)
            reader = VLMReader(OpenAIProvider(ProviderConfig(model="test-only"), transport=transport))
            request = ObservationInput(image_path=image, roi=[5, 5, 115, 75], dataset={"question_id": "test1"})
            labels = root / "labels.jsonl"
            labels.write_text(json.dumps({"question_id": "test1", "evaluator": "interval_matching",
                                          "evaluator_kwargs": {"interval": [30, 30], "units": ["psi"]}}) + "\n")
            with patch.dict(os.environ, {"METER_READER_API_KEY": "test-only"}), patch("socket.socket", side_effect=AssertionError("No live API")):
                experiment = run_benchmark([request], reader, root / "experiments")
            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(evaluate_main(["--experiment", str(experiment), "--labels", str(labels),
                                               "--output-dir", str(root / "evaluations")]), 0)
            evaluation_path = Path(stdout.getvalue().strip())
            report = json.loads(evaluation_path.read_text())
            self.assertEqual(report["summaries"]["A"]["combined_accuracy"], 1)
            self.assertEqual(report["summaries"]["C"]["combined_accuracy"], 0)
            self.assertEqual(report["summaries"]["D"]["abstention_rate"], 1)
            d = report["conditions"]["D"][0]
            self.assertEqual(d["agent_stop_reason"], "candidate_conflict")
            self.assertEqual(len(d["candidate_diagnostics"]), 2)
            self.assertTrue(all(not c["is_evaluation_candidate"] for c in d["candidate_diagnostics"]))
            self.assertIn("analog_interpolation_value_conflict", d["candidate_diagnostics"][1]["analog_diagnostic"]["consistency_findings"])
            with (evaluation_path.parent / "diagnostics.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual([r["value"] for r in rows if r["condition"] == "D"], ["30.0", "165.0"])
            manifest = json.loads(experiment.read_text())
            result_path = experiment.parent / manifest["conditions"]["A"][0]["result_path"]
            raw = json.loads(result_path.read_text())
            RunResult.model_validate(raw)
            self.assertEqual(raw["inference"]["prompt_version"], PROMPT_VERSION)
            self.assertEqual(raw["api_calls"], 1)  # No extra diagnostic API call.
            with Image.open(image) as pixels:
                self.assertEqual(pixels.size, (120, 80))
            self.assertNotIn("evaluator_kwargs", json.dumps(sent))


if __name__ == "__main__":
    unittest.main()
