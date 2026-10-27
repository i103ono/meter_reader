"""Bounded planning with fabricated HTTP only; not an image-reading benchmark."""
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import main
from image_io import ImageLoadError
from planner import MockComparator, MockPlanner, VLMComparator, VLMPlanner
from provider import HTTPResult, OpenAIProvider, ProviderConfig, TokenPricing, TransportFailure
from reader import MockReader
from recording import run_observation
from replay_corrections import replay
from roi_locator import ROILocator
from schemas import ObservationInput, ReaderOutput, RunResult, ScaleInfo
from vlm_reader import VLMReader

MODEL = "stage4-offline-fixture-not-a-real-model"
SECRET = "fixture-not-real-key"


def response(document, **changes):
    body = {
        "id": "resp_fixture", "model": MODEL, "status": "completed",
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(document)}]}],
    }
    body.update(changes)
    return HTTPResult(json.dumps(body).encode(), 200, "req_fixture")


def reading(value=None, unit=None, kind="analog", raw=None, issues=()):
    return ReaderOutput(meter_type=kind, value=value, unit=unit, raw_value_text=raw,
                        value_status="readable" if value is not None else "unreadable",
                        unit_status="readable" if unit is not None else "unreadable",
                        issue_codes=list(issues)).model_dump()


def plan(action="apply", recipe="resize_gamma"):
    return {"action": action, "recipe_id": recipe if action == "apply" else None,
            "reason": "Fabricated test-only planning decision", "reason_codes": ["fixture"]}


def comparison(support="c1", **changes):
    return {"supported_candidate_id": support, "readable": support is not None,
            "contradictions": [], "reason": "Fabricated test-only comparison", **changes}


class Clock:
    now = 0.0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class Stage4Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "meter.png"
        y, x = np.indices((96, 128))
        with Image.fromarray(np.stack((x, y, x + y), axis=-1).astype(np.uint8)) as image:
            image.save(self.image)
        self.original = self.image.read_bytes()
        self.sent = []
        self.clock = Clock()
        env = patch.dict(os.environ, {"METER_READER_API_KEY": SECRET})
        env.start()
        self.addCleanup(env.stop)

    def clients(self, outcomes, config=None, transport=None):
        queue = list(outcomes)
        def queued(request, timeout):
            self.sent.append((json.loads(request.data), timeout))
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        provider = OpenAIProvider(config or ProviderConfig(model=MODEL), transport=transport or queued,
                                  clock=self.clock, sleep=self.clock.sleep)
        return VLMReader(provider), VLMPlanner(provider), VLMComparator(provider)

    def execute(self, outputs, config=None, *, auto_roi=False, **fields):
        reader, planner, comparator = self.clients(outputs, config)
        path = run_observation(ObservationInput(image_path=self.image, **fields), reader, self.root / "runs",
                               planner=planner, comparator=comparator,
                               roi_locator=ROILocator(reader.provider) if auto_roi else None)
        result = json.loads(path.read_text())
        RunResult.model_validate(result)
        return path, result, reader

    def test_two_planned_candidates_are_independent_and_replay_offline(self):
        outputs = [response(reading()), response(plan()), response(reading()),
                   response(plan(recipe="clahe_mild")), response(reading())]
        path, result, _ = self.execute(outputs, roi=[10, 10, 100, 80], question="Read psi")
        self.assertEqual(result["stage"], 4)
        self.assertEqual(result["api_calls"], 5)
        self.assertEqual(result["correction_candidates_generated"], 2)
        self.assertEqual(result["agent"]["stop_reason"], "candidate_limit")
        self.assertEqual([c["image_size"] for c in result["candidates"]], [[90, 70], [180, 140], [90, 70]])
        self.assertTrue(all(c["parent_candidate_id"] == "c0" for c in result["candidates"][1:]))
        self.assertTrue(all(c["input_roi_sha256"] == result["candidates"][0]["image_sha256"] for c in result["candidates"]))
        self.assertEqual(result["status"], "unreadable")
        self.assertFalse(result["auto_accept"])
        self.assertIsNone(result["effective_value"])
        self.assertEqual(result["reading"], result["candidates"][0]["reading"])
        self.assertEqual(self.image.read_bytes(), self.original)
        with patch("socket.socket", side_effect=AssertionError("Offline replay")):
            report = json.loads(replay(path, self.root / "replay").read_text())
        self.assertTrue(report["all_pixels_match"])
        self.assertTrue(all(c["file_hash_matches"] for c in report["candidates"]))

    def test_planner_history_is_current_run_only_and_reader_remains_independent(self):
        path, result, _ = self.execute(
            [response(reading()), response(plan()), response(reading()),
             response(plan(recipe="clahe_mild")), response(reading())],
            question="Read the upper display", meter_id="private-meter", captured_at="2026-10-27T10:00:00+09:00",
            dataset={"source": {"truth_value": 987654}}, acceptance_policy={"previous_value": 765432})
        plans, reads = [], []
        for payload, _ in self.sent:
            text = json.dumps(payload)
            for hidden in ("987654", "765432", "truth_value", "previous_value", "private-meter", "2026-10-27"):
                self.assertNotIn(hidden, text)
            target = json.loads(payload["input"][0]["content"][0]["text"])
            if payload["text"]["format"]["name"] == "meter_planner_v1":
                plans.append(target)
            else:
                reads.append(target)
                self.assertEqual(set(target), {"question", "target_type"})
        self.assertEqual([len(p["history"]) for p in plans], [1, 2])
        self.assertEqual(plans[1]["tried_recipes"], ["resize_gamma"])
        self.assertNotIn("resize_gamma", plans[1]["recipes"])
        self.assertTrue(all(r == reads[0] for r in reads))
        self.assertNotIn(SECRET, path.read_text())

    def test_stop_and_recapture_do_not_create_corrections(self):
        for action in ("stop", "recommend_recapture"):
            with self.subTest(action=action):
                _, result, _ = self.execute([response(reading(20, "psi")), response(plan(action))])
                self.assertEqual(len(result["candidates"]), 1)
                self.assertEqual(result["api_calls"], 2)
                self.assertEqual(result["agent"]["stop_reason"], "planner_" + action)
                self.assertEqual(result["next_action"], "recapture" if action != "stop" else "human_review")
                self.assertEqual(result["selected_candidate_id"], "c0" if action == "stop" else None)

    def test_unknown_and_executable_recipe_text_are_rejected_without_execution(self):
        for recipe in ("not_registered", "__import__('os').system('bad')", "gamma_mild;bad"):
            with self.subTest(recipe=recipe), patch("candidate_io.apply_recipe", side_effect=AssertionError("Must not execute")):
                path, result, _ = self.execute([response(reading()), response(plan(recipe=recipe))])
            self.assertEqual(result["agent"]["stop_reason"], "invalid_planner_decision")
            self.assertEqual(result["correction_candidates_generated"], 0)
            self.assertEqual(result["api_calls"], 2)
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["agent"]["events"][0]["status"], "error")
            self.assertFalse((path.parent / "candidates").exists())

    def test_repeated_recipe_is_not_executed_twice(self):
        _, result, _ = self.execute([response(reading()), response(plan()), response(reading()), response(plan())])
        self.assertEqual(result["agent"]["stop_reason"], "invalid_planner_decision")
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(result["api_calls"], 4)
        self.assertEqual(result["agent"]["events"][-2]["inference"]["error_code"], "invalid_planner_decision")

    def test_malformed_or_incoherent_plans_are_not_repaired(self):
        invalid = [{}, {**plan(), "action": "execute"}, {**plan("stop"), "recipe_id": "gamma_mild"},
                   {**plan(), "recipe_id": None}, {**plan(), "reason": " "}, {**plan(), "value": 123}]
        for document in invalid:
            with self.subTest(document=document):
                _, result, _ = self.execute([response(reading()), response(document)])
                self.assertEqual(result["api_calls"], 2)
                self.assertEqual(result["agent"]["stop_reason"], "invalid_planner_decision")
                self.assertIsNotNone(result["agent"]["events"][0]["inference"]["attempts"][0]["raw_output"])

    def test_disagreeing_readings_stop_without_vote_or_final_comparison(self):
        _, result, _ = self.execute([response(reading(20, "psi")), response(plan()), response(reading(40, "psi"))])
        self.assertEqual(result["agent"]["stop_reason"], "candidate_conflict")
        self.assertIn("value_conflict", result["reason_codes"])
        self.assertIsNone(result["selected_candidate_id"])
        self.assertEqual(result["api_calls"], 3)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["reading"]["value"], 20)
        self.assertEqual(result["agent"]["events"][-1]["reason_codes"], ["comparison_skipped_conflict"])

    def test_unit_magnitude_conflict_is_not_normalized_or_converted(self):
        _, result, _ = self.execute([response(reading(1, "Pa")), response(plan()), response(reading(1, "MPa"))])
        self.assertIn("unit_conflict", result["reason_codes"])
        self.assertIsNone(result["selected_candidate_id"])

    def test_analog_tolerance_is_explicit_and_tied_to_its_unit(self):
        for policy, conflict in [(None, True), ({"analog_absolute_tolerance": "0.02", "unit": "MPa"}, False),
                                 ({"analog_absolute_tolerance": "0.02", "unit": "psi"}, True)]:
            with self.subTest(policy=policy):
                _, result, _ = self.execute([response(reading(0.4, "MPa")), response(plan()), response(reading(0.42, "MPa"))],
                                            acceptance_policy={"comparison": policy} if policy else None)
                self.assertEqual("value_conflict" in result["reason_codes"], conflict)
                self.assertFalse(result["auto_accept"])
                self.assertEqual(result["selected_candidate_id"], None if conflict else "c0")

    def test_digital_decimal_sign_and_raw_text_are_preserved(self):
        _, same, _ = self.execute([response(reading(0.1, "kWh", "digital", "000.10")), response(plan()),
                                   response(reading(0.10, "kWh", "digital", "+0.100"))])
        self.assertNotIn("value_conflict", same["reason_codes"])
        self.assertEqual(same["selected_candidate_id"], "c0")
        _, opposite, _ = self.execute([response(reading(-1.2, "kWh", "digital", "-1.20")), response(plan()),
                                       response(reading(1.2, "kWh", "digital", "1.20"))])
        self.assertIn("value_conflict", opposite["reason_codes"])
        self.assertIsNone(opposite["selected_candidate_id"])

    def test_raw_value_inconsistent_with_numeric_value_blocks_selection(self):
        _, result, _ = self.execute([response(reading(1.2, "kWh", "digital", "12")), response(plan("stop"))])
        self.assertIn("raw_value_mismatch", result["reason_codes"])
        self.assertIsNone(result["selected_candidate_id"])

    def test_final_comparison_can_support_existing_corrected_candidate_only(self):
        path, result, _ = self.execute([response(reading()), response(plan()), response(reading(20, "psi")), response(comparison())])
        self.assertEqual(result["selected_candidate_id"], "c1")
        self.assertEqual(result["api_calls"], 4)
        self.assertEqual(result["status"], "needs_review")
        self.assertIsNone(result["effective_value"])
        self.assertFalse(result["auto_accept"])
        self.assertIsNone(result["reading"]["value"])
        compare_payload = self.sent[-1][0]
        self.assertEqual(compare_payload["text"]["format"]["name"], "meter_comparison_v1")
        self.assertEqual(len([c for c in compare_payload["input"][0]["content"] if c["type"] == "input_image"]), 2)
        target = json.loads(compare_payload["input"][0]["content"][0]["text"])
        self.assertEqual(target["image_order"], ["c0", "c1"])
        self.assertEqual(result["agent"]["events"][-1]["comparison"]["supported_candidate_id"], "c1")

    def test_final_comparison_rejects_new_values_or_unavailable_ids(self):
        for document in (comparison("c99"), {**comparison(), "value": 123}, comparison(readable=False),
                         comparison(contradictions=["conflict"])):
            with self.subTest(document=document):
                _, result, _ = self.execute([response(reading()), response(plan()), response(reading(20, "psi")), response(document)])
                self.assertIsNone(result["selected_candidate_id"])
                self.assertEqual(result["agent"]["events"][-1]["error_code"], "invalid_comparison")
                self.assertEqual(result["api_calls"], 4)

    def test_final_comparison_contradictions_or_no_support_remain_review(self):
        for document in (comparison(None, contradictions=["unreadable pointer"]), comparison(None), comparison("c0")):
            with self.subTest(document=document):
                _, result, _ = self.execute([response(reading()), response(plan()), response(reading(20, "psi")), response(document)])
                self.assertIsNone(result["selected_candidate_id"])
                self.assertEqual(result["status"], "needs_review")

    def test_partial_unit_and_ambiguous_target_never_become_a_selected_reading(self):
        for candidate in (reading(20), reading(20, "psi", issues=["ambiguous_scale"])):
            with self.subTest(candidate=candidate):
                _, result, _ = self.execute([response(candidate), response(plan("stop"))])
                self.assertIsNone(result["selected_candidate_id"])
                self.assertEqual(result["status"], "needs_review")

    def test_baseline_api_failure_stops_before_planning(self):
        _, result, _ = self.execute([TransportFailure("authentication_error", status=401)])
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["agent"]["stop_reason"], "baseline_error")
        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(result["agent"]["events"][0]["status"], "skipped")

    def test_planner_api_failure_preserves_original_observation(self):
        _, result, _ = self.execute([response(reading(20, "psi")), TransportFailure("permission_error", status=403)])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["reading"]["value"], 20)
        self.assertEqual(result["agent"]["stop_reason"], "permission_error")
        self.assertIsNone(result["error"])

    def test_candidate_api_failure_is_not_image_unreadability(self):
        _, result, _ = self.execute([response(reading()), response(plan()), TransportFailure("authentication_error", status=401)])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["agent"]["stop_reason"], "authentication_error")
        self.assertEqual(result["candidates"][1]["status"], "error")

    def test_correction_failure_is_recorded_without_reading_nonexistent_image(self):
        with patch("tools.MAX_OUTPUT_PIXELS", 100):
            _, result, _ = self.execute([response(reading()), response(plan())])
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["agent"]["stop_reason"], "correction_error")
        self.assertIsNone(result["candidates"][1]["image_path"])

    def test_final_compare_at_sixth_call_and_budget_block_with_auto_roi(self):
        outputs = [response(reading()), response(plan()), response(reading()), response(plan(recipe="clahe_mild")),
                   response(reading(20, "psi")), response(comparison("c2"))]
        _, result, _ = self.execute(outputs)
        self.assertEqual(result["api_calls"], 6)
        self.assertEqual(result["selected_candidate_id"], "c2")
        roi = {"status": "located", "roi": [10, 10, 100, 80], "target_description": "Test fixture only",
               "reason": None, "issue_codes": []}
        _, limited, _ = self.execute([response(roi), *outputs[:-1]], auto_roi=True)
        self.assertEqual(limited["api_calls"], 6)
        self.assertIsNone(limited["selected_candidate_id"])
        self.assertEqual(limited["agent"]["events"][-1]["status"], "skipped")
        self.assertIn("final_comparison_unavailable", limited["reason_codes"])
        self.assertEqual(limited["agent"]["events"][-1]["error_code"], "observation_api_call_limit")

    def test_transient_retries_share_six_call_limit_and_skip_second_plan(self):
        failure = TransportFailure("rate_limit_error", status=429)
        _, result, reader = self.execute([failure, response(reading()), failure, response(plan()), failure, response(reading())])
        self.assertEqual(result["api_calls"], 6)
        self.assertEqual(reader.provider.observation_api_calls, 6)
        self.assertEqual(result["agent"]["stop_reason"], "observation_api_call_limit")
        self.assertEqual(result["agent"]["events"][-2]["status"], "skipped")
        self.assertEqual(result["correction_candidates_generated"], 1)

    def test_deadline_shared_by_reader_and_planner_and_checked_before_correction(self):
        timeouts = []
        def transport(request, timeout):
            timeouts.append(timeout)
            self.clock.now += 60
            return response(reading() if len(timeouts) == 1 else plan())
        reader, planner, comparator = self.clients([], transport=transport)
        with patch("candidate_io.apply_recipe", side_effect=AssertionError("Expired before correction")):
            path = run_observation(ObservationInput(image_path=self.image), reader, self.root / "runs",
                                   planner=planner, comparator=comparator)
        result = json.loads(path.read_text())
        self.assertEqual(timeouts, [30, 30])
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["agent"]["stop_reason"], "deadline_exceeded")
        self.assertIsNone(result["candidates"][1]["image_path"])

    def test_cost_budget_shared_by_roles_and_checked_before_correction(self):
        pricing = TokenPricing(model=MODEL, input_usd_per_million="1", output_usd_per_million="2",
                               max_input_tokens_per_call=1000, source="fixture test prices")
        config = ProviderConfig(model=MODEL, pricing=pricing, max_output_tokens=128, max_cost_usd="0.00129")
        with patch("candidate_io.apply_recipe", side_effect=AssertionError("No remaining cost reservation")):
            _, result, _ = self.execute([response(reading()), response(plan())], config=config)
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["agent"]["stop_reason"], "cost_budget_exceeded")
        self.assertEqual(Decimal(result["agent"]["events"][0]["inference"]["estimated_cost_usd"]), Decimal("0.000025"))

    def test_mock_agent_uses_no_network_even_with_key_and_saves_distinct_runs(self):
        results = []
        with patch("socket.socket", side_effect=AssertionError("Offline mock")):
            for _ in range(2):
                with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                    code = main([str(self.image), "--agent", "--mode", "mock", "--observation-id", "same-observation",
                                 "--output-dir", str(self.root / "runs")])
                self.assertEqual(code, 0)
                results.append(json.loads(Path(stdout.getvalue().strip()).read_text()))
        self.assertNotEqual(results[0]["run_id"], results[1]["run_id"])
        for result in results:
            self.assertEqual(result["api_calls"], 0)
            self.assertEqual(result["agent"]["planner_id"], "mock-fixed-planner-v1")
            self.assertEqual(result["correction_candidates_generated"], 2)
            self.assertEqual(result["status"], "needs_review")
            self.assertIsNone(result["selected_candidate_id"])
            self.assertIsNone(result["reading"]["value"])

    def test_cli_budget_stop_is_review_and_technical_plan_error_exits_nonzero(self):
        for outcomes, config, expected in [([response(reading())], ProviderConfig(model=MODEL, max_observation_api_calls=1), 0),
                                            ([response(reading()), response({})], None, 1)]:
            reader, _, _ = self.clients(outcomes, config)
            with patch("cli.configured_reader", return_value=reader), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                status = main([str(self.image), "--agent", "--mode", "inference", "--model", MODEL,
                               "--output-dir", str(self.root / "runs")])
            self.assertEqual(status, expected)
            self.assertEqual(json.loads(Path(stdout.getvalue().strip()).read_text())["status"], "needs_review")

    def test_invalid_policy_or_cli_combinations_fail_before_calls(self):
        for options in (["--agent", "--recipes", "gamma_mild"], ["--agent", "--auto-roi", "--roi-only", "--mode", "inference"],
                        ["--comparison-policy", "missing.json"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(self.image), *options])
        policy = self.root / "policy.json"
        for document in ({"analog_absolute_tolerance": "-1", "unit": "psi"},
                         {"analog_absolute_tolerance": "NaN", "unit": "psi"}, {"unit": "psi"}):
            policy.write_text(json.dumps(document))
            with redirect_stderr(io.StringIO()), patch("cli.configured_reader", side_effect=AssertionError("No provider")):
                code = main([str(self.image), "--agent", "--comparison-policy", str(policy)])
            self.assertEqual(code, 1)

    def test_cli_comparison_policy_is_recorded_without_enabling_acceptance(self):
        policy = self.root / "policy.json"
        policy.write_text(json.dumps({"analog_absolute_tolerance": "0.02", "unit": "MPa"}))
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            status = main([str(self.image), "--agent", "--comparison-policy", str(policy), "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 0)
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["agent"]["comparison_policy"], {"analog_absolute_tolerance": "0.02", "unit": "MPa"})
        self.assertFalse(result["auto_accept"])

    def test_different_provider_budgets_are_rejected(self):
        reader, planner, comparator = self.clients([])
        other_reader, other_planner, _ = self.clients([])
        with self.assertRaises(ValueError):
            run_observation(ObservationInput(image_path=self.image), reader, self.root / "runs",
                            planner=other_planner, comparator=comparator)
        self.assertFalse((self.root / "runs").exists())

    def test_planner_cannot_modify_original_roi(self):
        class Mutating(MockPlanner):
            def plan(self, request):
                request.image_path.write_bytes(b"tampered")
                return super().plan(request)
        with self.assertRaises(ImageLoadError):
            run_observation(ObservationInput(image_path=self.image), MockReader(), self.root / "runs",
                            planner=Mutating(), comparator=MockComparator())
        self.assertEqual(self.image.read_bytes(), self.original)

    def test_scale_unit_inconsistent_with_read_value_remains_review(self):
        document = reading(20, "psi")
        document["scale"] = ScaleInfo(min_value=0, max_value=60, unit="bar", status="readable", source="image").model_dump()
        _, result, _ = self.execute([response(document), response(plan("stop"))])
        self.assertIn("scale_unit_conflict", result["reason_codes"])
        self.assertIsNone(result["selected_candidate_id"])

    def test_cap_after_apply_decision_records_unexecuted_candidate(self):
        with patch("candidate_io.apply_recipe", side_effect=AssertionError("No remaining API budget")):
            _, result, _ = self.execute([response(reading()), response(plan())],
                                        config=ProviderConfig(model=MODEL, max_observation_api_calls=2))
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["candidates"][1]["api_calls"], 0)
        self.assertIsNone(result["candidates"][1]["image_path"])
        self.assertEqual(result["agent"]["events"][-2]["status"], "skipped")

    def test_refused_or_incomplete_planning_has_no_repair_request(self):
        for changes, code in [({"status": "incomplete"}, "incomplete"),
                               ({"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "test"}]}]}, "refused")]:
            with self.subTest(code=code):
                _, result, _ = self.execute([response(reading()), response(plan(), **changes)])
                self.assertEqual(result["api_calls"], 2)
                self.assertEqual(result["agent"]["stop_reason"], code)

    def test_comparison_api_error_retains_corrected_observation_without_selection(self):
        _, result, _ = self.execute([response(reading()), response(plan()), response(reading(20, "psi")),
                                    TransportFailure("permission_error", status=403)])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"][1]["reading"]["value"], 20)
        self.assertIsNone(result["selected_candidate_id"])
        self.assertEqual(result["agent"]["events"][-1]["error_code"], "permission_error")

    def test_comparator_cannot_modify_saved_corrected_image(self):
        reader, planner, comparator = self.clients([response(reading()), response(plan()), response(reading(20, "psi"))])
        def mutating(request):
            request.corrected_path.write_bytes(b"tampered")
            return comparison()
        with patch.object(comparator, "compare", side_effect=mutating), self.assertRaises(ImageLoadError):
            run_observation(ObservationInput(image_path=self.image), reader, self.root / "runs",
                            planner=planner, comparator=comparator)
        self.assertEqual(self.image.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
