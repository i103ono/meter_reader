"""ROI plumbing with fabricated boxes/HTTP only, never a localization benchmark."""
from __future__ import annotations

import base64
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
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import main
from provider import HTTPResult, OpenAIProvider, ProviderConfig, TokenPricing, TransportFailure
from reader import MockReader, ReaderError
from recording import run_observation
from replay_corrections import replay
from roi_locator import ROILocator, load_proposal, map_roi, model_image_size, parse_proposal
from schemas import ObservationInput, ReaderOutput, ROIProposalOutput, RunResult
from tools import pixel_sha256
from vlm_reader import VLMReader
from vlm_request import strict_schema

MODEL = "offline-roi-test-not-a-real-model"
SECRET = "fixture-key-never-real"


def proposal(status="located", roi=(20, 10, 110, 80)):
    return {
        "status": status, "roi": list(roi) if status == "located" else None,
        "target_description": "Fabricated test dial" if status == "located" else None,
        "reason": None if status == "located" else "Fabricated unresolved target",
        "issue_codes": [] if status == "located" else ["ambiguous_target" if status == "ambiguous" else "no_meter"],
    }


def response(document, **changes):
    body = {
        "id": "resp_roi_fixture", "model": MODEL, "status": "completed",
        "usage": {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(document)},
        ]}],
    }
    body.update(changes)
    return HTTPResult(json.dumps(body).encode(), 200, "req_roi_fixture")


class Clock:
    now = 0.0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


class ROILocatorTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "input.png"
        y, x = np.indices((96, 128))
        with Image.fromarray(np.stack((x, y, x + y), axis=-1).astype(np.uint8)) as image:
            image.save(self.image)
        self.original = self.image.read_bytes()
        self.sent = []
        self.clock = Clock()
        env = patch.dict(os.environ, {"METER_READER_API_KEY": SECRET})
        env.start()
        self.addCleanup(env.stop)

    def clients(self, outcomes=None, config=None, transport=None):
        outcomes = list(outcomes or [response(proposal()), response(ReaderOutput().model_dump())])
        def queued(request, timeout):
            self.sent.append((json.loads(request.data), timeout))
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        provider = OpenAIProvider(config or ProviderConfig(model=MODEL), transport=transport or queued,
                                  clock=self.clock, sleep=self.clock.sleep)
        return VLMReader(provider), ROILocator(provider)

    def execute(self, outcomes=None, config=None, *, recipes=(), roi_only=False, **fields):
        reader, locator = self.clients(outcomes, config)
        path = run_observation(ObservationInput(image_path=self.image, **fields), reader, self.root / "runs",
                               recipes=recipes, roi_locator=locator, roi_only=roi_only)
        result = json.loads(path.read_text())
        RunResult.model_validate(result)
        return path, result, reader

    def test_proposal_and_independent_read_have_distinct_inputs_and_no_label_leakage(self):
        path, result, _ = self.execute(question="Read the upper display", target_type="digital",
                                      dataset={"source": {"secret_truth": 987654}},
                                      acceptance_policy={"previous_value": 765432},
                                      meter_id="not-forwarded")
        locator, reading = [entry[0] for entry in self.sent]
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(locator["text"]["format"]["name"], "meter_roi_v1")
        self.assertEqual(reading["text"]["format"]["name"], "meter_reading_v03")
        locator_target = json.loads(locator["input"][0]["content"][0]["text"])
        reader_target = json.loads(reading["input"][0]["content"][0]["text"])
        self.assertEqual(locator_target, {"question": "Read the upper display", "target_type": "digital",
                                          "image_width": 128, "image_height": 96})
        self.assertEqual(reader_target, {"question": "Read the upper display", "target_type": "digital"})
        for payload in (locator, reading):
            encoded = json.dumps(payload)
            for hidden in ("987654", "765432", "secret_truth", "previous_value", "not-forwarded"):
                self.assertNotIn(hidden, encoded)
            self.assertFalse(payload["store"])
        self.assertNotIn("Fabricated test dial", json.dumps(reading))
        for payload, size in [(locator, (128, 96)), (reading, (90, 70))]:
            data_url = payload["input"][0]["content"][1]["image_url"]
            with Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))) as image:
                self.assertEqual(image.size, size)
        self.assertEqual(result["roi"], [20, 10, 110, 80])
        self.assertEqual(result["roi_source"], "vlm")
        self.assertEqual(result["roi_proposal"]["response"]["roi"], [20, 10, 110, 80])
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["roi_proposal"]["inference"]["attempts"][0]["request_id"], "req_roi_fixture")
        self.assertFalse(result["auto_accept"])
        self.assertIsNone(result["effective_value"])
        self.assertIn("roi_proposal_unverified", result["reason_codes"])
        self.assertNotIn(SECRET, path.read_text())
        self.assertEqual(self.image.read_bytes(), self.original)
        self.assertEqual((path.parent / "source.png").read_bytes(), self.original)
        with Image.open(self.image) as image, Image.open(path.parent / "roi.png") as crop:
            with image.crop((20, 10, 110, 80)) as expected:
                self.assertEqual(pixel_sha256(crop), pixel_sha256(expected))
        with Image.open(path.parent / "roi_preview.png") as preview, Image.open(path.parent / "normalized.png") as normalized:
            self.assertEqual(preview.size, normalized.size)
            self.assertNotEqual(pixel_sha256(preview), pixel_sha256(normalized))

    def test_roi_only_does_not_call_reader_and_saves_review_assets(self):
        with patch.object(VLMReader, "read", side_effect=AssertionError("Reader must not run")):
            path, result, _ = self.execute([response(proposal())], roi_only=True, observation_id="same-observation")
        self.assertEqual(result["api_calls"], 1)
        self.assertFalse(result["read_requested"])
        self.assertIsNone(result["reading"])
        self.assertIsNone(result["inference"])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["roi_proposal"]["observation_id"], "same-observation")
        for name in ("roi_proposal.json", "roi_preview.png", "roi_locator_input.png", "roi.png"):
            self.assertTrue((path.parent / name).is_file())
        self.assertEqual(load_proposal(path.parent / "roi_proposal.json"), RunResult.model_validate(result).roi_proposal)

    def test_ambiguous_or_absent_target_stops_before_cropping_or_reading(self):
        for status in ("ambiguous", "not_found"):
            with self.subTest(status=status), patch.object(VLMReader, "read", side_effect=AssertionError("No read")):
                path, result, _ = self.execute([response(proposal(status))], recipes=("resize_gamma",))
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["api_calls"], 1)
            self.assertIsNone(result["roi"])
            self.assertIsNone(result["reading"])
            self.assertIsNone(result["source_image"]["roi_path"])
            self.assertFalse((path.parent / "roi.png").exists())
            self.assertEqual(result["candidates"], [])
            self.assertIn("roi_" + status, result["reason_codes"])
            self.assertTrue((path.parent / "roi_preview.png").exists())

    def test_invalid_coordinate_types_bounds_and_empty_regions_are_not_repaired(self):
        for roi in ([True, 0, 10, 10], [0.0, 0, 10, 10], ["0", 0, 10, 10],
                    [-1, 0, 10, 10], [0, 0, 129, 10], [0, 0, 10, 97],
                    [5, 0, 5, 10], [10, 0, 5, 10], [0, 0, 10], [0, 0, 10, 10, 20]):
            with self.subTest(roi=roi):
                path, result, _ = self.execute([response(proposal(roi=roi))])
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["roi_proposal"]["error_code"], "invalid_roi_proposal")
                self.assertIsNone(result["roi_proposal"]["proposed_roi"])
                self.assertEqual(result["api_calls"], 1)
                self.assertFalse((path.parent / "roi.png").exists())

    def test_malformed_json_missing_fields_and_ambiguous_coordinates_are_rejected(self):
        invalid = ["not JSON", "{}", json.dumps({**proposal(), "value": 10}),
                   json.dumps({**proposal("ambiguous"), "roi": [0, 0, 10, 10]}),
                   json.dumps({**proposal(), "roi": None}),
                   json.dumps({**proposal(), "issue_codes": ["ambiguous_scale"]}),
                   json.dumps({**proposal(), "target_description": " "})]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ReaderError):
                parse_proposal(raw, (128, 96))

    def test_json_schema_has_strict_array_coordinates_without_prefix_items(self):
        schema = strict_schema(ROIProposalOutput)
        self.assertEqual(set(schema["required"]), set(ROIProposalOutput.model_fields))
        self.assertFalse(schema["additionalProperties"])
        coordinates = next(item for item in schema["properties"]["roi"]["anyOf"] if item["type"] == "array")
        self.assertEqual(coordinates["items"], {"type": "integer"})
        self.assertEqual(coordinates["minItems"], 4)
        self.assertEqual(coordinates["maxItems"], 4)
        self.assertNotIn("prefixItems", json.dumps(schema))

    def test_downsample_maps_each_axis_outward_and_uses_original_pixels(self):
        with Image.new("RGB", (3333, 2001), (75, 85, 95)) as large:
            large.save(self.image)
        original = self.image.read_bytes()
        self.assertEqual(model_image_size((3333, 2001)), (1600, 960))
        roi = (101, 51, 1201, 801)
        _, result, _ = self.execute([response(proposal(roi=roi))], roi_only=True)
        expected = [210, 106, 2502, 1670]
        self.assertEqual(result["roi"], expected)
        self.assertEqual(result["roi_proposal"]["model_image_size"], [1600, 960])
        self.assertEqual(result["source_image"]["normalized_size"], [3333, 2001])
        self.assertEqual(self.image.read_bytes(), original)
        self.assertEqual(map_roi((0, 0, 1600, 960), (1600, 960), (3333, 2001)), (0, 0, 3333, 2001))
        with self.assertRaises(ValueError):
            map_roi((-1, 0, 1600, 960), (1600, 960), (3333, 2001))

    def test_exif_rotation_and_locator_coordinates_share_one_basis(self):
        with Image.open(self.image) as image:
            exif = Image.Exif()
            exif[274] = 6
            image.save(self.image, exif=exif)
        original = self.image.read_bytes()
        path, result, _ = self.execute([response(proposal(roi=(5, 10, 70, 100)))], roi_only=True)
        self.assertEqual(result["source_image"]["normalized_size"], [96, 128])
        payload = self.sent[0][0]
        target = json.loads(payload["input"][0]["content"][0]["text"])
        self.assertEqual((target["image_width"], target["image_height"]), (96, 128))
        with Image.open(path.parent / "normalized.png") as image, Image.open(path.parent / "roi.png") as crop:
            with image.crop((5, 10, 70, 100)) as expected:
                self.assertEqual(pixel_sha256(crop), pixel_sha256(expected))
        self.assertEqual(self.image.read_bytes(), original)

    def test_saved_proposal_can_be_reused_without_network_or_credentials(self):
        path, original, _ = self.execute([response(proposal())], roi_only=True, question="Read the top display")
        saved = path.parent / "roi_proposal.json"
        files = {p: p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
        with patch("socket.socket", side_effect=AssertionError("Offline reuse")), patch.dict(os.environ, {}, clear=True):
            record = load_proposal(saved)
            again = run_observation(ObservationInput(image_path=self.image, question=record.question), MockReader(),
                                    self.root / "runs", roi_proposal=record, roi_proposal_reference=str(saved))
        result = json.loads(again.read_text())
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["roi_source"], "saved_vlm")
        self.assertEqual(result["roi_proposal"], original["roi_proposal"])
        self.assertEqual(result["observation_id"], original["observation_id"])
        self.assertNotEqual(result["run_id"], original["run_id"])
        self.assertEqual((again.parent / "roi.png").read_bytes(), (path.parent / "roi.png").read_bytes())
        self.assertTrue(all(p.read_bytes() == content for p, content in files.items()))

    def test_manual_override_preserves_original_proposal_and_previous_run(self):
        path, original, _ = self.execute([response(proposal())], roi_only=True)
        record = load_proposal(path.parent / "roi_proposal.json")
        override = [10, 5, 120, 90]
        again = run_observation(ObservationInput(image_path=self.image, roi=override), MockReader(),
                                self.root / "runs", roi_proposal=record)
        result = json.loads(again.read_text())
        self.assertEqual(result["roi_source"], "manual_override")
        self.assertEqual(result["roi"], override)
        self.assertEqual(result["roi_proposal"]["proposed_roi"], original["roi"])
        self.assertEqual(json.loads(path.read_text()), original)
        self.assertEqual(result["api_calls"], 0)
        self.assertFalse(result["auto_accept"])

    def test_unresolved_proposal_requires_manual_box_to_continue(self):
        path, _, _ = self.execute([response(proposal("ambiguous"))], roi_only=True)
        record = load_proposal(path.parent / "roi_proposal.json")
        again = run_observation(ObservationInput(image_path=self.image, roi=[0, 0, 128, 96]), MockReader(),
                                self.root / "runs", roi_proposal=record)
        result = json.loads(again.read_text())
        self.assertEqual(result["roi_source"], "manual_override")
        self.assertIsNotNone(result["reading"])
        self.assertEqual(result["roi_proposal"]["status"], "ambiguous")

    def test_saved_proposal_rejects_different_image_target_and_tampered_mapping(self):
        path, original, _ = self.execute([response(proposal())], roi_only=True, question="upper display")
        record = load_proposal(path.parent / "roi_proposal.json")
        for changes in ({"question": "lower display"}, {"target_type": "digital"}):
            fields = {"question": "upper display", **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_observation(ObservationInput(image_path=self.image, **fields), MockReader(),
                                self.root / "runs", roi_proposal=record)
        for changes in ({"proposed_roi": (0, 0, 1, 1)}, {"normalized_pixel_sha256": "0" * 64},
                        {"model_image_size": (64, 48)}, {"input_sha256": "0" * 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                run_observation(ObservationInput(image_path=self.image, question="upper display"), MockReader(),
                                self.root / "runs", roi_proposal=record.model_copy(update=changes))
        with Image.new("RGB", (128, 96), "red") as image:
            image.save(self.image)
        with self.assertRaises(ValueError):
            run_observation(ObservationInput(image_path=self.image, question="upper display"), MockReader(),
                            self.root / "runs", roi_proposal=record)

    def test_result_or_labels_cannot_be_loaded_as_proposals(self):
        path, _, _ = self.execute([response(proposal())], roi_only=True)
        with self.assertRaises(ValueError):
            load_proposal(path)
        path.write_text(json.dumps({"question_id": "test", "value": 1234}))
        with self.assertRaises(ValueError):
            load_proposal(path)

    def test_rgba_crop_and_locator_input_preserve_alpha(self):
        with Image.open(self.image) as image:
            rgba = image.convert("RGBA")
        try:
            rgba.putalpha(77)
            rgba.save(self.image)
        finally:
            rgba.close()
        path, result, _ = self.execute([response(proposal())], roi_only=True)
        with Image.open(path.parent / "roi.png") as crop, Image.open(path.parent / "roi_locator_input.png") as model:
            self.assertEqual(crop.mode, "RGBA")
            self.assertEqual(crop.getpixel((0, 0))[3], 77)
            self.assertEqual(model.mode, "RGBA")
            self.assertEqual(model.getpixel((0, 0))[3], 77)

    def test_invalid_json_records_raw_output_without_repair_requests(self):
        bad = response({})
        path, result, _ = self.execute([bad])
        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(result["roi_proposal"]["error_code"], "invalid_roi_proposal")
        self.assertEqual(result["roi_proposal"]["inference"]["attempts"][0]["raw_output"], "{}")
        self.assertFalse((path.parent / "roi.png").exists())

    def test_technical_errors_are_recorded_without_reader_or_fallback_crop(self):
        outcomes = [TransportFailure("authentication_error", status=401),
                    response(proposal(), status="incomplete"),
                    response(proposal(), output=[{"type": "message", "content": [{"type": "refusal", "refusal": "test"}]}])]
        for outcome, code in zip(outcomes, ("authentication_error", "incomplete", "refused")):
            with self.subTest(code=code):
                path, result, _ = self.execute([outcome])
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["roi_proposal"]["error_code"], code)
                self.assertIsNone(result["reading"])
                self.assertEqual(result["api_calls"], 1)
                self.assertFalse((path.parent / "roi.png").exists())
                self.assertNotIn(SECRET, path.read_text())

    def test_missing_credential_records_zero_calls_and_a_failure_trace(self):
        with patch.dict(os.environ, {}, clear=True):
            _, result, _ = self.execute()
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["roi_proposal"]["error_code"], "credential_missing")
        self.assertEqual(len(self.sent), 0)

    def test_candidates_use_the_same_proposed_original_roi_and_replay_offline(self):
        outputs = [response(proposal())] + [response(ReaderOutput().model_dump())] * 3
        path, result, _ = self.execute(outputs, recipes=("resize_gamma", "clahe_mild"))
        self.assertEqual(result["api_calls"], 4)
        self.assertEqual([c["image_size"] for c in result["candidates"]], [[90, 70], [180, 140], [90, 70]])
        self.assertTrue(all(c["input_roi_sha256"] == result["candidates"][0]["image_sha256"] for c in result["candidates"]))
        self.assertIsNone(result["selected_candidate_id"])
        with patch("socket.socket", side_effect=AssertionError("No live API")):
            report = json.loads(replay(path, self.root / "replays").read_text())
        self.assertTrue(report["all_pixels_match"])
        self.assertEqual(report["api_calls"], 0)

    def test_six_attempt_cap_includes_roi_and_all_reader_retries(self):
        outputs = []
        for document in [proposal()] + [ReaderOutput().model_dump()] * 3:
            outputs += [TransportFailure("rate_limit_error", status=429), response(document)]
        _, result, reader = self.execute(outputs, recipes=("resize_gamma", "clahe_mild"))
        self.assertEqual(len(self.sent), 6)
        self.assertEqual(reader.provider.observation_api_calls, 6)
        self.assertEqual(result["api_calls"], 6)
        self.assertEqual(len(result["roi_proposal"]["inference"]["attempts"]), 2)
        self.assertEqual([c["api_calls"] for c in result["candidates"]], [2, 2, 0])
        self.assertEqual(result["candidates"][2]["error_code"], "observation_api_call_limit")
        self.assertIsNone(result["candidates"][2]["image_path"])

    def test_configured_attempt_cap_blocks_after_roi_before_reader(self):
        _, result, _ = self.execute([response(proposal())], config=ProviderConfig(model=MODEL, max_observation_api_calls=1))
        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(result["inference"]["attempts"], [])
        self.assertIn("observation_api_call_limit", result["reason_codes"])
        for value in (0, 7, True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ProviderConfig(model=MODEL, max_observation_api_calls=value)

    def test_shared_deadline_does_not_reset_after_locator(self):
        timeouts = []
        def transport(request, timeout):
            timeouts.append(timeout)
            self.clock.now += 100
            return response(proposal() if len(timeouts) == 1 else ReaderOutput().model_dump())
        reader, locator = self.clients(transport=transport)
        path = run_observation(ObservationInput(image_path=self.image), reader, self.root / "runs",
                               roi_locator=locator, recipes=("gamma_mild",))
        result = json.loads(path.read_text())
        self.assertEqual(timeouts, [30, 20])
        self.assertEqual(result["api_calls"], 2)
        self.assertIn("deadline_exceeded", result["reason_codes"])
        self.assertEqual(result["candidates"][1]["error_code"], "deadline_exceeded")

    def test_shared_cost_budget_reserves_roi_and_blocks_read_before_network(self):
        pricing = TokenPricing(model=MODEL, input_usd_per_million="1", output_usd_per_million="2",
                               max_input_tokens_per_call=1000, source="fabricated test rates")
        config = ProviderConfig(model=MODEL, max_output_tokens=128, pricing=pricing, max_cost_usd="0.00127")
        _, result, _ = self.execute([response(proposal())], config=config)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(result["api_calls"], 1)
        self.assertIn("cost_budget_exceeded", result["reason_codes"])
        self.assertEqual(Decimal(result["roi_proposal"]["inference"]["estimated_cost_usd"]), Decimal("0.000025"))

    def test_observation_counter_resets_for_new_run_but_cost_budget_is_shared(self):
        reader, locator = self.clients([response(proposal())] * 2,
                                        ProviderConfig(model=MODEL, max_observation_api_calls=1))
        for _ in range(2):
            path = run_observation(ObservationInput(image_path=self.image), reader, self.root / "runs",
                                   roi_locator=locator, roi_only=True)
            self.assertEqual(json.loads(path.read_text())["api_calls"], 1)
        self.assertEqual(len(self.sent), 2)

    def test_separate_provider_budgets_and_manual_auto_conflicts_are_rejected(self):
        reader, locator = self.clients()
        another_reader, _ = self.clients()
        for request, actual_reader in [(ObservationInput(image_path=self.image), another_reader),
                                        (ObservationInput(image_path=self.image, roi=[0, 0, 10, 10]), reader)]:
            with self.assertRaises(ValueError):
                run_observation(request, actual_reader, self.root / "runs", roi_locator=locator)
        self.assertFalse((self.root / "runs").exists())

    def test_cli_saved_proposal_reuses_target_and_can_override_roi(self):
        path, original, _ = self.execute([response(proposal())], roi_only=True, question="top display", target_type="digital")
        with patch("socket.socket", side_effect=AssertionError("Offline CLI")), redirect_stdout(io.StringIO()) as stdout:
            status = main([str(self.image), "--roi-proposal", str(path.parent / "roi_proposal.json"),
                           "--legacy-roi", "1", "2", "30", "40", "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 0)
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["question"], "top display")
        self.assertEqual(result["target_type"], "digital")
        self.assertEqual(result["roi"], [1, 2, 31, 42])
        self.assertEqual(result["roi_source"], "manual_override")
        self.assertEqual(result["observation_id"], original["observation_id"])
        self.assertEqual(result["api_calls"], 0)

    def test_cli_auto_roi_and_roi_only_with_simulated_transport(self):
        reader, _ = self.clients([response(proposal())])
        with patch("cli.configured_reader", return_value=reader), redirect_stdout(io.StringIO()) as stdout:
            status = main([str(self.image), "--auto-roi", "--roi-only", "--mode", "inference",
                           "--model", MODEL, "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 0)
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["api_calls"], 1)
        self.assertFalse(result["read_requested"])

    def test_cli_rejects_invalid_combinations_before_network(self):
        invalid = [["--auto-roi"], ["--roi-only"],
                   ["--auto-roi", "--mode", "inference", "--roi", "0", "0", "1", "1"],
                   ["--auto-roi", "--mode", "inference", "--roi-only", "--recipes", "gamma_mild"],
                   ["--auto-roi", "--mode", "inference", "--roi-proposal", "missing.json"]]
        for extra in invalid:
            with self.subTest(extra=extra), patch("socket.socket", side_effect=AssertionError("No network")), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main([str(self.image), *extra])
            self.assertEqual(raised.exception.code, 2)

    def test_cli_api_errors_save_result_and_exit_nonzero(self):
        reader, _ = self.clients([TransportFailure("authentication_error", status=401)])
        with patch("cli.configured_reader", return_value=reader), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            status = main([str(self.image), "--auto-roi", "--mode", "inference", "--model", MODEL,
                           "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(Path(stdout.getvalue().strip()).read_text())["status"], "error")
        self.assertIn("saved: 1, errors: 1", stderr.getvalue())

    def test_cli_manifest_auto_roi_preserves_question_and_rejects_existing_manual_boxes(self):
        manifest = self.root / "inputs.jsonl"
        manifest.write_text(json.dumps({"image_path": self.image.name, "question": "upper display"}) + "\n")
        reader, _ = self.clients([response(proposal())])
        with patch("cli.configured_reader", return_value=reader), redirect_stdout(io.StringIO()) as stdout:
            status = main(["--inputs", str(manifest), "--auto-roi", "--roi-only", "--mode", "inference",
                           "--model", MODEL, "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 0)
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["question"], "upper display")
        manifest.write_text(json.dumps({"image_path": self.image.name, "roi": [0, 0, 10, 10]}) + "\n")
        with patch("cli.configured_reader", side_effect=AssertionError("Reject before provider init")), redirect_stderr(io.StringIO()):
            status = main(["--inputs", str(manifest), "--auto-roi", "--mode", "inference", "--model", MODEL])
        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
