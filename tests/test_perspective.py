"""Known synthetic geometry and simulated HTTP; never a meter accuracy benchmark."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import main
from geometry_locator import VLMGeometryLocator
from perspective import GeometryError, perspective_step
from planner import VLMPlanner, VLMComparator
from provider import HTTPResult, OpenAIProvider, ProviderConfig, TokenPricing, TransportFailure
from reader import MockReader
from recording import run_observation
from replay_corrections import replay
from schemas import GeometryOutput, ObservationInput, PerspectiveSpec, ReaderOutput, RunResult
from tools import apply_recipe, pixel_sha256
from vlm_reader import VLMReader
from vlm_request import strict_schema

MODEL = "perspective-offline-fixture-not-a-real-model"


def rect_spec(points=((12, 10), (112, 18), (104, 86), (20, 80)), ratio=1):
    return PerspectiveSpec(reference="rectangle", source_points=points, aspect_ratio=ratio,
                           evidence="Synthetic known physical square; test only")


def dial_spec():
    return PerspectiveSpec(reference="dial_cardinals", source_points=((63.5, 0), (127, 47.5), (63.5, 95), (0, 47.5)),
                           evidence="Synthetic known circle with physical cardinal markers")


def response(document):
    body = {"status": "completed", "model": MODEL, "usage": {"input_tokens": 11, "output_tokens": 7},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(document)}]}]}
    return HTTPResult(json.dumps(body).encode(), 200, "req_fixture")


def plan(recipe="perspective_rectify"):
    return {"action": "apply", "recipe_id": recipe, "reason": "Synthetic test choice", "reason_codes": ["fixture"]}


def geometry(status="ready", points=None):
    return {"status": status, "points": points or dial_spec().source_points if status == "ready" else None,
            "evidence": "Synthetic physical cardinal markers, not ellipse extrema", "issue_codes": []}


class PerspectiveTests(unittest.TestCase):
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
        self.env = patch.dict(os.environ, {"METER_READER_API_KEY": "fixture-key-not-real"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def clients(self, outputs, config=None, clock=None):
        queue = list(outputs)
        def transport(request, timeout):
            self.sent.append((json.loads(request.data), timeout))
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        kwargs = {"clock": clock} if clock else {}
        provider = OpenAIProvider(config or ProviderConfig(model=MODEL), transport=transport, sleep=lambda _: None, **kwargs)
        return VLMReader(provider), VLMPlanner(provider), VLMComparator(provider), VLMGeometryLocator(provider)

    def execute(self, outputs, *, config=None, manual=None, enabled=True, **fields):
        reader, planner, comparator, locator = self.clients(outputs, config)
        path = run_observation(ObservationInput(image_path=self.image, **fields), reader, self.root / "runs",
                               planner=planner, comparator=comparator, geometry_locator=locator if enabled else None,
                               perspective_config=manual)
        result = json.loads(path.read_text())
        RunResult.model_validate(result)
        return path, result

    def test_known_projective_correspondences_restore_colored_features(self):
        with Image.new("RGB", (128, 128), "white") as front:
            draw = ImageDraw.Draw(front)
            draw.ellipse((20, 20, 108, 108), outline="black", width=2)
            markers = [((35, 35), (255, 0, 0)), ((95, 35), (0, 255, 0)),
                       ((95, 95), (0, 0, 255)), ((35, 95), (0, 0, 0))]
            for (x, y), color in markers:
                draw.rectangle((x - 6, y - 6, x + 6, y + 6), fill=color)
            source = np.float32([[0, 0], [127, 0], [127, 127], [0, 127]])
            spec = rect_spec()
            forward = cv2.getPerspectiveTransform(source, np.float32(spec.source_points))
            oblique = cv2.warpPerspective(np.array(front), forward, (128, 96), borderValue=(255, 255, 255))
        with Image.fromarray(oblique) as image:
            before = pixel_sha256(image)
            corrected = apply_recipe(image, "perspective_rectify", perspective=spec)
            try:
                self.assertEqual(corrected.image.size, (128, 128))
                for point, color in markers:
                    self.assertEqual(corrected.image.getpixel(point), color)
                self.assertEqual(pixel_sha256(image), before)
                matrix = np.asarray(corrected.steps[0].parameters["matrix"])
                mapped = cv2.perspectiveTransform(np.float64(spec.source_points).reshape(1, 4, 2), matrix)[0]
                np.testing.assert_allclose(mapped, source, atol=1e-3)
            finally:
                corrected.image.close()

    def test_rectangle_identity_preserves_pixels_exactly(self):
        spec = rect_spec(((0, 0), (127, 0), (127, 95), (0, 95)), ratio=128 / 96)
        with Image.open(self.image) as original:
            corrected = apply_recipe(original, "perspective_rectify", perspective=spec)
            try:
                self.assertEqual(pixel_sha256(corrected.image), pixel_sha256(original))
            finally:
                corrected.image.close()

    def test_cardinal_reference_uses_diamond_to_circle_mapping(self):
        step = perspective_step(dial_spec(), (128, 96))
        self.assertEqual(step.parameters["output_size"], [128, 128])
        self.assertEqual(step.parameters["destination_points"], [[63.5, 0], [127, 63.5], [63.5, 127], [0, 63.5]])

    def test_invalid_order_duplicate_collinear_small_and_out_of_bounds_rejected(self):
        for points in [((0, 0), (127, 95), (127, 0), (0, 95)), ((0, 0), (0, 95), (127, 95), (127, 0)),
                       ((0, 0), (127, 0), (127, 0), (0, 95)), ((0, 0), (10, 10), (20, 20), (30, 30)),
                       ((0, 0), (3, 0), (3, 3), (0, 3)), ((-1, 0), (127, 0), (127, 95), (0, 95)),
                       ((0, 0), (128, 0), (127, 95), (0, 95))]:
            with self.subTest(points=points), self.assertRaises(GeometryError):
                perspective_step(rect_spec(points), (128, 96))

    def test_bad_types_nonfinite_and_unknown_rectangle_ratio_rejected(self):
        for points in [((True, 0), (127, 0), (127, 95), (0, 95)),
                       (("0", 0), (127, 0), (127, 95), (0, 95)),
                       ((float("nan"), 0), (127, 0), (127, 95), (0, 95))]:
            with self.subTest(points=points), self.assertRaises(ValidationError):
                rect_spec(points)
        for ratio in (None, 0, -1, 5, float("inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValidationError):
                rect_spec(ratio=ratio)

    def test_output_cannot_sample_outside_roi_or_exceed_resource_limit(self):
        spec = PerspectiveSpec(reference="dial_cardinals", source_points=((64, 0), (127, 48), (60, 80), (0, 40)), evidence="test")
        with self.assertRaises(GeometryError):
            perspective_step(spec, (128, 96))
        with patch("perspective.MAX_PIXELS", 100), self.assertRaises(GeometryError):
            perspective_step(rect_spec(), (128, 96))

    def test_rgba_identity_and_alpha_interpolation_are_preserved(self):
        with Image.new("RGBA", (128, 96), (200, 50, 100, 77)) as image:
            spec = rect_spec(((0, 0), (127, 0), (127, 95), (0, 95)), ratio=128 / 96)
            corrected = apply_recipe(image, "perspective_rectify", perspective=spec)
            try:
                self.assertEqual(corrected.image.mode, "RGBA")
                self.assertEqual(corrected.image.getpixel((32, 32)), (200, 50, 100, 77))
            finally:
                corrected.image.close()

    def test_fixed_manual_perspective_and_other_candidate_are_independent_offline(self):
        with patch("socket.socket", side_effect=AssertionError("No network")):
            path = run_observation(ObservationInput(image_path=self.image), MockReader(), self.root / "runs",
                                   recipes=("perspective_rectify", "clahe_mild"), perspective_config=rect_spec())
        result = json.loads(path.read_text())
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["candidates"][1]["processing_version"], "perspective-v1")
        self.assertEqual(result["candidates"][1]["geometry"]["source"], "manual")
        self.assertEqual(result["candidates"][1]["steps"][0]["operation"], "perspective")
        self.assertTrue((path.parent / "candidates/c1_perspective_points.png").exists())
        self.assertTrue(all(c["parent_candidate_id"] == "c0" for c in result["candidates"][1:]))
        self.assertEqual(self.image.read_bytes(), self.original)
        with patch("socket.socket", side_effect=AssertionError("Offline replay")):
            report = json.loads(replay(path, self.root / "replays").read_text())
        self.assertTrue(report["all_pixels_match"])
        self.assertTrue(all(c["file_hash_matches"] for c in report["candidates"]))

    def test_replay_rejects_tampered_matrix_reference_and_roi_binding(self):
        for change in ("matrix", "reference", "hash"):
            path = run_observation(ObservationInput(image_path=self.image), MockReader(), self.root / "runs",
                                   recipes=("perspective_rectify",), perspective_config=rect_spec())
            result = json.loads(path.read_text())
            candidate = result["candidates"][1]
            if change == "matrix":
                candidate["steps"][0]["parameters"]["matrix"][0][0] += .1
            elif change == "reference":
                candidate["geometry"]["spec"]["source_points"][0][0] += 1
            else:
                candidate["geometry"]["input_roi_sha256"] = "0" * 64
            path.write_text(json.dumps(result))
            with self.subTest(change=change), self.assertRaises(ValueError):
                replay(path, self.root / "replays")

    def test_geometry_api_is_separate_from_reader_and_consumes_shared_budget(self):
        outputs = [response(ReaderOutput().model_dump()), response(plan()), response(geometry()),
                   response(ReaderOutput().model_dump()), response(plan("clahe_mild")), response(ReaderOutput().model_dump())]
        _, result = self.execute(outputs, question="Read psi", dataset={"source": {"truth": 987654}},
                                  acceptance_policy={"previous_value": 765432})
        self.assertEqual(result["api_calls"], 6)
        corrected = result["candidates"][1]
        self.assertEqual(corrected["api_calls"], 2)
        self.assertEqual(corrected["geometry"]["api_calls"], 1)
        self.assertEqual(corrected["image_size"], [128, 128])
        self.assertEqual(corrected["geometry"]["status"], "ready")
        geometry_payload = self.sent[2][0]
        self.assertEqual(geometry_payload["text"]["format"]["name"], "meter_geometry_v1")
        target = json.loads(geometry_payload["input"][0]["content"][0]["text"])
        self.assertEqual(set(target), {"target_type", "question", "image_width", "image_height"})
        self.assertNotIn("987654", json.dumps(self.sent))
        self.assertNotIn("765432", json.dumps(self.sent))
        for payload, _ in self.sent:
            if payload["text"]["format"]["name"] == "meter_reading_v02":
                reader_target = json.loads(payload["input"][0]["content"][0]["text"])
                self.assertEqual(set(reader_target), {"target_type", "question"})

    def test_ambiguous_or_occluded_geometry_stops_without_transform_or_reader(self):
        for status in ("ambiguous", "not_found"):
            with self.subTest(status=status), patch("candidate_io.apply_recipe", side_effect=AssertionError("Do not warp")):
                path, result = self.execute([response(ReaderOutput().model_dump()), response(plan()), response(geometry(status))])
            self.assertEqual(result["api_calls"], 3)
            self.assertEqual(result["status"], "needs_review")
            self.assertEqual(result["agent"]["stop_reason"], "geometry_" + status)
            self.assertEqual(result["correction_candidates_generated"], 0)
            self.assertIsNone(result["candidates"][1]["image_path"])
            self.assertEqual(result["candidates"][1]["geometry"]["status"], status)
            self.assertFalse((path.parent / "candidates").exists())

    def test_invalid_geometry_is_preserved_as_error_without_retry(self):
        for document in ({}, {**geometry(), "points": [[0, 0]] * 4},
                         {**geometry(), "issue_codes": ["ellipse_only"]}, {**geometry(), "value": 33}):
            with self.subTest(document=document):
                _, result = self.execute([response(ReaderOutput().model_dump()), response(plan()), response(document)])
                self.assertEqual(result["api_calls"], 3)
                self.assertEqual(result["agent"]["stop_reason"], "invalid_geometry")
                self.assertEqual(result["candidates"][1]["geometry"]["status"], "error")
                self.assertEqual(result["candidates"][1]["geometry"]["inference"]["error_code"], "invalid_geometry")

    def test_disabled_perspective_is_not_offered_or_executed(self):
        _, result = self.execute([response(ReaderOutput().model_dump()), response(plan())], enabled=False)
        self.assertEqual(result["agent"]["stop_reason"], "invalid_planner_decision")
        planner_payload = json.loads(self.sent[1][0]["input"][0]["content"][0]["text"])
        self.assertNotIn("perspective_rectify", planner_payload["recipes"])

    def test_geometry_can_consume_last_call_and_prevent_warp(self):
        with patch("candidate_io.apply_recipe", side_effect=AssertionError("No budget after geometry")):
            _, result = self.execute([response(ReaderOutput().model_dump()), response(plan()), response(geometry())],
                                      config=ProviderConfig(model=MODEL, max_observation_api_calls=3))
        self.assertEqual(result["api_calls"], 3)
        self.assertEqual(result["agent"]["stop_reason"], "observation_api_call_limit")
        self.assertEqual(result["candidates"][1]["geometry"]["status"], "ready")
        self.assertIsNone(result["candidates"][1]["image_path"])

    def test_geometry_network_error_and_retry_share_global_count(self):
        failure = TransportFailure("rate_limit_error", status=429)
        _, result = self.execute([response(ReaderOutput().model_dump()), response(plan()), failure, response(geometry()),
                                  response(ReaderOutput().model_dump())], config=ProviderConfig(model=MODEL, max_observation_api_calls=5))
        self.assertEqual(result["api_calls"], 5)
        self.assertEqual(result["candidates"][1]["geometry"]["api_calls"], 2)
        self.assertEqual(result["candidates"][1]["api_calls"], 3)
        self.assertEqual(result["agent"]["stop_reason"], "observation_api_call_limit")

    def test_manual_config_is_validated_before_reading(self):
        invalid = rect_spec(((0, 0), (128, 0), (127, 95), (0, 95)))
        with self.assertRaises(GeometryError):
            self.execute([], manual=invalid)
        self.assertEqual(self.sent, [])

    def test_cli_manual_fixed_recipe_is_offline_and_auto_requires_agent(self):
        config = self.root / "perspective.json"
        config.write_text(rect_spec().model_dump_json())
        with patch("socket.socket", side_effect=AssertionError("Offline")), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            status = main([str(self.image), "--mode", "mock", "--recipes", "perspective_rectify",
                           "--perspective-config", str(config), "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(Path(stdout.getvalue().strip()).read_text())["candidates"][1]["geometry"]["source"], "manual")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([str(self.image), "--allow-perspective"])
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main([str(self.image), "--recipes", "perspective_rectify"]), 1)

    def test_cli_geometry_ambiguity_is_review_and_invalid_points_are_error(self):
        for document, expected in [(geometry("ambiguous"), 0), ({}, 1)]:
            reader, _, _, _ = self.clients([response(ReaderOutput().model_dump()), response(plan()), response(document)])
            with patch("cli.configured_reader", return_value=reader), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                status = main([str(self.image), "--mode", "inference", "--model", MODEL, "--agent", "--allow-perspective",
                               "--output-dir", str(self.root / "runs")])
            self.assertEqual(status, expected)

    def test_geometry_schema_is_strict_and_has_no_numeric_measurement_field(self):
        schema = strict_schema(GeometryOutput)
        self.assertEqual(set(schema["required"]), {"status", "points", "evidence", "issue_codes"})
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("prefixItems", json.dumps(schema))

    def test_cost_budget_can_block_after_successful_geometry_before_warp(self):
        prices = TokenPricing(model=MODEL, input_usd_per_million="1", output_usd_per_million="2",
                              max_input_tokens_per_call=1000, source="fixture rates")
        config = ProviderConfig(model=MODEL, pricing=prices, max_output_tokens=128, max_cost_usd="0.00132")
        with patch("candidate_io.apply_recipe", side_effect=AssertionError("No cost headroom after geometry")):
            _, result = self.execute([response(ReaderOutput().model_dump()), response(plan()), response(geometry())], config=config)
        self.assertEqual(result["api_calls"], 3)
        self.assertEqual(result["agent"]["stop_reason"], "cost_budget_exceeded")
        self.assertEqual(result["candidates"][1]["geometry"]["status"], "ready")

    def test_deadline_does_not_reset_for_geometry(self):
        class Clock:
            now = 0
            def __call__(self):
                return self.now
        clock = Clock()
        reader, planner, comparator, locator = self.clients([], clock=clock)
        def transport(request, timeout):
            name = json.loads(request.data)["text"]["format"]["name"]
            clock.now += 41
            return response(ReaderOutput().model_dump() if name == "meter_reading_v02" else (
                plan() if name == "meter_planner_v1" else geometry()))
        reader.provider.transport = transport
        with patch("candidate_io.apply_recipe", side_effect=AssertionError("Expired during geometry")):
            path = run_observation(ObservationInput(image_path=self.image), reader, self.root / "runs",
                                   planner=planner, comparator=comparator, geometry_locator=locator)
        result = json.loads(path.read_text())
        self.assertEqual(result["api_calls"], 3)
        self.assertEqual(result["agent"]["stop_reason"], "deadline_exceeded")
        self.assertEqual(result["candidates"][1]["geometry"]["status"], "error")

    def test_manual_points_are_roi_local_after_exif_normalization(self):
        with Image.open(self.image) as image:
            exif = Image.Exif()
            exif[274] = 6
            image.save(self.image, exif=exif)
        spec = rect_spec(((0, 0), (79, 0), (79, 69), (0, 69)), ratio=80 / 70)
        path = run_observation(ObservationInput(image_path=self.image, roi=[10, 20, 90, 90]), MockReader(),
                               self.root / "runs", recipes=("perspective_rectify",), perspective_config=spec)
        result = json.loads(path.read_text())
        self.assertEqual(result["source_image"]["normalized_size"], [96, 128])
        self.assertEqual(result["candidates"][0]["image_size"], [80, 70])
        self.assertEqual(result["candidates"][0]["pixel_sha256"], result["candidates"][1]["pixel_sha256"])

    def test_digital_auto_reference_is_unresolved_without_an_extra_api_call(self):
        _, result = self.execute([response(ReaderOutput().model_dump()), response(plan())], target_type="digital")
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["agent"]["stop_reason"], "geometry_ambiguous")
        self.assertEqual(result["candidates"][1]["geometry"]["api_calls"], 0)

    def test_extreme_warp_is_rejected_even_when_points_are_convex(self):
        with self.assertRaises(GeometryError):
            perspective_step(rect_spec(((0, 0), (999, 0), (999, 8), (0, 999))), (1000, 1000))


if __name__ == "__main__":
    unittest.main()
