"""Fixed recipe correctness and reproducibility, never numeric reading accuracy."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import main
from image_io import ImageLoadError
from quality import measure_quality
from reader import MockReader, ReaderError
from recording import run_observation
from replay_corrections import replay, main as replay_main
from schemas import ObservationInput, ReaderOutput, RunResult
from tools import RECIPES, CorrectionError, apply_recipe, gamma_luminance, pixel_sha256, recipe_steps, validate_recipes


class Stage3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.png"
        y, x = np.indices((96, 128))
        array = np.stack(((x * 2 + y) % 180, (x + y * 2) % 150, (x + y) % 120), axis=-1).astype(np.uint8)
        with Image.fromarray(array) as image:
            image.save(self.input)
        self.original = self.input.read_bytes()

    def run_recipes(self, recipes=("resize_gamma", "clahe_mild"), reader=None, **fields):
        path = run_observation(ObservationInput(image_path=self.input, **fields),
                               reader or MockReader(), self.root / "runs", recipes=recipes)
        result = json.loads(path.read_text())
        RunResult.model_validate(result)
        return path, result

    def test_gamma_uses_exponent_three_quarters_not_reciprocal(self):
        channel = np.array([[0, 64, 128, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(gamma_luminance(channel), [[0, 90, 152, 255]])
        with Image.new("RGB", (8, 8), (64, 64, 64)) as image:
            correction = apply_recipe(image, "gamma_mild")
            try:
                self.assertEqual(correction.image.getpixel((0, 0)), (90, 90, 90))
            finally:
                correction.image.close()

    def test_all_seven_recipes_preserve_input_and_expected_geometry(self):
        self.assertEqual(len(RECIPES), 7)
        with Image.open(self.input) as image:
            before = pixel_sha256(image)
            for recipe in RECIPES:
                with self.subTest(recipe=recipe):
                    correction = apply_recipe(image, recipe)
                    try:
                        doubled = any(step.operation == "resize" for step in correction.steps)
                        self.assertEqual(correction.image.size, (256, 192) if doubled else (128, 96))
                        self.assertEqual(correction.image.mode, "RGB")
                        self.assertNotEqual(pixel_sha256(correction.image), before)
                        self.assertEqual(pixel_sha256(image), before)
                    finally:
                        correction.image.close()
        self.assertEqual(self.input.read_bytes(), self.original)

    def test_combined_recipe_order_and_recorded_parameters(self):
        expected = {
            "resize_2x": ["resize"], "gamma_mild": ["gamma"], "clahe_mild": ["clahe"],
            "gamma_clahe": ["gamma", "clahe"], "resize_gamma": ["gamma", "resize"],
            "resize_clahe": ["clahe", "resize"], "resize_gamma_clahe": ["gamma", "clahe", "resize"],
        }
        for recipe, order in expected.items():
            steps = recipe_steps(recipe, (128, 96))
            self.assertEqual([step.operation for step in steps], order)
            for step in steps:
                if step.operation == "gamma":
                    self.assertEqual(step.parameters["exponent"], 0.75)
                elif step.operation == "clahe":
                    self.assertEqual(step.parameters["clip_limit"], 2.0)
                    self.assertEqual(step.parameters["effective_tile_grid"], [4, 3])
                else:
                    self.assertEqual(step.parameters["output_size"], [256, 192])

    def test_small_roi_clahe_uses_coarser_tiles_and_handles_one_pixel(self):
        for size, grid in [((15, 23), [1, 1]), ((128, 64), [4, 2]), ((500, 500), [8, 8]), ((1, 1), [1, 1])]:
            with self.subTest(size=size), Image.new("RGB", size, (32, 32, 32)) as image:
                correction = apply_recipe(image, "clahe_mild")
                try:
                    self.assertEqual(correction.steps[0].parameters["effective_tile_grid"], grid)
                    self.assertEqual(correction.image.size, size)
                finally:
                    correction.image.close()

    def test_alpha_is_preserved_for_photometric_operations(self):
        with Image.open(self.input) as rgb:
            image = rgb.convert("RGBA")
        self.addCleanup(image.close)
        with Image.new("L", image.size, 77) as alpha:
            image.putalpha(alpha)
        before = pixel_sha256(image)
        for recipe in ("gamma_mild", "clahe_mild", "gamma_clahe"):
            correction = apply_recipe(image, recipe)
            try:
                self.assertEqual(correction.image.mode, "RGBA")
                np.testing.assert_array_equal(np.array(correction.image)[:, :, 3], 77)
                self.assertEqual(pixel_sha256(image), before)
            finally:
                correction.image.close()

    def test_registry_rejects_unknown_duplicate_and_excess_recipes(self):
        for names in (["not_registered"], ["gamma_mild", "gamma_mild"],
                      ["gamma_mild", "clahe_mild", "resize_2x"], "gamma_mild", [1]):
            with self.subTest(names=names), self.assertRaises(CorrectionError):
                validate_recipes(names)
        with Image.open(self.input) as image, self.assertRaises(CorrectionError):
            apply_recipe(image, "__import__('os')")
        with self.assertRaises(TypeError):
            RECIPES["not_registered"] = ()  # Mapping itself is immutable.

    def test_resize_resource_limit_is_checked_before_allocation(self):
        with Image.open(self.input) as image, patch("tools.MAX_OUTPUT_PIXELS", 100):
            with self.assertRaises(CorrectionError):
                apply_recipe(image, "resize_2x")

    def test_quality_metrics_are_descriptive_and_finite_for_tiny_images(self):
        with Image.new("RGB", (1, 1), "black") as black:
            metrics = measure_quality(black)
            self.assertEqual(metrics.luminance_p50, 0)
            self.assertEqual(metrics.dark_fraction, 1)
            self.assertEqual(metrics.saturated_fraction, 0)
            self.assertEqual(metrics.local_mean_spread, 0)
            self.assertEqual(metrics.parameters["effective_grid"], [1, 1])
        with Image.new("RGBA", (2, 2), (0, 0, 0, 0)) as transparent:
            metrics = measure_quality(transparent)
            self.assertEqual(metrics.luminance_p50, 255)
            self.assertEqual(metrics.dark_fraction, 0)
            self.assertEqual(metrics.saturated_fraction, 1)
        with Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (8, 1))).convert("RGB") as gradient:
            metrics = measure_quality(gradient)
            self.assertAlmostEqual(metrics.luminance_p50, 127.5)
            self.assertGreater(metrics.local_mean_spread, 100)

    def test_manifest_contains_original_and_independent_unselected_candidates(self):
        path, result = self.run_recipes(roi=[10, 10, 100, 80], observation_id="same-observation")
        self.assertEqual(result["stage"], 3)
        self.assertEqual(result["baseline_candidate_id"], "c0")
        self.assertIsNone(result["selected_candidate_id"])
        self.assertEqual(result["correction_candidates_generated"], 2)
        self.assertEqual(result["api_calls"], 0)
        self.assertFalse(result["auto_accept"])
        self.assertEqual(result["reading"], result["candidates"][0]["reading"])
        self.assertEqual(len(result["candidates"]), 3)
        original, *candidates = result["candidates"]
        self.assertEqual(original["image_path"], "roi.png")
        self.assertEqual(original["image_size"], [90, 70])
        for candidate in candidates:
            self.assertEqual(candidate["parent_candidate_id"], "c0")
            self.assertEqual(candidate["input_roi_sha256"], original["image_sha256"])
            self.assertEqual(candidate["processing_version"], "fixed-recipes-v1")
            self.assertTrue(candidate["runtime"]["opencv"])
            output = path.parent / candidate["image_path"]
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), candidate["image_sha256"])
            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(pixel_sha256(image), candidate["pixel_sha256"])
        self.assertEqual(self.input.read_bytes(), self.original)

    def test_recipe_order_does_not_chain_candidate_outputs(self):
        _, together = self.run_recipes(("gamma_mild", "gamma_clahe"))
        _, alone = self.run_recipes(("gamma_clahe",))
        self.assertEqual(together["candidates"][2]["pixel_sha256"], alone["candidates"][1]["pixel_sha256"])
        self.assertEqual(together["candidates"][2]["image_sha256"], alone["candidates"][1]["image_sha256"])

    def test_every_reader_request_has_only_independent_image_and_target(self):
        calls = []
        class Spy:
            mode = "inference"
            reader_id = "offline-test-reader"
            def read(self, request):
                calls.append(request)
                return ReaderOutput(value=12.3, value_status="uncertain", issue_codes=["fabricated_test_only"])
        _, result = self.run_recipes(reader=Spy(), question="Read the upper display",
                                     dataset={"source": {"truth_value": 98765}})
        self.assertEqual(len(calls), 3)
        self.assertEqual({call.image_path.name for call in calls}, {"roi.png", "c1_resize_gamma.png", "c2_clahe_mild.png"})
        for call in calls:
            self.assertEqual(set(vars(call)), {"image_path", "target_type", "question"})
            self.assertEqual(call.question, "Read the upper display")
        self.assertIsNone(result["api_calls"])  # Uninstrumented custom Reader: do not invent usage.
        self.assertEqual(result["status"], "needs_review")
        self.assertIsNone(result["effective_value"])

    def test_candidate_reader_error_preserves_baseline_and_returns_nonzero(self):
        class Failing:
            mode = "inference"
            reader_id = "offline-test-reader"
            def read(self, request):
                if request.image_path.name != "roi.png":
                    raise ReaderError("Synthetic candidate failure", code="test_failure")
                return ReaderOutput(value=0, value_status="readable")
        with (
            patch("cli.configured_reader", return_value=Failing()),
            redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr,
        ):
            status = main([str(self.input), "--recipes", "gamma_mild", "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 1)
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["reading"]["value"], 0)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["candidates"][1]["status"], "error")
        self.assertIn("saved: 1, errors: 1", stderr.getvalue())

    def test_processing_error_is_saved_without_reading_failed_image(self):
        with patch("tools.MAX_OUTPUT_PIXELS", 100):
            _, result = self.run_recipes(("resize_2x",))
        self.assertEqual(result["correction_candidates_generated"], 0)
        self.assertEqual(result["candidates"][1]["error_code"], "correction_error")
        self.assertIsNone(result["candidates"][1]["image_path"])
        self.assertEqual(result["candidates"][1]["api_calls"], 0)
        self.assertIsNotNone(result["candidates"][0]["reading"])

    def test_baseline_reader_cannot_change_original_roi(self):
        class Mutating:
            mode = "mock"
            reader_id = "invalid-reader"
            def read(self, request):
                request.image_path.write_bytes(b"changed")
                return ReaderOutput()
        with self.assertRaises(ImageLoadError):
            self.run_recipes(reader=Mutating())
        self.assertEqual(self.input.read_bytes(), self.original)

    def test_exif_roi_basis_is_used_for_all_corrections(self):
        with Image.open(self.input) as image:
            exif = Image.Exif()
            exif[274] = 6
            image.save(self.input, exif=exif)
        _, result = self.run_recipes(roi=[0, 0, 20, 30])
        self.assertEqual(result["source_image"]["normalized_size"], [96, 128])
        self.assertEqual(result["candidates"][0]["image_size"], [20, 30])
        self.assertEqual(result["candidates"][1]["image_size"], [40, 60])

    def test_replay_is_identical_offline_and_leaves_source_run_unchanged(self):
        path, result = self.run_recipes()
        source_files = {file: file.read_bytes() for file in path.parent.rglob("*") if file.is_file()}
        with patch("socket.socket", side_effect=AssertionError("No networking")), patch.dict("os.environ", {}, clear=True):
            output = replay(path, self.root / "replays")
        report = json.loads(output.read_text())
        self.assertTrue(report["all_pixels_match"])
        self.assertEqual(report["api_calls"], 0)
        self.assertFalse(report["readings_recomputed"])
        self.assertTrue(all(row["file_hash_matches"] and row["runtime_matches"] for row in report["candidates"]))
        self.assertTrue(all(file.read_bytes() == content for file, content in source_files.items()))
        self.assertEqual(self.input.read_bytes(), self.original)

    def test_replay_can_regenerate_when_source_photo_and_candidates_are_unavailable(self):
        path, result = self.run_recipes()
        self.input.unlink()
        for candidate in result["candidates"][1:]:
            (path.parent / candidate["image_path"]).unlink()
        report = json.loads(replay(path, self.root / "replays").read_text())
        self.assertTrue(report["all_pixels_match"])

    def test_replay_rejects_tampered_roi_parameters_and_paths(self):
        for kind in ("roi", "parameters", "path", "recipe"):
            with self.subTest(kind=kind):
                path, result = self.run_recipes()
                if kind == "roi":
                    (path.parent / "roi.png").write_bytes(b"tampered")
                elif kind == "parameters":
                    result["candidates"][1]["steps"][0]["parameters"]["exponent"] = 1.5
                elif kind == "path":
                    result["candidates"][0]["image_path"] = "../../outside.png"
                else:
                    result["candidates"][1]["recipe_id"] = "run_arbitrary_code"
                path.write_text(json.dumps(result))
                with self.assertRaises(ValueError):
                    replay(path, self.root / "replays")

    def test_replay_reports_pixel_mismatch_and_runtime_differences(self):
        path, result = self.run_recipes()
        result["candidates"][1]["runtime"]["opencv"] = "different"
        path.write_text(json.dumps(result))
        report = json.loads(replay(path, self.root / "replays").read_text())
        self.assertTrue(report["all_pixels_match"])
        self.assertFalse(report["candidates"][0]["runtime_matches"])
        result["candidates"][1]["pixel_sha256"] = "different"
        path.write_text(json.dumps(result))
        with redirect_stdout(io.StringIO()) as stdout:
            status = replay_main([str(path), "--output-dir", str(self.root / "replays")])
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(Path(stdout.getvalue().strip()).read_text())["all_pixels_match"])

    def test_cli_validates_recipes_before_creating_outputs(self):
        for recipes in (["not_registered"], ["gamma_mild", "gamma_mild"],
                        ["resize_2x", "gamma_mild", "clahe_mild"]):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                status = main([str(self.input), "--recipes", *recipes, "--output-dir", str(self.root / "runs")])
            self.assertEqual(status, 1)
            self.assertFalse((self.root / "runs").exists())

    def test_cli_and_replay_commands_in_subprocesses(self):
        root = Path(__file__).resolve().parents[1]
        run = subprocess.run([sys.executable, "src/cli.py", str(self.input), "--recipes", "resize_gamma",
                              "--output-dir", str(self.root / "runs")], cwd=root, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        again = subprocess.run([sys.executable, "src/replay_corrections.py", run.stdout.strip(),
                                "--output-dir", str(self.root / "replays")], cwd=root, text=True, capture_output=True)
        self.assertEqual(again.returncode, 0, again.stderr)

    def test_inference_counts_at_most_six_attempts_for_three_independent_reads(self):
        from provider import HTTPResult, OpenAIProvider, ProviderConfig, TransportFailure
        from vlm_reader import VLMReader
        sent = []
        def transport(request, timeout):
            sent.append(json.loads(request.data))
            if len(sent) % 2 == 1:
                raise TransportFailure("rate_limit_error", status=429)
            body = {
                "status": "completed", "model": "test-model",
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": ReaderOutput().model_dump_json()},
                ]}],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            }
            return HTTPResult(json.dumps(body).encode(), 200)
        provider = OpenAIProvider(ProviderConfig(model="test-model"), transport=transport, sleep=lambda _: None)
        with patch.dict("os.environ", {"METER_READER_API_KEY": "fixture-not-real"}):
            _, result = self.run_recipes(reader=VLMReader(provider), question="Read the upper display")
        self.assertEqual(len(sent), 6)
        self.assertEqual(result["api_calls"], 6)
        self.assertEqual([c["api_calls"] for c in result["candidates"]], [2, 2, 2])
        self.assertEqual([len(c["inference"]["attempts"]) for c in result["candidates"]], [2, 2, 2])
        self.assertEqual(result["inference"], result["candidates"][0]["inference"])
        self.assertEqual(len({payload["input"][0]["content"][0]["text"] for payload in sent}), 1)
        self.assertIsNone(result["selected_candidate_id"])

    def test_observation_deadline_is_shared_by_original_and_corrected_reads(self):
        from provider import HTTPResult, OpenAIProvider, ProviderConfig
        from vlm_reader import VLMReader
        class Clock:
            now = 0
            def __call__(self):
                return self.now
        clock = Clock()
        timeouts = []
        def transport(request, timeout):
            timeouts.append(timeout)
            clock.now += 100
            body = {
                "status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": ReaderOutput().model_dump_json()},
                ]}],
            }
            return HTTPResult(json.dumps(body).encode(), 200)
        provider = OpenAIProvider(ProviderConfig(model="test-model"), transport=transport, clock=clock)
        with patch.dict("os.environ", {"METER_READER_API_KEY": "fixture-not-real"}):
            _, result = self.run_recipes(reader=VLMReader(provider))
        self.assertEqual(timeouts, [30, 20])
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["candidates"][1]["error_code"], "deadline_exceeded")
        self.assertEqual(result["candidates"][2]["error_code"], "deadline_exceeded")
        self.assertIsNone(result["candidates"][2]["image_path"])


if __name__ == "__main__":
    unittest.main()
