"""Offline stage-1 contract tests; generated images are not an accuracy dataset."""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli import load_inputs, main
from image_io import (
    ImageLoadError, load_image, prepare_image, save_png_exclusive,
    validate_roi_xyxy, xywh_to_xyxy, xyxy_to_xywh,
)
from reader import MockReader, ReaderError
from recording import run_observation
from schemas import MeterObservationInput, ObservationInput, ReaderOutput, RunResult, ScaleInfo


class Stage1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "meter.png"
        self.out = self.root / "runs"
        self.make_image(self.image)

    def make_image(self, path, orientation=None, capture=None, offset=None, ifd=False):
        image = Image.new("RGB", (6, 4))
        for y in range(4):
            for x in range(6):
                image.putpixel((x, y), (x * 30, y * 60, 15))
        exif = Image.Exif()
        if orientation is not None:
            exif[274] = orientation
        values = {}
        if capture is not None:
            values[36867] = capture
        if offset is not None:
            values[36881] = offset
        if ifd:
            exif[34665] = values
        else:
            exif.update(values)
        image.save(path, exif=exif)
        image.close()

    def read(self, **kwargs):
        request = ObservationInput(image_path=self.image, **kwargs)
        path = run_observation(request, MockReader(), self.out)
        data = json.loads(path.read_text())
        RunResult.model_validate(data)
        return path, data

    def test_png_and_jpeg_source_bytes_are_unchanged_and_archived(self):
        for extension in ("png", "jpg", "jpeg"):
            with self.subTest(extension=extension):
                path = self.root / f"source.{extension}"
                self.make_image(path)
                original = path.read_bytes()
                result = run_observation(ObservationInput(image_path=path), MockReader(), self.out)
                data = json.loads(result.read_text())
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(data["input_sha256"], hashlib.sha256(original).hexdigest())
                self.assertEqual((result.parent / data["source_image"]["stored_path"]).read_bytes(), original)

    def test_extension_content_and_corruption_validation(self):
        gif = self.root / "disguised.png"
        with Image.new("RGB", (3, 3)) as image:
            image.save(gif, format="GIF")
        wrong = self.root / "png.jpg"
        wrong.write_bytes(self.image.read_bytes())
        corrupt = self.root / "broken.jpg"
        corrupt.write_bytes(b"not an image")
        text = self.root / "image.txt"
        text.write_bytes(self.image.read_bytes())
        for path in (gif, wrong, corrupt, text, self.root / "absent.png", self.root):
            with self.subTest(path=path), self.assertRaises(ImageLoadError):
                prepare_image(path)
        self.assertFalse(self.out.exists())

    def test_exif_rotation_precedes_roi_and_orientation_is_not_reapplied(self):
        self.make_image(self.image, orientation=6)
        before = self.image.read_bytes()
        path, data = self.read(roi=[0, 0, 2, 3])
        self.assertEqual(data["source_image"]["original_size"], [6, 4])
        self.assertEqual(data["source_image"]["normalized_size"], [4, 6])
        self.assertEqual(data["source_image"]["exif_orientation"], 6)
        with Image.open(path.parent / "roi.png") as crop:
            self.assertEqual(crop.size, (2, 3))
            self.assertEqual(crop.getpixel((0, 0)), (0, 180, 15))
            self.assertEqual(crop.getpixel((1, 2)), (60, 120, 15))
            self.assertIsNone(crop.getexif().get(274))
        self.assertEqual(self.image.read_bytes(), before)
        with self.assertRaises(ValueError):
            self.read(roi=[0, 0, 6, 4])  # Valid before rotation, invalid afterward.

    def test_exif_mirrored_orientation(self):
        self.make_image(self.image, orientation=2)
        path, _ = self.read(roi=[0, 0, 1, 1])
        with Image.open(path.parent / "roi.png") as crop:
            self.assertEqual(crop.getpixel((0, 0)), (150, 0, 15))

    def test_null_roi_means_full_normalized_image(self):
        path, data = self.read()
        self.assertIsNone(data["roi"])
        self.assertEqual(data["roi_coordinate_system"], "exif_normalized_xyxy_exclusive")
        with Image.open(path.parent / "roi.png") as crop:
            self.assertEqual(crop.size, (6, 4))

    def test_png_transparency_is_preserved(self):
        with Image.new("RGBA", (6, 4), (100, 120, 150, 80)) as image:
            image.save(self.image)
        path, _ = self.read(roi=[0, 0, 2, 2])
        with Image.open(path.parent / "roi.png") as crop:
            self.assertEqual(crop.mode, "RGBA")
            self.assertEqual(crop.getpixel((0, 0)), (100, 120, 150, 80))

    def test_roi_adapters_roundtrip(self):
        self.assertEqual(xywh_to_xyxy([1, 1, 3, 2]), (1, 1, 4, 3))
        self.assertEqual(xyxy_to_xywh([1, 1, 4, 3]), (1, 1, 3, 2))
        self.assertEqual(validate_roi_xyxy([0, 0, 6, 4], (6, 4)), (0, 0, 6, 4))

    def test_invalid_roi_is_never_clamped_or_coerced(self):
        bad = (
            [-1, 0, 2, 2], [0, 0, 7, 4], [0, 0, 6, 5], [1, 0, 1, 2],
            [0, 2, 3, 1], [0, 0, 0, 0], [0, 0, 2], [0, 0, 2, 2, 2],
            [False, 0, 2, 2], [0, 0, 2.0, 2], ["0", 0, 2, 2],
            "0,0,2,2", {"x1": 0}, None,
        )
        for roi in bad:
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                validate_roi_xyxy(roi, (6, 4))
        for roi in bad[:-1]:
            with self.subTest(schema_roi=roi), self.assertRaises((ValidationError, ValueError)):
                self.read(roi=roi)
        for legacy in ([0, 0, 0, 1], [0, 0, 1, -1], [True, 0, 1, 1]):
            with self.assertRaises(ValueError):
                xywh_to_xyxy(legacy)

    def test_capture_unknown_is_not_processing_time(self):
        _, result = self.read()
        self.assertIsNone(result["captured_at"])
        self.assertIsNone(result["captured_timezone"])
        self.assertEqual(result["timestamp_source"], "unknown")
        processed = datetime.fromisoformat(result["processed_at"].replace("Z", "+00:00"))
        self.assertEqual(processed.utcoffset().total_seconds(), 0)
        self.assertLess(abs((datetime.now(timezone.utc) - processed).total_seconds()), 10)

    def test_explicit_capture_preserves_timezone_evidence_and_stores_utc(self):
        _, result = self.read(captured_at="2026-10-27T10:00:00+09:00")
        self.assertEqual(result["captured_at"], "2026-10-27T01:00:00Z")
        self.assertEqual(result["captured_timezone"], "+09:00")
        self.assertEqual(result["timestamp_source"], "user")

    def test_capture_rejects_naive_numeric_and_inconsistent_sources(self):
        for value in ("2026-10-27", "2026-10-27T10:00:00", "0", "123456", 123456, True):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ObservationInput(image_path=self.image, captured_at=value)
        for kwargs in (
            {"timestamp_source": "user"},
            {"captured_at": "2026-10-27T10:00:00Z", "timestamp_source": "unknown"},
        ):
            with self.assertRaises(ValidationError):
                ObservationInput(image_path=self.image, **kwargs)

    def test_exif_capture_with_offset(self):
        self.make_image(self.image, capture="2026:10:27 10:00:00", offset="+09:00", ifd=True)
        _, result = self.read()
        self.assertEqual(result["captured_at"], "2026-10-27T01:00:00Z")
        self.assertEqual(result["timestamp_source"], "exif")
        self.assertEqual(result["timestamp_metadata"]["exif_offset_original"], "+09:00")

    def test_naive_exif_requires_explicit_timezone(self):
        self.make_image(self.image, capture="2026:10:27 10:00:00")
        _, unknown = self.read()
        self.assertIsNone(unknown["captured_at"])
        self.assertEqual(unknown["timestamp_metadata"]["notes"], ["exif_timezone_missing"])
        path = run_observation(ObservationInput(image_path=self.image), MockReader(), self.out,
                               exif_timezone="Asia/Tokyo")
        known = json.loads(path.read_text())
        self.assertEqual(known["captured_at"], "2026-10-27T01:00:00Z")
        self.assertEqual(known["captured_timezone"], "Asia/Tokyo")
        self.assertEqual(known["timestamp_metadata"]["assumed_timezone"], "Asia/Tokyo")

    def test_invalid_exif_and_dst_ambiguity_remain_unknown(self):
        for raw, offset, zone in (
            ("invalid", "+09:00", None),
            ("2026:10:27 10:00:00", "+25:00", None),
            ("2026:11:01 01:30:00", None, "America/New_York"),
            ("2026:03:08 02:30:00", None, "America/New_York"),
        ):
            with self.subTest(raw=raw, offset=offset, zone=zone):
                self.make_image(self.image, capture=raw, offset=offset)
                path = run_observation(ObservationInput(image_path=self.image), MockReader(),
                                       self.out, exif_timezone=zone)
                result = json.loads(path.read_text())
                self.assertIsNone(result["captured_at"])
                self.assertEqual(result["timestamp_source"], "unknown")
                self.assertTrue(result["timestamp_metadata"]["notes"])
        with self.assertRaises(ValueError):
            run_observation(ObservationInput(image_path=self.image), MockReader(),
                            self.out, exif_timezone="NoSuch/Timezone")

    def test_user_capture_takes_priority_over_exif(self):
        self.make_image(self.image, capture="2000:01:01 00:00:00", offset="+00:00")
        _, result = self.read(captured_at="2026-10-27T10:00:00+09:00")
        self.assertEqual(result["captured_at"], "2026-10-27T01:00:00Z")
        self.assertEqual(result["timestamp_source"], "user")
        self.assertEqual(result["timestamp_metadata"]["exif_datetime_original"], "2000:01:01 00:00:00")

    def test_mock_nulls_are_not_measurements_or_accepted(self):
        _, result = self.read(acceptance_policy={"auto_accept": True})
        self.assertEqual(result["mode"], "mock")
        self.assertEqual(result["status"], "needs_review")
        self.assertFalse(result["auto_accept"])
        self.assertIsNone(result["effective_value"])
        self.assertEqual(result["api_calls"], 0)
        for key in ("value", "raw_value_text", "unit", "raw_unit_text"):
            self.assertIsNone(result["reading"][key])
        for key in ("min_value", "max_value", "minor_division", "unit"):
            self.assertIsNone(result["reading"]["scale"][key])
        self.assertIn("mock_not_a_measurement", result["reading"]["issue_codes"])

    def test_reading_validation_and_ambiguity(self):
        for kwargs in (
            {"value_status": "readable"}, {"unit_status": "readable"},
            {"value": float("nan")}, {"value": float("inf")}, {"value": True},
            {"unit": ""}, {"unit": "MPa"}, {"value": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                ReaderOutput(**kwargs)
        for source in ("manual", "registry"):
            with self.assertRaises(ValidationError):
                ReaderOutput(scale=ScaleInfo(source=source))
        for kwargs in (
            {"min_value": 1, "max_value": 0, "source": "image", "status": "partial"},
            {"minor_division": 0, "source": "image", "status": "partial"},
            {"min_value": 0}, {"status": "readable", "source": "image"},
            {"unit": " ", "source": "image", "status": "partial"},
        ):
            with self.assertRaises(ValidationError):
                ScaleInfo(**kwargs)
        reading = ReaderOutput(
            meter_type="analog", value=-0.40, raw_value_text="-0.40", value_status="uncertain",
            issue_codes=["ambiguous_scale"], scale=ScaleInfo(status="partial", source="image"),
        )
        self.assertEqual(reading.raw_value_text, "-0.40")
        self.assertIsNone(reading.unit)
        self.assertIsNone(reading.scale.minor_division)
        digital = ReaderOutput(
            meter_type="digital", value=0, value_status="readable",
            scale=ScaleInfo(status="not_applicable"),
        )
        self.assertEqual(digital.value, 0)  # Genuine zero remains distinct from missing.

    def test_observation_and_runs_are_distinct_and_never_overwrite(self):
        first, data1 = self.read(observation_id="same-observation", meter_id="plant-A")
        previous_bytes = {p.name: p.read_bytes() for p in first.parent.iterdir()}
        second, data2 = self.read(observation_id="same-observation", meter_id="plant-A")
        self.assertNotEqual(first, second)
        self.assertNotEqual(data1["run_id"], data2["run_id"])
        self.assertEqual(data1["observation_id"], data2["observation_id"])
        self.assertEqual(previous_bytes, {p.name: p.read_bytes() for p in first.parent.iterdir()})
        self.assertEqual(data2["meter_id"], "plant-A")
        self.assertIsNone(self.read()[1]["meter_id"])

    def test_same_basename_and_pathlike_id_cannot_collide(self):
        folder = self.root / "other"
        folder.mkdir()
        other = folder / self.image.name
        self.make_image(other)
        first, _ = self.read(observation_id="../../escape")
        second = run_observation(ObservationInput(image_path=other, observation_id="../../escape"),
                                 MockReader(), self.out)
        self.assertNotEqual(first.parent, second.parent)
        self.assertEqual(first.parent.parent, self.out)
        self.assertEqual(second.parent.parent, self.out)

    def test_exclusive_save_does_not_overwrite_source_or_existing_asset(self):
        before = self.image.read_bytes()
        with Image.new("RGB", (2, 2)) as image:
            with self.assertRaises(FileExistsError):
                save_png_exclusive(image, self.image)
        self.assertEqual(self.image.read_bytes(), before)
        with patch("recording.uuid4") as uuid:
            uuid.return_value.hex = "collision"
            self.read()
            with self.assertRaises(FileExistsError):
                self.read()

    def test_future_web_type_requires_meter_id(self):
        with self.assertRaises(ValidationError):
            MeterObservationInput(image_path=self.image)
        with self.assertRaises(ValidationError):
            MeterObservationInput(image_path=self.image, meter_id=" ")
        self.assertEqual(MeterObservationInput(image_path=self.image, meter_id="A").meter_id, "A")

    def test_reader_is_replaceable_and_ground_truth_never_enters_request(self):
        labels = self.root / "labels.jsonl"
        truth = '{"evaluator":"interval","evaluator_kwargs":{"low":1,"high":2},"units":["MPa","bar"]}\n'
        labels.write_text(truth)
        observed = []

        class SpyReader:
            mode = "inference"
            reader_id = "test-spy-no-network"

            def read(self, request):
                observed.append(request)
                return ReaderOutput(
                    value=0.4, raw_value_text="0.40", value_status="uncertain",
                    issue_codes=["ambiguous_scale"],
                )

        question = "上段の表示について、左側の単位で読んでください"
        request = ObservationInput(
            image_path=self.image, question=question,
            dataset={"name": "MeasureBench", "revision": "test-only", "question_id": 7,
                     "source": {"url": "source-reference"}, "category": "pressure_gauge"},
        )
        path = run_observation(request, SpyReader(), self.out)
        self.assertEqual(set(vars(observed[0])), {"image_path", "target_type", "question"})
        self.assertEqual(observed[0].question, question)
        self.assertEqual(observed[0].image_path.name, "roi.png")
        result = json.loads(path.read_text())
        self.assertEqual(result["mode"], "inference")
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["question"], question)
        self.assertEqual(result["dataset"]["question_id"], 7)
        self.assertEqual(result["reading"]["raw_value_text"], "0.40")
        self.assertEqual(labels.read_text(), truth)
        self.assertFalse((path.parent / "labels.jsonl").exists())
        for key in ("labels", "value", "evaluator", "evaluator_kwargs"):
            with self.assertRaises(ValidationError):
                ObservationInput.model_validate({"image_path": self.image, key: "not reader input"})

    def test_reader_error_is_saved_as_error_not_unreadable(self):
        class FailedReader:
            mode = "inference"
            reader_id = "test-error"

            def read(self, request):
                raise ReaderError("synthetic adapter failure")

        path = run_observation(ObservationInput(image_path=self.image), FailedReader(), self.out)
        result = json.loads(path.read_text())
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["reading"])
        self.assertIsNone(result["api_calls"])

    def test_cli_no_api_key_and_network_calls_are_forbidden(self):
        original = self.image.read_bytes()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("socket.socket", side_effect=AssertionError("network forbidden")) as socket,
            patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")) as urlopen,
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()),
        ):
            status = main([str(self.image), "--roi", "1", "1", "4", "3",
                           "--output-dir", str(self.out)])
        self.assertEqual(status, 0)
        socket.assert_not_called()
        urlopen.assert_not_called()
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["mode"], "mock")
        self.assertEqual(result["roi"], [1, 1, 4, 3])
        self.assertEqual(self.image.read_bytes(), original)

    def test_cli_legacy_roi_adapter(self):
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            status = main([str(self.image), "--legacy-roi", "1", "1", "3", "2",
                           "--output-dir", str(self.out)])
        self.assertEqual(status, 0)
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["roi"], [1, 1, 4, 3])

    def test_manifest_twenty_synthetic_inputs_and_category_metadata(self):
        categories = {"pressure_gauge": 10, "electricity_meter": 6, "flow_meter": 1,
                      "pulse_oximeter": 1, "digital_caliper": 2}
        manifest = self.root / "inputs.jsonl"
        records = []
        for category, count in categories.items():
            for number in range(count):
                records.append({
                    "image_path": self.image.name,
                    "observation_id": f"synthetic-{category}-{number}",
                    "question": f"Test target {category} {number}",
                    "dataset": {"name": "synthetic-not-MeasureBench", "category": category,
                                "revision": "fixture", "seed": 42, "question_id": len(records)},
                })
        manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
        self.assertEqual(len(load_inputs(manifest)), 20)
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            status = main(["--inputs", str(manifest), "--output-dir", str(self.out)])
        self.assertEqual(status, 0)
        paths = stdout.getvalue().splitlines()
        self.assertEqual(len(paths), 20)
        results = [json.loads(Path(path).read_text()) for path in paths]
        self.assertEqual(Counter(row["dataset"]["category"] for row in results), categories)
        self.assertEqual(len({row["run_id"] for row in results}), 20)
        self.assertTrue(all(row["status"] == "needs_review" and row["mode"] == "mock" for row in results))
        self.assertTrue(all(row["reading"]["value"] is None for row in results))

    def test_manifest_reports_lines_and_does_not_accept_labels(self):
        for text in ("", '{"image_path":"meter.png","labels":{"value":1}}\n', "not-json\n"):
            manifest = self.root / "inputs.jsonl"
            manifest.write_text(text)
            with self.assertRaises(ValueError):
                load_inputs(manifest)
        manifest.write_text('{"image_path":"meter.png"}\n{"image_path":"missing.png"}\n')
        with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            status = main(["--inputs", str(manifest), "--output-dir", str(self.out)])
        self.assertEqual(status, 1)
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        self.assertIn("inputs: 2, saved: 1, errors: 1", stderr.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--inputs", str(manifest), "--meter-id", "not-a-manifest-override"])

    def test_cli_subprocess_and_legacy_crop_cli_compatibility(self):
        env = dict(os.environ)
        env.pop("OPENAI_API_KEY", None)
        result = subprocess.run(
            [sys.executable, "src/cli.py", str(self.image), "--output-dir", str(self.out)],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(result.stdout.strip()).is_file())
        with load_image(self.image) as image:
            self.assertEqual(image.size, (6, 4))  # Legacy loader does not normalize itself.
        crop = self.root / "legacy.png"
        result = subprocess.run(
            [sys.executable, "src/crop_roi.py", str(self.image), "--roi", "1", "1", "3", "2",
             "--output", str(crop)], cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with Image.open(crop) as image:
            self.assertEqual(image.size, (3, 2))  # Existing CLI still uses xywh.


if __name__ == "__main__":
    unittest.main()
