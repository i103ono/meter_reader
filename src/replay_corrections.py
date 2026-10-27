#!/usr/bin/env python3
"""Regenerate saved fixed corrections without loading a Reader or calling an API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from uuid import uuid4

from PIL import Image
from pydantic import ValidationError

from image_io import save_png_exclusive
from quality import measure_quality
from schemas import RunResult
from tools import apply_recipe, pixel_sha256, recipe_steps, runtime_versions, validate_recipes


def _asset(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Saved image path must be relative to its run")
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Saved image path escapes its run")
    return resolved


def replay(result_path: Path, output_dir: Path = Path("outputs/replays")) -> Path:
    result_path = result_path.expanduser().resolve()
    original_document = result_path.read_bytes()
    result = RunResult.model_validate_json(original_document)
    if result.stage not in (3, 4) or not result.candidates or result.baseline_candidate_id != "c0":
        raise ValueError("Replay requires a stage-3/4 result with an original ROI")
    baseline, *corrected = result.candidates
    if (baseline.candidate_id != "c0" or baseline.recipe_id is not None
            or baseline.parent_candidate_id is not None or baseline.image_path is None):
        raise ValueError("Invalid baseline candidate")
    recipe_ids = validate_recipes([candidate.recipe_id for candidate in corrected])
    if not recipe_ids:
        raise ValueError("No saved corrections to replay")
    if any(candidate.image_path is None or candidate.status == "error" and candidate.pixel_sha256 is None
           for candidate in corrected):
        raise ValueError("Cannot verify a candidate that was not generated")
    source_roi = _asset(result_path.parent, baseline.image_path)
    roi_bytes = source_roi.read_bytes()
    roi_hash = hashlib.sha256(roi_bytes).hexdigest()
    if roi_hash != baseline.image_sha256 or roi_hash != baseline.input_roi_sha256:
        raise ValueError("Original ROI hash does not match the saved result")
    runtime = runtime_versions()
    with Image.open(source_roi) as roi:
        roi.load()
        if pixel_sha256(roi) != baseline.pixel_sha256:
            raise ValueError("Original ROI pixels do not match the saved result")
        for index, candidate in enumerate(corrected, 1):
            if candidate.candidate_id != f"c{index}" or candidate.parent_candidate_id != "c0":
                raise ValueError("Corrections must independently reference the original ROI")
            if candidate.input_roi_sha256 != roi_hash:
                raise ValueError("Candidate references a different ROI")
            expected = recipe_steps(candidate.recipe_id, roi.size)
            if [step.model_dump() for step in candidate.steps] != [step.model_dump() for step in expected]:
                raise ValueError("Saved parameters do not match the registered recipe version")
        root = output_dir.expanduser().resolve() / ("replay_" + uuid4().hex)
        root.mkdir(parents=True, exist_ok=False)
        with (root / "roi.png").open("xb") as handle:
            handle.write(roi_bytes)
        entries = []
        for candidate in corrected:
            correction = apply_recipe(roi, candidate.recipe_id)
            try:
                path = root / f"{candidate.candidate_id}_{candidate.recipe_id}.png"
                save_png_exclusive(correction.image, path)
                pixel_hash = pixel_sha256(correction.image)
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                entries.append({
                    "candidate_id": candidate.candidate_id, "recipe_id": candidate.recipe_id,
                    "image_path": path.name, "image_sha256": file_hash, "pixel_sha256": pixel_hash,
                    "pixels_match": pixel_hash == candidate.pixel_sha256,
                    "file_hash_matches": file_hash == candidate.image_sha256,
                    "runtime_matches": runtime == candidate.runtime,
                    "quality": measure_quality(correction.image).model_dump(mode="json"),
                })
            finally:
                correction.image.close()
    if source_roi.read_bytes() != roi_bytes or result_path.read_bytes() != original_document:
        raise ValueError("Source ROI or result changed during replay")
    report = {
        "source_result": str(result_path), "source_result_sha256": hashlib.sha256(original_document).hexdigest(),
        "source_run_id": result.run_id, "input_roi_sha256": roi_hash, "runtime": runtime,
        "api_calls": 0, "readings_recomputed": False,
        "all_pixels_match": all(entry["pixels_match"] for entry in entries),
        "candidates": entries,
    }
    path = root / "replay.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/replays"))
    args = parser.parse_args(argv)
    try:
        path = replay(args.result, args.output_dir)
        print(path)
        return 0 if json.loads(path.read_text())["all_pixels_match"] else 1
    except ValidationError:
        print("Error: invalid saved result contract.", file=sys.stderr)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
