#!/usr/bin/env python3
"""Stage 1 CLI. This module imports neither an OCR engine nor an AI client."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pydantic import ValidationError

from image_io import ImageLoadError, xywh_to_xyxy
from reader import MockReader
from recording import run_observation
from schemas import ObservationInput


def load_inputs(path: Path) -> list[ObservationInput]:
    """Relative image paths are relative to the manifest, never the shell cwd."""
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                record = ObservationInput.model_validate(data)
                if not record.image_path.is_absolute():
                    record = record.model_copy(update={"image_path": path.resolve().parent / record.image_path})
                records.append(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError("inputs.jsonl contains no input records")
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Save a v0.2 stage-1 mock run (no inference or API).")
    parser.add_argument("image", nargs="?", type=Path, help="One JPEG/PNG input")
    parser.add_argument("--inputs", type=Path, help="Input-only JSONL manifest (never labels)")
    roi = parser.add_mutually_exclusive_group()
    roi.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    roi.add_argument("--legacy-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                     help="Explicitly convert the legacy xywh ROI to new xyxy coordinates")
    parser.add_argument("--target-type", choices=("auto", "analog", "digital"))
    parser.add_argument("--meter-id")
    parser.add_argument("--observation-id")
    parser.add_argument("--captured-at", help="Timezone-aware ISO 8601 capture time; unknown if omitted")
    parser.add_argument("--question", help="Which display/scale to read; retained and forwarded to Reader")
    parser.add_argument("--exif-timezone", help="Explicit IANA zone for timezone-less DateTimeOriginal")
    parser.add_argument("--mode", choices=("mock",), default="mock")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args(argv)
    if (args.image is None) == (args.inputs is None):
        parser.error("provide exactly one image path or --inputs")
    if args.inputs and any(value is not None for value in (
        args.roi, args.legacy_roi, args.target_type, args.meter_id,
        args.observation_id, args.captured_at, args.question,
    )):
        parser.error("with --inputs, set per-image options in JSONL instead of CLI overrides")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.inputs:
            requests = load_inputs(args.inputs)
        else:
            roi = xywh_to_xyxy(args.legacy_roi) if args.legacy_roi is not None else args.roi
            requests = [ObservationInput(
                image_path=args.image, roi=roi, target_type=args.target_type or "auto",
                meter_id=args.meter_id, observation_id=args.observation_id,
                captured_at=args.captured_at, question=args.question,
            )]
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    failed = 0
    for index, request in enumerate(requests, 1):
        try:
            result = run_observation(request, MockReader(), args.output_dir, exif_timezone=args.exif_timezone)
            print(result, flush=True)
        except (OSError, ValueError, ImageLoadError) as exc:
            failed += 1
            print(f"Error: input {index} ({request.image_path}): {exc}", file=sys.stderr)
    print(f"inputs: {len(requests)}, saved: {len(requests) - failed}, errors: {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
