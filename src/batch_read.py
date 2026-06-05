#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageOps

from analog_calibration import calculate_value
from easyocr_engine import OcrError, read_text
from image_io import ImageLoadError, load_image
from parse_reading import parse_reading
from read_analog_meter import detect_dial, detect_needle, resolve_debug_output, save_debug


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FIELDNAMES = [
    "filename",
    "type",
    "status",
    "value",
    "unit",
    "error",
    "raw_text",
    "confidence",
    "angle_degree",
    "center_x",
    "center_y",
    "radius",
    "debug_image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-read images and save CSV results.")
    parser.add_argument(
        "--images-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "images"),
        help="Image directory. Default: data/images",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "results.csv"),
        help="Output CSV path. Default: outputs/results.csv",
    )
    parser.add_argument("--type", choices=("digital", "analog"), required=True)
    parser.add_argument("--unit", default="")
    parser.add_argument("--digits-only", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=1280,
        help="EasyOCR canvas size for digital OCR. Lower is faster. Default: 1280.",
    )
    parser.add_argument("--min-value", type=float)
    parser.add_argument("--max-value", type=float)
    parser.add_argument("--min-angle", type=float, default=225.0)
    parser.add_argument("--max-angle", type=float, default=345.0)
    parser.add_argument(
        "--direction",
        choices=("clockwise", "counterclockwise"),
        default="counterclockwise",
    )
    parser.add_argument("--endpoint-tolerance", type=float, default=5.0)
    parser.add_argument("--dial-min-radius", type=int, default=20)
    parser.add_argument("--dial-max-radius", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Process only the first N images.")
    return parser.parse_args()


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        ],
        key=lambda path: path.name.casefold(),
    )


def empty_row(filename: str, meter_type: str) -> dict[str, str]:
    return {field: "" for field in FIELDNAMES} | {
        "filename": filename,
        "type": meter_type,
    }


def read_digital_row(path: Path, args: argparse.Namespace) -> dict[str, str]:
    row = empty_row(path.name, args.type)
    try:
        raw_text, confidence = read_text(
            path,
            digits_only=args.digits_only,
            gpu=args.gpu,
            canvas_size=args.canvas_size,
        )
        if not raw_text.strip():
            raise RuntimeError("No text detected.")
        reading = parse_reading(raw_text)
    except (OcrError, RuntimeError, ValueError) as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        return row

    row["status"] = "ok"
    row["value"] = f"{reading.value:g}"
    row["unit"] = reading.unit or ""
    row["raw_text"] = raw_text
    if confidence is not None:
        row["confidence"] = f"{confidence:.3f}"
    return row


def read_analog_row(path: Path, args: argparse.Namespace) -> dict[str, str]:
    row = empty_row(path.name, args.type)
    if args.min_value is None or args.max_value is None:
        row["status"] = "error"
        row["error"] = "--min-value and --max-value are required for analog meters."
        return row

    try:
        img = load_image(path)
        try:
            transposed = ImageOps.exif_transpose(img)
            if transposed is not img:
                img.close()
                img = transposed
            rgb = np.array(img.convert("RGB"))
        finally:
            img.close()

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        center_x, center_y, radius = detect_dial(
            gray,
            min_radius=args.dial_min_radius,
            max_radius=args.dial_max_radius,
        )
        angle, line, candidate_count = detect_needle(gray, center_x, center_y, radius)
        value = calculate_value(
            angle=angle,
            min_value=args.min_value,
            max_value=args.max_value,
            min_angle=args.min_angle,
            max_angle=args.max_angle,
            direction=args.direction,
            endpoint_tolerance=args.endpoint_tolerance,
        )
        debug_output = resolve_debug_output(path)
        save_debug(bgr, debug_output, center_x, center_y, radius, angle, line)
    except (ImageLoadError, RuntimeError, OSError, ValueError) as exc:
        row["status"] = "error"
        row["error"] = str(exc)
        return row

    row["status"] = "ok"
    row["value"] = f"{value:g}"
    row["unit"] = args.unit
    row["angle_degree"] = f"{angle:.1f}"
    row["center_x"] = str(center_x)
    row["center_y"] = str(center_y)
    row["radius"] = str(radius)
    row["debug_image"] = str(debug_output)
    row["raw_text"] = f"candidate_lines={candidate_count}"
    return row


def validate_args(args: argparse.Namespace) -> str | None:
    images_dir = Path(args.images_dir).expanduser()
    if not images_dir.exists():
        return f"Image directory not found: {images_dir}"
    if not images_dir.is_dir():
        return f"Image path is not a directory: {images_dir}"
    if args.limit is not None and args.limit <= 0:
        return "--limit must be greater than 0."
    if args.canvas_size <= 0:
        return "--canvas-size must be greater than 0."
    for name in ("min_value", "max_value", "min_angle", "max_angle", "endpoint_tolerance"):
        value = getattr(args, name)
        if value is not None and not math.isfinite(value):
            return f"--{name.replace('_', '-')} must be a finite number."
    return None


def main() -> int:
    args = parse_args()
    error = validate_args(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    images_dir = Path(args.images_dir).expanduser()
    images = collect_images(images_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        print("No matching image files found.")
        return 0

    rows = []
    for index, path in enumerate(images, start=1):
        print(f"Processing {index}/{len(images)}: {path.name}", flush=True)
        if args.type == "digital":
            rows.append(read_digital_row(path, args))
        else:
            rows.append(read_analog_row(path, args))

    output_path = Path(args.output).expanduser()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        print(f"Error: Failed to save results CSV: {exc}", file=sys.stderr)
        return 1

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"Saved: {output_path}")
    print(f"processed: {len(rows)}")
    print(f"ok: {ok_count}")
    print(f"error: {len(rows) - ok_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
