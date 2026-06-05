#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from easyocr_engine import OcrError, read_text
from parse_reading import parse_reading
from read_analog_meter import (
    detect_dial,
    detect_needle,
    resolve_debug_output,
    save_debug,
    validate_args as validate_analog_args,
)
from analog_calibration import calculate_value
from image_io import ImageLoadError, load_image

import cv2
import numpy as np
from PIL import ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read a digital or analog meter.")
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument("--type", choices=("digital", "analog"), required=True)
    parser.add_argument("--unit", default="")
    parser.add_argument("--digits-only", action="store_true")
    parser.add_argument("--gpu", action="store_true")
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
    parser.add_argument("--center-x", type=int)
    parser.add_argument("--center-y", type=int)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--dial-min-radius", type=int, default=20)
    parser.add_argument("--dial-max-radius", type=int, default=0)
    parser.add_argument("--debug-output")
    return parser.parse_args()


def validate_input(input_path: Path) -> str | None:
    if not input_path.exists():
        return f"Input file not found: {input_path}"
    if not input_path.is_file():
        return f"Input path is not a file: {input_path}"
    return None


def read_digital(input_path: Path, digits_only: bool, gpu: bool) -> int:
    try:
        raw_text, confidence = read_text(input_path, digits_only=digits_only, gpu=gpu)
    except OcrError as exc:
        print(f"Error: OCR failed: {exc}", file=sys.stderr)
        return 1

    if not raw_text.strip():
        print("Error: No text detected.", file=sys.stderr)
        return 1

    try:
        reading = parse_reading(raw_text)
    except ValueError as exc:
        print(f"Error: Failed to parse OCR text: {exc}", file=sys.stderr)
        print(f"raw_text: {raw_text}", file=sys.stderr)
        return 1

    print("type: digital")
    print(f"value: {reading.value:g}")
    print(f"unit: {reading.unit or ''}")
    print(f"raw_text: {raw_text}")
    if confidence is not None:
        print(f"confidence: {confidence:.1f}")
    return 0


def read_analog(args: argparse.Namespace, input_path: Path) -> int:
    if args.min_value is None or args.max_value is None:
        print(
            "Error: --min-value and --max-value are required for analog meters.",
            file=sys.stderr,
        )
        return 1

    error = validate_analog_args(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    try:
        img = load_image(input_path)
    except ImageLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

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

    try:
        if args.center_x is None:
            center_x, center_y, radius = detect_dial(
                gray,
                min_radius=args.dial_min_radius,
                max_radius=args.dial_max_radius,
            )
        else:
            center_x, center_y, radius = args.center_x, args.center_y, args.radius
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
        debug_output = (
            Path(args.debug_output).expanduser()
            if args.debug_output
            else resolve_debug_output(input_path)
        )
        save_debug(bgr, debug_output, center_x, center_y, radius, angle, line)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("type: analog")
    print(f"value: {value:g}")
    print(f"unit: {args.unit}")
    print(f"angle_degree: {angle:.1f}")
    print(f"center_x: {center_x}")
    print(f"center_y: {center_y}")
    print(f"radius: {radius}")
    print(f"candidate_lines: {candidate_count}")
    print(f"debug_image: {debug_output}")
    return 0


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    error = validate_input(input_path)
    if error:
        print(error, file=sys.stderr)
        return 1

    if args.type == "digital":
        return read_digital(input_path, digits_only=args.digits_only, gpu=args.gpu)
    return read_analog(args, input_path)


if __name__ == "__main__":
    raise SystemExit(main())
