#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
    parser.add_argument("--config", help="Meter config JSON path.")
    parser.add_argument("--type", choices=("digital", "analog"))
    parser.add_argument("--unit")
    parser.add_argument("--digits-only", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--min-value", type=float)
    parser.add_argument("--max-value", type=float)
    parser.add_argument("--min-angle", type=float)
    parser.add_argument("--max-angle", type=float)
    parser.add_argument(
        "--direction",
        choices=("clockwise", "counterclockwise"),
    )
    parser.add_argument("--endpoint-tolerance", type=float, default=5.0)
    parser.add_argument("--center-x", type=int)
    parser.add_argument("--center-y", type=int)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--dial-min-radius", type=int, default=20)
    parser.add_argument("--dial-max-radius", type=int, default=0)
    parser.add_argument("--debug-output")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Config file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Config path is not a file: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Failed to read config: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid config JSON: {path} (line {exc.lineno}, column {exc.colno})"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError("Config must contain a JSON object.")
    return config


def apply_config(args: argparse.Namespace, config: dict[str, Any]) -> None:
    config_fields = {
        "meter_type": "type",
        "unit": "unit",
        "min_value": "min_value",
        "max_value": "max_value",
        "min_angle": "min_angle",
        "max_angle": "max_angle",
        "direction": "direction",
    }
    for config_key, arg_name in config_fields.items():
        if getattr(args, arg_name) is None and config.get(config_key) is not None:
            setattr(args, arg_name, config[config_key])

    if args.type not in ("digital", "analog"):
        raise ValueError("meter_type must be 'digital' or 'analog'.")
    if args.direction not in (None, "clockwise", "counterclockwise"):
        raise ValueError("direction must be 'clockwise' or 'counterclockwise'.")

    for key in ("min_value", "max_value", "min_angle", "max_angle"):
        value = getattr(args, key)
        if value is None:
            continue
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a number.")
        try:
            setattr(args, key, float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number.") from exc

    args.unit = "" if args.unit is None else str(args.unit)
    args.min_angle = 225.0 if args.min_angle is None else args.min_angle
    args.max_angle = 345.0 if args.max_angle is None else args.max_angle
    args.direction = args.direction or "counterclockwise"


def crop_config_roi(input_path: Path, roi: Any) -> Path:
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("roi must be a list of [x, y, width, height].")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in roi):
        raise ValueError("roi values must be integers.")
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("roi must have x/y >= 0 and width/height > 0.")

    try:
        img = load_image(input_path)
    except ImageLoadError as exc:
        raise ValueError(str(exc)) from exc
    try:
        transposed = ImageOps.exif_transpose(img)
        if transposed is not img:
            img.close()
            img = transposed
        if x + width > img.width or y + height > img.height:
            raise ValueError(
                f"roi is outside the image bounds (image size: {img.width}x{img.height})."
            )
        cropped = img.crop((x, y, x + width, y + height))
    finally:
        img.close()

    output_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / f"{input_path.stem}_config_roi.png"
    )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)
    except OSError as exc:
        raise ValueError(f"Failed to save config ROI: {output_path} ({exc})") from exc
    finally:
        cropped.close()
    return output_path


def validate_input(input_path: Path) -> str | None:
    if not input_path.exists():
        return f"Input file not found: {input_path}"
    if not input_path.is_file():
        return f"Input path is not a file: {input_path}"
    return None


def read_digital(input_path: Path, digits_only: bool, gpu: bool, unit: str = "") -> int:
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
    print(f"unit: {unit or reading.unit or ''}")
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

    config: dict[str, Any] = {}
    try:
        if args.config:
            config = load_config(Path(args.config).expanduser())
        apply_config(args, config)
        if config.get("roi") is not None:
            input_path = crop_config_roi(input_path, config["roi"])
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.type == "digital":
        return read_digital(
            input_path,
            digits_only=args.digits_only,
            gpu=args.gpu,
            unit=args.unit,
        )
    return read_analog(args, input_path)


if __name__ == "__main__":
    raise SystemExit(main())
