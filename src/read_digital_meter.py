#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from easyocr_engine import OcrError, read_text
from parse_reading import parse_reading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read a digital meter value from a single image."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument(
        "--psm",
        type=int,
        default=7,
        help="Accepted for compatibility; ignored by EasyOCR.",
    )
    parser.add_argument(
        "--digits-only",
        action="store_true",
        help="Recognize digits and dot only (0123456789.).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for EasyOCR if available.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"Input path is not a file: {input_path}", file=sys.stderr)
        return 1
    try:
        raw_text, confidence = read_text(
            input_path,
            digits_only=args.digits_only,
            gpu=args.gpu,
        )
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

    print(f"raw_text: {raw_text}")
    print(f"value: {reading.value:g}")
    print(f"unit: {reading.unit or ''}")
    if confidence is not None:
        print(f"confidence: {confidence:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
