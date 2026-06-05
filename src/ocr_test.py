#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from easyocr_engine import OcrError, read_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EasyOCR on a single image.")
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument(
        "--psm",
        type=int,
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
        text, _confidence = read_text(
            input_path,
            digits_only=args.digits_only,
            gpu=args.gpu,
        )
    except OcrError as exc:
        print(f"Error: OCR failed: {exc}", file=sys.stderr)
        return 1

    if not text.strip():
        print("No text detected.")
        return 0

    print(text.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
