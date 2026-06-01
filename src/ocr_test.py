#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytesseract

from image_io import ImageLoadError, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR on a single image.")
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument(
        "--psm",
        type=int,
        help="Tesseract page segmentation mode (PSM).",
    )
    parser.add_argument(
        "--digits-only",
        action="store_true",
        help="Recognize digits and dot only (0123456789.).",
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

    if args.psm is not None and not 0 <= args.psm <= 13:
        print("Error: --psm must be between 0 and 13.", file=sys.stderr)
        return 1

    try:
        img = load_image(input_path)
    except ImageLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    config_parts: list[str] = []
    if args.psm is not None:
        config_parts.append(f"--psm {args.psm}")
    if args.digits_only:
        config_parts.append("-c tessedit_char_whitelist=0123456789.")
    config = " ".join(config_parts)

    try:
        text = pytesseract.image_to_string(img, config=config)
    except (pytesseract.TesseractError, pytesseract.TesseractNotFoundError) as exc:
        print(f"Error: OCR failed: {exc}", file=sys.stderr)
        return 1
    finally:
        img.close()

    if not text.strip():
        print("No text detected.")
        return 0

    print(text.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
