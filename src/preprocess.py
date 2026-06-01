#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image

from image_io import ImageLoadError, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a single image and save grayscale/binary outputs."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Threshold value for binarization (0-255).",
    )
    parser.add_argument(
        "--resize",
        type=float,
        default=1.0,
        help="Resize scale factor (e.g., 0.5 or 2.0).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images to outputs/.",
    )
    return parser.parse_args()


def resolve_outputs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"Input path is not a file: {input_path}", file=sys.stderr)
        return 1

    if not 0 <= args.threshold <= 255:
        print("Error: --threshold must be between 0 and 255.", file=sys.stderr)
        return 1
    if not math.isfinite(args.resize) or args.resize <= 0:
        print(
            "Error: --resize must be a finite number greater than 0.",
            file=sys.stderr,
        )
        return 1

    try:
        img = load_image(input_path)
    except ImageLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.resize != 1.0:
        width, height = img.size
        new_width = max(1, int(round(width * args.resize)))
        new_height = max(1, int(round(height * args.resize)))
        original = img
        img = original.resize((new_width, new_height), resample=Image.LANCZOS)
        original.close()

    gray = img.convert("L")
    binary = gray.point(lambda value: 255 if value >= args.threshold else 0)

    outputs_dir = resolve_outputs_dir()
    stem = input_path.stem
    gray_path = outputs_dir / f"{stem}_gray.png"
    binary_path = outputs_dir / f"{stem}_binary.png"
    debug_path = outputs_dir / f"{stem}_resized.png"

    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        if args.debug:
            img.save(debug_path)
        gray.save(gray_path)
        binary.save(binary_path)
    except OSError as exc:
        print(
            f"Error: Failed to save preprocessed images: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        img.close()
        gray.close()
        binary.close()

    if args.debug:
        print(f"Saved: {debug_path}")
    print(f"Saved: {gray_path}")
    print(f"Saved: {binary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
