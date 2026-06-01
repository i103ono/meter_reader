#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from image_io import ImageLoadError, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a single image and save grayscale/binary outputs."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
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

    try:
        img = load_image(input_path)
    except ImageLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        gray = img.convert("L")
        binary = gray.point(lambda value: 255 if value >= 128 else 0)
    finally:
        img.close()

    outputs_dir = resolve_outputs_dir()
    stem = input_path.stem
    gray_path = outputs_dir / f"{stem}_gray.png"
    binary_path = outputs_dir / f"{stem}_binary.png"

    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        gray.save(gray_path)
        binary.save(binary_path)
    except OSError as exc:
        print(
            f"Error: Failed to save preprocessed images: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Saved: {gray_path}")
    print(f"Saved: {binary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
