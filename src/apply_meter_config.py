#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import ImageOps

from image_io import ImageLoadError, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop an image using roi from a meter config JSON."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument("config", help="Meter config JSON path.")
    parser.add_argument(
        "--output",
        help="Output path. Default: outputs/<input_stem>_config_roi.png",
    )
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


def parse_roi(value: Any) -> tuple[int, int, int, int]:
    if value is None:
        raise ValueError("Config does not contain roi.")
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("roi must be a list of [x, y, width, height].")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("roi values must be integers.")
    x, y, width, height = value
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("roi must have x/y >= 0 and width/height > 0.")
    return x, y, width, height


def default_output_path(input_path: Path) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / f"{input_path.stem}_config_roi.png"
    )


def crop_image(
    input_path: Path,
    roi: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    try:
        img = load_image(input_path)
    except ImageLoadError as exc:
        raise ValueError(str(exc)) from exc

    try:
        transposed = ImageOps.exif_transpose(img)
        if transposed is not img:
            img.close()
            img = transposed
        x, y, width, height = roi
        if x + width > img.width or y + height > img.height:
            raise ValueError(
                f"roi is outside the image bounds (image size: {img.width}x{img.height})."
            )
        cropped = img.crop((x, y, x + width, y + height))
    finally:
        img.close()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)
    except OSError as exc:
        raise ValueError(f"Failed to save cropped image: {output_path} ({exc})") from exc
    finally:
        cropped.close()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    config_path = Path(args.config).expanduser()
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else default_output_path(input_path)
    )

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}", file=sys.stderr)
        return 1

    try:
        config = load_config(config_path)
        roi = parse_roi(config.get("roi"))
        crop_image(input_path, roi, output_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
