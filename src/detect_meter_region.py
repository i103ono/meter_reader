#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import ImageOps

from image_io import ImageLoadError, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop an analog meter region from an image."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="ROI coordinates: x y w h",
    )
    roi_group.add_argument(
        "--interactive",
        action="store_true",
        help="Select ROI using a mouse-driven window.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output path for the cropped image. "
            "Default: outputs/<input_stem>_meter_region.png"
        ),
    )
    return parser.parse_args()


def resolve_default_output(input_path: Path) -> Path:
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    return outputs_dir / f"{input_path.stem}_meter_region.png"


def select_interactive_roi(img) -> tuple[int, int, int, int]:
    try:
        import cv2  # type: ignore
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Interactive ROI selection requires {exc.name}.") from exc

    display = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    roi = cv2.selectROI("Select meter region", display, showCrosshair=True)
    cv2.destroyAllWindows()
    return tuple(int(value) for value in roi)


def validate_roi(x: int, y: int, w: int, h: int, width: int, height: int) -> str | None:
    if w <= 0 or h <= 0:
        return "ROI width and height must be greater than 0."
    if x < 0 or y < 0:
        return "ROI x and y must be 0 or greater."
    if x + w > width or y + h > height:
        return f"ROI is outside the image bounds (image size: {width}x{height})."
    return None


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
        transposed = ImageOps.exif_transpose(img)
        if transposed is not img:
            img.close()
            img = transposed

        if args.interactive:
            try:
                x, y, w, h = select_interactive_roi(img)
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            if w == 0 or h == 0:
                print("Error: No ROI selected.", file=sys.stderr)
                return 1
        else:
            x, y, w, h = args.roi

        error = validate_roi(x, y, w, h, img.width, img.height)
        if error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        cropped = img.crop((x, y, x + w, y + h))
    finally:
        img.close()

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else resolve_default_output(input_path)
    )
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)
    except OSError as exc:
        print(f"Error: Failed to save meter region: {exc}", file=sys.stderr)
        return 1
    finally:
        cropped.close()

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
