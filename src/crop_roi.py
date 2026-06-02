#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import ImageOps

from image_io import ImageLoadError, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a region of interest (ROI) from an image."
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
        required=True,
        help="Output path for the cropped image (e.g., outputs/meter_0005_roi.png)",
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
                import cv2  # type: ignore
                import numpy as np
            except ModuleNotFoundError as exc:
                print(
                    f"Error: Interactive ROI selection requires {exc.name}.",
                    file=sys.stderr,
                )
                return 1

            display = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            roi = cv2.selectROI(
                "Select ROI", display, showCrosshair=True, fromCenter=False
            )
            cv2.destroyAllWindows()
            x, y, w, h = (int(value) for value in roi)
            if w == 0 or h == 0:
                print("Error: No ROI selected.", file=sys.stderr)
                return 1
        else:
            x, y, w, h = args.roi

        if w <= 0 or h <= 0:
            print(
                "Error: ROI width and height must be greater than 0.",
                file=sys.stderr,
            )
            return 1
        if x < 0 or y < 0:
            print("Error: ROI x and y must be 0 or greater.", file=sys.stderr)
            return 1
        if x + w > img.width or y + h > img.height:
            print(
                "Error: ROI is outside the image bounds "
                f"(image size: {img.width}x{img.height}).",
                file=sys.stderr,
            )
            return 1
        cropped = img.crop((x, y, x + w, y + h))
    finally:
        img.close()

    output_path = Path(args.output).expanduser()
    output_dir = output_path.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)
    except OSError as exc:
        print(f"Error: Failed to save cropped image: {exc}", file=sys.stderr)
        return 1
    finally:
        cropped.close()

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
