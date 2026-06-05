#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageOps

from image_io import ImageLoadError, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a circular analog meter dial."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument("--min-radius", type=int, default=20)
    parser.add_argument("--max-radius", type=int, default=0)
    parser.add_argument("--dp", type=float, default=1.2)
    parser.add_argument("--min-dist", type=float, default=100.0)
    parser.add_argument("--param1", type=float, default=100.0)
    parser.add_argument("--param2", type=float, default=30.0)
    parser.add_argument(
        "--debug-output",
        help="Debug image path. Default: outputs/<input_stem>_dial_debug.png",
    )
    return parser.parse_args()


def resolve_debug_output(input_path: Path) -> Path:
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    return outputs_dir / f"{input_path.stem}_dial_debug.png"


def validate_args(args: argparse.Namespace) -> str | None:
    if args.min_radius < 0:
        return "--min-radius must be 0 or greater."
    if args.max_radius < 0:
        return "--max-radius must be 0 or greater."
    if args.max_radius and args.max_radius < args.min_radius:
        return "--max-radius must be greater than or equal to --min-radius."
    if args.dp <= 0:
        return "--dp must be greater than 0."
    if args.min_dist <= 0:
        return "--min-dist must be greater than 0."
    if args.param1 <= 0 or args.param2 <= 0:
        return "--param1 and --param2 must be greater than 0."
    return None


def main() -> int:
    args = parse_args()
    error = validate_args(args)
    if error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

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

        rgb = np.array(img.convert("RGB"))
    finally:
        img.close()

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=args.dp,
        minDist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        minRadius=args.min_radius,
        maxRadius=args.max_radius,
    )
    if circles is None or len(circles[0]) == 0:
        print("Error: No dial circle detected.", file=sys.stderr)
        return 1

    x, y, radius = np.round(circles[0][0]).astype(int)
    debug = bgr.copy()
    cv2.circle(debug, (x, y), radius, (0, 255, 0), 3)
    cv2.circle(debug, (x, y), 3, (0, 0, 255), -1)

    debug_output = (
        Path(args.debug_output).expanduser()
        if args.debug_output
        else resolve_debug_output(input_path)
    )
    try:
        debug_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(debug_output), debug):
            print(f"Error: Failed to save debug image: {debug_output}", file=sys.stderr)
            return 1
    except OSError as exc:
        print(f"Error: Failed to save debug image: {exc}", file=sys.stderr)
        return 1

    print(f"center_x: {x}")
    print(f"center_y: {y}")
    print(f"radius: {radius}")
    print(f"debug_image: {debug_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
