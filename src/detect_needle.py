#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageOps

from image_io import ImageLoadError, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect an analog meter needle and estimate its angle."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument("--center-x", type=int, required=True)
    parser.add_argument("--center-y", type=int, required=True)
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--threshold1", type=float, default=50.0)
    parser.add_argument("--threshold2", type=float, default=150.0)
    parser.add_argument("--hough-threshold", type=int, default=20)
    parser.add_argument("--min-line-length", type=int)
    parser.add_argument("--max-line-gap", type=int, default=20)
    parser.add_argument("--center-tolerance", type=float)
    parser.add_argument(
        "--debug-output",
        help="Debug image path. Default: outputs/<input_stem>_needle_debug.png",
    )
    return parser.parse_args()


def resolve_debug_output(input_path: Path) -> Path:
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    return outputs_dir / f"{input_path.stem}_needle_debug.png"


def validate_args(args: argparse.Namespace) -> str | None:
    if args.radius <= 0:
        return "--radius must be greater than 0."
    if args.center_x < 0 or args.center_y < 0:
        return "--center-x and --center-y must be 0 or greater."
    if args.threshold1 <= 0 or args.threshold2 <= 0:
        return "--threshold1 and --threshold2 must be greater than 0."
    if args.hough_threshold <= 0:
        return "--hough-threshold must be greater than 0."
    if args.min_line_length is not None and args.min_line_length <= 0:
        return "--min-line-length must be greater than 0."
    if args.max_line_gap < 0:
        return "--max-line-gap must be 0 or greater."
    if args.center_tolerance is not None and args.center_tolerance < 0:
        return "--center-tolerance must be 0 or greater."
    return None


def point_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    projection = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    projection = max(0.0, min(1.0, projection))
    nearest_x = x1 + projection * dx
    nearest_y = y1 + projection * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def endpoint_angle(
    center_x: int,
    center_y: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    first_distance = math.hypot(x1 - center_x, y1 - center_y)
    second_distance = math.hypot(x2 - center_x, y2 - center_y)
    needle_x, needle_y = (x1, y1) if first_distance >= second_distance else (x2, y2)
    angle = math.degrees(math.atan2(center_y - needle_y, needle_x - center_x))
    return angle % 360


def endpoint_distance(
    center_x: int,
    center_y: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    return max(
        math.hypot(x1 - center_x, y1 - center_y),
        math.hypot(x2 - center_x, y2 - center_y),
    )


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

    height, width = rgb.shape[:2]
    if args.center_x >= width or args.center_y >= height:
        print(
            f"Error: Center is outside the image bounds (image size: {width}x{height}).",
            file=sys.stderr,
        )
        return 1

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, args.threshold1, args.threshold2)
    mask = np.zeros_like(edges)
    cv2.circle(mask, (args.center_x, args.center_y), args.radius, 255, -1)
    edges = cv2.bitwise_and(edges, mask)

    min_line_length = args.min_line_length or max(20, int(args.radius * 0.08))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=args.hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=args.max_line_gap,
    )
    if lines is None:
        print("Error: No needle candidate lines detected.", file=sys.stderr)
        return 1

    center_tolerance = (
        args.center_tolerance
        if args.center_tolerance is not None
        else max(12.0, args.radius * 0.12)
    )
    candidates: list[tuple[float, float, tuple[int, int, int, int]]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = (int(value) for value in line)
        distance = point_segment_distance(args.center_x, args.center_y, x1, y1, x2, y2)
        if distance > center_tolerance:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        far_distance = endpoint_distance(args.center_x, args.center_y, x1, y1, x2, y2)
        angle = endpoint_angle(args.center_x, args.center_y, x1, y1, x2, y2)
        score = far_distance * max(1.0, length)
        candidates.append((score, angle, (x1, y1, x2, y2)))

    if not candidates:
        print("Error: No needle candidate lines pass center filtering.", file=sys.stderr)
        return 1

    candidates.sort(reverse=True, key=lambda item: item[0])
    _score, angle, best_line = candidates[0]
    x1, y1, x2, y2 = best_line

    debug = bgr.copy()
    tip_length = int(args.radius * 0.45)
    tip_x = int(round(args.center_x + math.cos(math.radians(angle)) * tip_length))
    tip_y = int(round(args.center_y - math.sin(math.radians(angle)) * tip_length))
    cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.arrowedLine(
        debug,
        (args.center_x, args.center_y),
        (tip_x, tip_y),
        (0, 0, 255),
        4,
        tipLength=0.18,
    )
    cv2.circle(debug, (args.center_x, args.center_y), 4, (255, 0, 0), -1)

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

    print(f"candidate_lines: {len(candidates)}")
    print(f"needle_line: {x1},{y1},{x2},{y2}")
    print(f"angle_degree: {angle:.1f}")
    print(f"debug_image: {debug_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
