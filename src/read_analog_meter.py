#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageOps

from analog_calibration import calculate_value
from image_io import ImageLoadError, load_image
from detect_needle import endpoint_angle, endpoint_distance, point_segment_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read an analog meter from one image.")
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
    parser.add_argument("--unit", default="")
    parser.add_argument("--min-value", type=float, required=True)
    parser.add_argument("--max-value", type=float, required=True)
    parser.add_argument("--min-angle", type=float, default=225.0)
    parser.add_argument("--max-angle", type=float, default=345.0)
    parser.add_argument(
        "--direction",
        choices=("clockwise", "counterclockwise"),
        default="counterclockwise",
    )
    parser.add_argument("--endpoint-tolerance", type=float, default=5.0)
    parser.add_argument("--center-x", type=int)
    parser.add_argument("--center-y", type=int)
    parser.add_argument("--radius", type=int)
    parser.add_argument("--dial-min-radius", type=int, default=20)
    parser.add_argument("--dial-max-radius", type=int, default=0)
    parser.add_argument("--debug-output")
    return parser.parse_args()


def resolve_debug_output(input_path: Path) -> Path:
    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    return outputs_dir / f"{input_path.stem}_analog_debug.png"


def validate_args(args: argparse.Namespace) -> str | None:
    provided = [args.center_x is not None, args.center_y is not None, args.radius is not None]
    if any(provided) and not all(provided):
        return "--center-x, --center-y, and --radius must be specified together."
    if args.radius is not None and args.radius <= 0:
        return "--radius must be greater than 0."
    if args.dial_min_radius < 0 or args.dial_max_radius < 0:
        return "--dial-min-radius and --dial-max-radius must be 0 or greater."
    if args.dial_max_radius and args.dial_max_radius < args.dial_min_radius:
        return "--dial-max-radius must be greater than or equal to --dial-min-radius."
    for name in ("min_value", "max_value", "min_angle", "max_angle", "endpoint_tolerance"):
        if not math.isfinite(getattr(args, name)):
            return f"--{name.replace('_', '-')} must be a finite number."
    return None


def detect_dial(gray: np.ndarray, min_radius: int, max_radius: int) -> tuple[int, int, int]:
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100.0,
        param1=100.0,
        param2=30.0,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None or len(circles[0]) == 0:
        raise RuntimeError("No dial circle detected.")
    x, y, radius = np.round(circles[0][0]).astype(int)
    return int(x), int(y), int(radius)


def detect_needle(
    gray: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> tuple[float, tuple[int, int, int, int], int]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50.0, 150.0)
    mask = np.zeros_like(edges)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=max(20, int(radius * 0.08)),
        maxLineGap=20,
    )
    if lines is None:
        raise RuntimeError("No needle candidate lines detected.")

    center_tolerance = max(12.0, radius * 0.12)
    candidates: list[tuple[float, float, tuple[int, int, int, int]]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = (int(value) for value in line)
        distance = point_segment_distance(center_x, center_y, x1, y1, x2, y2)
        if distance > center_tolerance:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        far_distance = endpoint_distance(center_x, center_y, x1, y1, x2, y2)
        angle = endpoint_angle(center_x, center_y, x1, y1, x2, y2)
        candidates.append((far_distance * max(1.0, length), angle, (x1, y1, x2, y2)))

    if not candidates:
        raise RuntimeError("No needle candidate lines pass center filtering.")
    candidates.sort(reverse=True, key=lambda item: item[0])
    _score, angle, line = candidates[0]
    return angle, line, len(candidates)


def save_debug(
    bgr: np.ndarray,
    output_path: Path,
    center_x: int,
    center_y: int,
    radius: int,
    angle: float,
    line: tuple[int, int, int, int],
) -> None:
    debug = bgr.copy()
    x1, y1, x2, y2 = line
    tip_length = int(radius * 0.45)
    tip_x = int(round(center_x + math.cos(math.radians(angle)) * tip_length))
    tip_y = int(round(center_y - math.sin(math.radians(angle)) * tip_length))
    cv2.circle(debug, (center_x, center_y), radius, (0, 255, 0), 2)
    cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.arrowedLine(debug, (center_x, center_y), (tip_x, tip_y), (0, 0, 255), 4)
    cv2.circle(debug, (center_x, center_y), 4, (255, 0, 0), -1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), debug):
        raise OSError(f"Failed to save debug image: {output_path}")


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

    try:
        if args.center_x is None:
            center_x, center_y, radius = detect_dial(
                gray,
                min_radius=args.dial_min_radius,
                max_radius=args.dial_max_radius,
            )
        else:
            center_x, center_y, radius = args.center_x, args.center_y, args.radius
        angle, line, candidate_count = detect_needle(gray, center_x, center_y, radius)
        value = calculate_value(
            angle=angle,
            min_value=args.min_value,
            max_value=args.max_value,
            min_angle=args.min_angle,
            max_angle=args.max_angle,
            direction=args.direction,
            endpoint_tolerance=args.endpoint_tolerance,
        )
        debug_output = (
            Path(args.debug_output).expanduser()
            if args.debug_output
            else resolve_debug_output(input_path)
        )
        save_debug(bgr, debug_output, center_x, center_y, radius, angle, line)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"value: {value:g}")
    print(f"unit: {args.unit}")
    print(f"angle_degree: {angle:.1f}")
    print(f"center_x: {center_x}")
    print(f"center_y: {center_y}")
    print(f"radius: {radius}")
    print(f"candidate_lines: {candidate_count}")
    print(f"debug_image: {debug_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
