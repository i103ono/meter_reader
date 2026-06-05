#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert analog meter needle angle to a meter value."
    )
    parser.add_argument("--angle", type=float, required=True)
    parser.add_argument("--min-value", type=float, required=True)
    parser.add_argument("--max-value", type=float, required=True)
    parser.add_argument("--min-angle", type=float, required=True)
    parser.add_argument("--max-angle", type=float, required=True)
    parser.add_argument(
        "--direction",
        choices=("clockwise", "counterclockwise"),
        default="clockwise",
        help="Direction from min-angle to max-angle. Default: clockwise.",
    )
    parser.add_argument(
        "--endpoint-tolerance",
        type=float,
        default=5.0,
        help="Clamp angles within this many degrees of min/max angles. Default: 5.",
    )
    return parser.parse_args()


def normalize_angle(angle: float) -> float:
    return angle % 360.0


def clockwise_delta(start_angle: float, end_angle: float) -> float:
    return (normalize_angle(end_angle) - normalize_angle(start_angle)) % 360.0


def angle_delta(start_angle: float, end_angle: float, direction: str) -> float:
    if direction == "clockwise":
        return clockwise_delta(start_angle, end_angle)
    return clockwise_delta(end_angle, start_angle)


def angular_distance(first_angle: float, second_angle: float) -> float:
    difference = abs(normalize_angle(first_angle) - normalize_angle(second_angle))
    return min(difference, 360.0 - difference)


def is_left_half(angle: float) -> bool:
    normalized = normalize_angle(angle)
    return 90.0 <= normalized <= 270.0


def should_clamp_endpoint(angle: float, endpoint_angle: float, tolerance: float) -> bool:
    return (
        is_left_half(angle) == is_left_half(endpoint_angle)
        and angular_distance(angle, endpoint_angle) <= tolerance
    )


def calculate_value(
    angle: float,
    min_value: float,
    max_value: float,
    min_angle: float,
    max_angle: float,
    direction: str = "clockwise",
    endpoint_tolerance: float = 5.0,
) -> float:
    angle_span = angle_delta(min_angle, max_angle, direction)
    if angle_span == 0:
        raise ValueError("min-angle and max-angle must not be the same direction.")
    if endpoint_tolerance < 0:
        raise ValueError("endpoint-tolerance must be 0 or greater.")

    if should_clamp_endpoint(angle, min_angle, endpoint_tolerance):
        return min_value
    if should_clamp_endpoint(angle, max_angle, endpoint_tolerance):
        return max_value

    value_span = max_value - min_value
    angle_offset = angle_delta(min_angle, angle, direction)
    ratio = angle_offset / angle_span
    return min_value + value_span * ratio


def validate_number(name: str, value: float) -> str | None:
    if not math.isfinite(value):
        return f"{name} must be a finite number."
    return None


def main() -> int:
    args = parse_args()
    for name in (
        "angle",
        "min_value",
        "max_value",
        "min_angle",
        "max_angle",
        "endpoint_tolerance",
    ):
        error = validate_number(f"--{name.replace('_', '-')}", getattr(args, name))
        if error:
            print(f"Error: {error}", file=sys.stderr)
            return 1

    try:
        value = calculate_value(
            angle=args.angle,
            min_value=args.min_value,
            max_value=args.max_value,
            min_angle=args.min_angle,
            max_angle=args.max_angle,
            direction=args.direction,
            endpoint_tolerance=args.endpoint_tolerance,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"value: {value:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
