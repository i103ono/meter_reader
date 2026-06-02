#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


NUMBER_CHARS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        ",": ".",
    }
)
NUMBER_CANDIDATE_RE = re.compile(r"[+-]?[0-9OoIl|SsB,.]+")
UNIT_RE = re.compile(r"^\s*([A-Za-z%/]+(?:\^?[0-9]+)?)")
UNIT_ALIASES = {
    "kwh": "kWh",
    "wh": "Wh",
    "kw": "kW",
    "w": "W",
    "v": "V",
    "a": "A",
    "ma": "mA",
    "m3": "m3",
    "l": "L",
    "kg": "kg",
    "bar": "bar",
}


@dataclass(frozen=True)
class Reading:
    value: float
    unit: str | None
    corrected_value_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a numeric value and unit from OCR text."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="OCR text. If omitted, text is read from stdin.",
    )
    return parser.parse_args()


def normalize_number(candidate: str) -> str:
    normalized = candidate.translate(NUMBER_CHARS)
    if normalized.count(".") > 1:
        first_dot = normalized.find(".")
        normalized = (
            normalized[: first_dot + 1]
            + normalized[first_dot + 1 :].replace(".", "")
        )
    return normalized


def normalize_unit(unit: str) -> str:
    compact = unit.strip()
    return UNIT_ALIASES.get(compact.lower(), compact)


def parse_reading(text: str) -> Reading:
    for match in NUMBER_CANDIDATE_RE.finditer(text):
        if not re.search(r"\d", match.group(0)):
            continue
        corrected = normalize_number(match.group(0))
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", corrected):
            continue

        remainder = text[match.end() :]
        unit_match = UNIT_RE.match(remainder)
        unit = normalize_unit(unit_match.group(1)) if unit_match else None
        return Reading(
            value=float(corrected),
            unit=unit,
            corrected_value_text=corrected,
        )

    raise ValueError("No numeric value found in OCR text.")


def main() -> int:
    args = parse_args()
    text = " ".join(args.text) if args.text else sys.stdin.read()
    if not text.strip():
        print("Error: OCR text is empty.", file=sys.stderr)
        return 1

    try:
        reading = parse_reading(text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"value: {reading.value:g}")
    print(f"unit: {reading.unit or ''}")
    print(f"corrected_value_text: {reading.corrected_value_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
