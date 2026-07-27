#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


LABEL_FIELDS = {"filename", "type", "value", "unit", "min_value", "max_value"}
RESULT_FIELDS = {"filename", "type", "status", "value", "unit"}


class EvaluationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare meter reading results with evaluation labels."
    )
    parser.add_argument(
        "--labels",
        default="data/evaluation_labels.csv",
        help="Label CSV path. Default: data/evaluation_labels.csv",
    )
    parser.add_argument(
        "--results",
        default="outputs/results.csv",
        help="Results CSV path. Default: outputs/results.csv",
    )
    return parser.parse_args()


def read_csv(path: Path, required_fields: set[str], name: str) -> list[dict[str, str]]:
    if not path.exists():
        raise EvaluationError(f"{name} file not found: {path}")
    if not path.is_file():
        raise EvaluationError(f"{name} path is not a file: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = required_fields - fields
            if missing:
                raise EvaluationError(
                    f"{name} CSV is missing columns: {', '.join(sorted(missing))}"
                )
            return list(reader)
    except OSError as exc:
        raise EvaluationError(f"Failed to read {name} CSV: {path} ({exc})") from exc


def index_by_filename(
    rows: list[dict[str, str]],
    name: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        filename = (row.get("filename") or "").strip()
        if not filename:
            raise EvaluationError(f"{name} CSV has an empty filename at line {line_number}.")
        if filename in indexed:
            raise EvaluationError(f"{name} CSV has duplicate filename: {filename}")
        indexed[filename] = row
    return indexed


def parse_number(value: str, field: str, filename: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise EvaluationError(
            f"Invalid {field} for {filename}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise EvaluationError(f"Invalid {field} for {filename}: {value!r}")
    return number


def normalize_unit(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("³", "^3")
        .replace("㎥", "m^3")
    )


def validate_labels(labels: list[dict[str, str]]) -> None:
    index_by_filename(labels, "Labels")
    for line_number, label in enumerate(labels, start=2):
        filename = label["filename"].strip()
        meter_type = label["type"].strip().lower()
        if meter_type not in ("digital", "analog"):
            raise EvaluationError(
                f"Invalid type for {filename or f'line {line_number}'}: "
                f"{label['type']!r}"
            )
        for field in ("value", "unit"):
            if not label[field].strip():
                raise EvaluationError(f"Missing {field} for {filename}.")
        parse_number(label["value"].strip(), "value", filename)
        if meter_type == "analog":
            for field in ("min_value", "max_value"):
                if not label[field].strip():
                    raise EvaluationError(f"Missing {field} for {filename}.")
                parse_number(label[field].strip(), field, filename)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def format_metric(value: float | None, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.6g}{suffix}"


def evaluate(
    labels: list[dict[str, str]],
    results: dict[str, dict[str, str]],
) -> dict[str, Any]:
    absolute_errors: list[float] = []
    digital_errors: list[float] = []
    analog_errors: list[float] = []
    normalized_errors: list[float] = []
    within_five_percent = 0
    unit_matches = 0
    unit_total = 0
    type_matches = 0
    success_count = 0
    missing_count = 0
    failed_files: list[str] = []

    for label in labels:
        filename = label["filename"].strip()
        expected_type = label["type"].strip().lower()
        expected_value = parse_number(label["value"].strip(), "value", filename)
        result = results.get(filename)
        if result is None:
            missing_count += 1
            failed_files.append(filename)
            continue

        if (result.get("type") or "").strip().lower() == expected_type:
            type_matches += 1

        result_value = (result.get("value") or "").strip()
        if (result.get("status") or "").strip().lower() != "ok" or not result_value:
            failed_files.append(filename)
            continue
        try:
            predicted_value = float(result_value)
        except ValueError:
            failed_files.append(filename)
            continue
        if not math.isfinite(predicted_value):
            failed_files.append(filename)
            continue

        success_count += 1
        absolute_error = abs(predicted_value - expected_value)
        absolute_errors.append(absolute_error)
        if expected_type == "digital":
            digital_errors.append(absolute_error)
        else:
            analog_errors.append(absolute_error)

        expected_unit = normalize_unit(label["unit"])
        if expected_unit:
            unit_total += 1
            if normalize_unit(result.get("unit") or "") == expected_unit:
                unit_matches += 1

        if expected_type == "analog":
            min_value = parse_number(label["min_value"].strip(), "min_value", filename)
            max_value = parse_number(label["max_value"].strip(), "max_value", filename)
            value_range = max_value - min_value
            if value_range <= 0:
                raise EvaluationError(
                    f"max_value must be greater than min_value for {filename}."
                )
            normalized_error = absolute_error / value_range
            normalized_errors.append(normalized_error)
            if normalized_error <= 0.05:
                within_five_percent += 1

    total = len(labels)
    return {
        "total": total,
        "success_count": success_count,
        "success_rate": success_count / total if total else None,
        "missing_count": missing_count,
        "mae": mean(absolute_errors),
        "digital_mae": mean(digital_errors),
        "analog_mae": mean(analog_errors),
        "max_absolute_error": max(absolute_errors) if absolute_errors else None,
        "unit_accuracy": unit_matches / unit_total if unit_total else None,
        "unit_matches": unit_matches,
        "unit_total": unit_total,
        "type_accuracy": type_matches / total if total else None,
        "normalized_mae": mean(normalized_errors),
        "within_five_percent_rate": (
            within_five_percent / len(normalized_errors)
            if normalized_errors
            else None
        ),
        "analog_evaluated": len(normalized_errors),
        "failed_files": failed_files,
    }


def print_report(metrics: dict[str, Any]) -> None:
    print(f"evaluated: {metrics['total']}")
    print(f"success: {metrics['success_count']}")
    print(f"success_rate: {format_metric(metrics['success_rate'] * 100 if metrics['success_rate'] is not None else None, '%')}")
    print(f"missing_results: {metrics['missing_count']}")
    print(f"mae: {format_metric(metrics['mae'])}")
    print(f"digital_mae: {format_metric(metrics['digital_mae'])}")
    print(f"analog_mae: {format_metric(metrics['analog_mae'])}")
    print(f"max_absolute_error: {format_metric(metrics['max_absolute_error'])}")
    print(
        "unit_accuracy: "
        f"{format_metric(metrics['unit_accuracy'] * 100 if metrics['unit_accuracy'] is not None else None, '%')} "
        f"({metrics['unit_matches']}/{metrics['unit_total']})"
    )
    print(
        "type_accuracy: "
        f"{format_metric(metrics['type_accuracy'] * 100 if metrics['type_accuracy'] is not None else None, '%')}"
    )
    print(f"analog_evaluated: {metrics['analog_evaluated']}")
    print(f"analog_normalized_mae: {format_metric(metrics['normalized_mae'] * 100 if metrics['normalized_mae'] is not None else None, '%')}")
    print(
        "analog_within_5_percent: "
        f"{format_metric(metrics['within_five_percent_rate'] * 100 if metrics['within_five_percent_rate'] is not None else None, '%')}"
    )
    if metrics["failed_files"]:
        print(f"failed_files: {', '.join(metrics['failed_files'])}")


def main() -> int:
    args = parse_args()
    try:
        labels = read_csv(Path(args.labels).expanduser(), LABEL_FIELDS, "Labels")
        if not labels:
            raise EvaluationError("Labels CSV contains no evaluation rows.")
        validate_labels(labels)
        result_rows = read_csv(
            Path(args.results).expanduser(),
            RESULT_FIELDS,
            "Results",
        )
        results = index_by_filename(result_rows, "Results")
        metrics = evaluate(labels, results)
    except EvaluationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_report(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
