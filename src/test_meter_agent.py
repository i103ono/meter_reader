#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from meter_agent import (
    AgentError,
    build_analog_local_command,
    render_api_reading,
    run_reader,
)


DEFAULT_INPUTS = (
    Path("data/images/meter_0005.jpg"),
    Path("data/images/meter_0001.jpg"),
    Path("data/images/meter_0030.jpg"),
)

MOCK_CONFIGS: dict[str, dict[str, Any]] = {
    "meter_0005": {
        "meter_type": "digital",
        "roi": None,
        "raw_text": "040731",
        "value": 40731,
        "unit": "m3",
        "min_value": None,
        "max_value": None,
        "min_angle": None,
        "max_angle": None,
        "direction": None,
        "confidence": 1.0,
        "notes": "Mock digital reading.",
    },
    "meter_0001": {
        "meter_type": "analog",
        "roi": None,
        "raw_text": "",
        "value": None,
        "unit": "MPa",
        "min_value": 0,
        "max_value": 70,
        "min_angle": None,
        "max_angle": None,
        "direction": None,
        "confidence": 1.0,
        "notes": "Mock analog settings; value is read locally.",
    },
    "meter_0030": {
        "meter_type": "analog",
        "roi": None,
        "raw_text": "",
        "value": None,
        "unit": "MPa",
        "min_value": 0,
        "max_value": 70,
        "min_angle": None,
        "max_angle": None,
        "direction": None,
        "confidence": 1.0,
        "notes": "Mock analog settings; value is read locally.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run representative checks for meter_agent.py."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input image paths. Default: representative local images if present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of images to check.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use built-in mock configs and do not call the OpenAI API.",
    )
    parser.add_argument(
        "--digits-only",
        action="store_true",
        help="Pass --digits-only to meter_agent.py in live mode.",
    )
    return parser.parse_args()


def select_inputs(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.inputs] if args.inputs else list(DEFAULT_INPUTS)
    existing = [path for path in paths if path.exists()]
    if args.limit is not None:
        if args.limit <= 0:
            raise AgentError("--limit must be greater than 0.")
        existing = existing[: args.limit]
    if not existing:
        raise AgentError("No input images found.")
    return existing


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        values[key.strip()] = value.strip()
    return values


def parse_agent_output(stdout: str) -> tuple[dict[str, Any], str, dict[str, str]]:
    lines = stdout.splitlines()
    config_lines: list[str] = []
    reading_lines: list[str] = []
    reading_source = ""
    section = ""

    for line in lines:
        if line == "config:":
            section = "config"
            continue
        if line.startswith("reading_source: "):
            section = ""
            reading_source = line.split(": ", 1)[1]
            continue
        if line == "reading:":
            section = "reading"
            continue
        if line.startswith("reader_input: "):
            section = ""
            continue
        if section == "config":
            config_lines.append(line)
        elif section == "reading":
            reading_lines.append(line)

    config: dict[str, Any] = {}
    if config_lines:
        config = json.loads("\n".join(config_lines))
    return config, reading_source, parse_key_values("\n".join(reading_lines))


def live_row(path: Path, digits_only: bool) -> dict[str, str]:
    command = [sys.executable, str(Path(__file__).with_name("meter_agent.py")), str(path)]
    if digits_only:
        command.append("--digits-only")
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        return {
            "input": str(path),
            "status": "failure",
            "meter_type": "",
            "value": "",
            "unit": "",
            "reading_source": "",
            "config": "",
            "error": f"Failed to run meter_agent.py: {exc}",
        }

    config: dict[str, Any] = {}
    reading_source = ""
    reading: dict[str, str] = {}
    error = result.stderr.strip()
    parse_failed = False
    if result.stdout.strip():
        try:
            config, reading_source, reading = parse_agent_output(result.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            error = f"Failed to parse meter_agent output: {exc}"
            parse_failed = True

    return {
        "input": str(path),
        "status": "success" if result.returncode == 0 and not parse_failed else "failure",
        "meter_type": str(config.get("meter_type", reading.get("type", ""))),
        "value": reading.get("value", ""),
        "unit": reading.get("unit", str(config.get("unit", ""))),
        "reading_source": reading_source,
        "config": compact_json(config) if config else "",
        "error": error,
    }


def mock_row(path: Path) -> dict[str, str]:
    config = MOCK_CONFIGS.get(path.stem)
    if config is None:
        config = {
            "meter_type": "digital",
            "roi": None,
            "raw_text": "",
            "value": None,
            "unit": "",
            "min_value": None,
            "max_value": None,
            "min_angle": None,
            "max_angle": None,
            "direction": None,
            "confidence": 0.0,
            "notes": "No dedicated mock config for this image.",
        }

    try:
        if config["meter_type"] == "digital":
            reading_text = render_api_reading(config)
            reading_source = "mock_openai"
            status = "success"
            error = ""
        else:
            command = build_analog_local_command(
                Path(__file__).with_name("read_meter.py"),
                path,
                config,
            )
            result = run_reader(command)
            reading_text = result.stdout
            reading_source = "mock_config_local_analog_reader"
            status = "success" if result.returncode == 0 else "failure"
            error = result.stderr.strip()
    except AgentError as exc:
        reading_text = ""
        reading_source = "mock"
        status = "failure"
        error = str(exc)

    reading = parse_key_values(reading_text)
    return {
        "input": str(path),
        "status": status,
        "meter_type": str(config.get("meter_type", reading.get("type", ""))),
        "value": reading.get("value", ""),
        "unit": reading.get("unit", str(config.get("unit", ""))),
        "reading_source": reading_source,
        "config": compact_json(config),
        "error": error,
    }


def main() -> int:
    args = parse_args()
    try:
        paths = select_inputs(args)
        rows = [mock_row(path) if args.mock else live_row(path, args.digits_only) for path in paths]
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    fieldnames = (
        "input",
        "status",
        "meter_type",
        "value",
        "unit",
        "reading_source",
        "config",
        "error",
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return 0 if all(row["status"] == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
