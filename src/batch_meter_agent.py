#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from batch_read import (
    FIELDNAMES,
    config_args,
    empty_row,
    read_analog_row,
    read_config_digital_row,
)
from batch_suggest_configs import select_images
from meter_agent import (
    DEFAULT_MODEL,
    AgentError,
    call_openai,
    load_env_file,
    normalize_config,
)
from suggest_meter_config import save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest configs and read multiple meter images in one run."
    )
    parser.add_argument(
        "--images-dir",
        default="data/images",
        help="Input image directory. Default: data/images",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Config output directory. Default: configs",
    )
    parser.add_argument(
        "--output",
        default="outputs/results.csv",
        help="Results CSV path. Default: outputs/results.csv",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of images to process.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the env file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--detail",
        choices=("low", "high", "auto"),
        default="high",
        help="Vision input detail level.",
    )
    parser.add_argument("--endpoint-tolerance", type=float, default=5.0)
    parser.add_argument("--dial-min-radius", type=int, default=20)
    parser.add_argument("--dial-max-radius", type=int, default=0)
    parser.set_defaults(
        type=None,
        unit=None,
        min_value=None,
        max_value=None,
        min_angle=None,
        max_angle=None,
        direction=None,
    )
    return parser.parse_args()


def save_results(rows: list[dict[str, str]], output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        raise AgentError(f"Failed to save results CSV: {output_path} ({exc})") from exc


def process_image(
    image_path: Path,
    config_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    config = normalize_config(call_openai(image_path, args.model, args.detail))
    config_path = config_dir / f"{image_path.stem}.json"
    save_config(config, config_path)

    process_args, saved_config = config_args(args, config_path)
    if process_args.type == "digital":
        return read_config_digital_row(image_path, saved_config)
    return read_analog_row(image_path, process_args)


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser()
    config_dir = Path(args.config_dir).expanduser()
    output_path = Path(args.output).expanduser()
    env_path = Path(args.env_file).expanduser()

    try:
        images = select_images(images_dir, args.limit)
        load_env_file(env_path)
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rows: list[dict[str, str]] = []
    for index, image_path in enumerate(images, start=1):
        print(f"Processing {index}/{len(images)}: {image_path.name}", flush=True)
        try:
            row = process_image(image_path, config_dir, args)
        except (AgentError, ValueError) as exc:
            row = empty_row(image_path.name, "")
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"Warning: {image_path.name}: {exc}", file=sys.stderr)
        row["filename"] = image_path.name
        rows.append(row)

    try:
        save_results(rows, output_path)
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    ok_count = sum(row["status"] == "ok" for row in rows)
    print(f"Saved: {output_path}")
    print(f"processed: {len(rows)}")
    print(f"ok: {ok_count}")
    print(f"error: {len(rows) - ok_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
