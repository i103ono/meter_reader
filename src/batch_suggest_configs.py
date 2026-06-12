#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from list_images import collect_images
from meter_agent import (
    DEFAULT_MODEL,
    AgentError,
    call_openai,
    load_env_file,
    normalize_config,
    validate_roi,
)
from suggest_meter_config import save_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest meter configs for images in filename order."
    )
    parser.add_argument(
        "--images-dir",
        default="data/images",
        help="Input image directory. Default: data/images",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Output config directory. Default: configs",
    )
    parser.add_argument(
        "--error-log",
        default="outputs/config_suggestion_errors.csv",
        help="Failure CSV path.",
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
    return parser.parse_args()


def select_images(images_dir: Path, limit: int | None) -> list[Path]:
    if not images_dir.exists():
        raise AgentError(f"Images directory not found: {images_dir}")
    if not images_dir.is_dir():
        raise AgentError(f"Images path is not a directory: {images_dir}")
    if limit is not None and limit <= 0:
        raise AgentError("--limit must be greater than 0.")

    images = sorted(collect_images(images_dir), key=lambda path: path.name.casefold())
    if limit is not None:
        images = images[:limit]
    if not images:
        raise AgentError(
            f"No .jpg, .jpeg, or .png files found in images directory: {images_dir}"
        )
    return images


def write_error_log(error_log: Path, failures: list[dict[str, str]]) -> None:
    try:
        error_log.parent.mkdir(parents=True, exist_ok=True)
        with error_log.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("filename", "error"))
            writer.writeheader()
            writer.writerows(failures)
    except OSError as exc:
        raise AgentError(f"Failed to write error log: {error_log} ({exc})") from exc


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser()
    config_dir = Path(args.config_dir).expanduser()
    error_log = Path(args.error_log).expanduser()
    env_path = Path(args.env_file).expanduser()

    try:
        images = select_images(images_dir, args.limit)
        load_env_file(env_path)
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    failures: list[dict[str, str]] = []
    success_count = 0
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path.name}")
        try:
            config = normalize_config(call_openai(image_path, args.model, args.detail))
            validate_roi(config.get("roi"), image_path)
            output_path = config_dir / f"{image_path.stem}.json"
            save_config(config, output_path)
        except AgentError as exc:
            failures.append({"filename": image_path.name, "error": str(exc)})
            print(f"Warning: {image_path.name}: {exc}", file=sys.stderr)
            continue

        success_count += 1
        print(f"Saved: {output_path}")

    try:
        write_error_log(error_log, failures)
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Completed: {success_count} succeeded, {len(failures)} failed")
    print(f"Error log: {error_log}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
