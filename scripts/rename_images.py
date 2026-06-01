#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FILENAME_PREFIX = "meter_"
FILENAME_WIDTH = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename images in data/images to sequential meter_XXXX names."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames. Without this flag, the script runs in dry-run mode.",
    )
    return parser.parse_args()


def resolve_images_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "images"


def collect_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]


def plan_renames(images: list[Path], images_dir: Path) -> list[tuple[Path, Path]]:
    images_set = set(images)
    protected_names = {
        path.name
        for path in images_dir.iterdir()
        if path.is_file() and path not in images_set
    }

    plans: list[tuple[Path, Path]] = []
    assigned: set[str] = set()
    counter = 1

    for source in sorted(images, key=lambda path: path.name.casefold()):
        while True:
            target_name = (
                f"{FILENAME_PREFIX}{counter:0{FILENAME_WIDTH}d}"
                f"{source.suffix.lower()}"
            )
            counter += 1
            if target_name in protected_names or target_name in assigned:
                continue
            assigned.add(target_name)
            plans.append((source, images_dir / target_name))
            break

    return plans


def print_plan(plans: list[tuple[Path, Path]], dry_run: bool) -> list[tuple[Path, Path]]:
    changes = [(source, target) for source, target in plans if source.name != target.name]
    if not changes:
        print("No changes.")
        return []

    prefix = "DRY-RUN" if dry_run else "RENAME"
    for source, target in changes:
        print(f"{prefix}: {source.name} -> {target.name}")
    return changes


def apply_renames(changes: list[tuple[Path, Path]]) -> None:
    temp_moves: list[tuple[Path, Path, Path]] = []
    for source, target in changes:
        temp_name = f".rename_tmp_{uuid.uuid4().hex}{source.suffix.lower()}"
        temp_path = source.with_name(temp_name)
        while temp_path.exists():
            temp_name = f".rename_tmp_{uuid.uuid4().hex}{source.suffix.lower()}"
            temp_path = source.with_name(temp_name)
        source.rename(temp_path)
        temp_moves.append((source, temp_path, target))

    for original, temp_path, target in temp_moves:
        if target.exists():
            raise FileExistsError(
                f"Target already exists, refusing to overwrite: {target.name}"
            )
        temp_path.rename(target)


def main() -> int:
    args = parse_args()
    images_dir = resolve_images_dir()
    if not images_dir.exists():
        print(f"Directory not found: {images_dir}", file=sys.stderr)
        return 1

    images = collect_images(images_dir)
    if not images:
        print("No matching image files found.")
        return 0

    plans = plan_renames(images, images_dir)
    changes = print_plan(plans, dry_run=not args.apply)

    if args.apply and changes:
        apply_renames(changes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
