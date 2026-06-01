#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from image_io import ImageLoadError, load_image

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_images_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "images"


def collect_images(images_dir: Path) -> list[Path]:
    return [
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    ]


def main() -> int:
    images_dir = resolve_images_dir()
    if not images_dir.exists():
        print(
            "Directory not found: "
            f"{images_dir} (create data/images/ and place .jpg/.jpeg/.png files there)",
            file=sys.stderr,
        )
        return 1
    if not images_dir.is_dir():
        print(f"Not a directory: {images_dir}", file=sys.stderr)
        return 1

    images = sorted(collect_images(images_dir), key=lambda path: path.name.casefold())
    if not images:
        print("No matching image files found.")
        return 0

    for path in images:
        try:
            img = load_image(path)
            width, height = img.size
            channels = len(img.getbands())
            img.close()
        except ImageLoadError as exc:
            print(f"Warning: {exc}", file=sys.stderr)
            continue

        print(f"{path.name}\t{width}x{height}\t{channels}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
