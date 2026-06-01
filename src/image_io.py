#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


class ImageLoadError(RuntimeError):
    pass


def load_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    try:
        with Image.open(image_path) as img:
            img.load()
            return img.copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError(
            f"Failed to load image: {image_path} ({exc})"
        ) from exc
