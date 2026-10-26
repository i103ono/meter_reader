#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError


class ImageLoadError(RuntimeError):
    pass


def _decode_image(source: Path | BinaryIO, label: str) -> Image.Image:
    try:
        with Image.open(source) as img:
            img.load()
            result = img.copy()
            result.format = img.format
            return result
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageLoadError(
            f"Failed to load image: {label} ({exc})"
        ) from exc


def load_image(path: str | Path) -> Image.Image:
    """Existing CLI API: decode without changing orientation or restricting formats."""
    image_path = Path(path)
    return _decode_image(image_path, str(image_path))


def _roi_values(roi) -> tuple[int, int, int, int]:
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("ROI must contain exactly four integers")
    if any(type(value) is not int for value in roi):
        raise ValueError("ROI coordinates must be integers (not bool, float or string)")
    return tuple(roi)


def validate_roi_xyxy(roi, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Coordinates on the EXIF-normalized image; right/bottom are exclusive."""
    x1, y1, x2, y2 = _roi_values(roi)
    width, height = size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"ROI must satisfy 0 <= x1 < x2 <= {width}, 0 <= y1 < y2 <= {height}")
    return x1, y1, x2, y2


def xywh_to_xyxy(roi) -> tuple[int, int, int, int]:
    """Explicit adapter for the legacy CLI/config contract, never infer its format."""
    x, y, width, height = _roi_values(roi)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("Legacy ROI requires x/y >= 0 and width/height > 0")
    return x, y, x + width, y + height


def xyxy_to_xywh(roi) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _roi_values(roi)
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("ROI must be a nonempty, nonnegative xyxy region")
    return x1, y1, x2 - x1, y2 - y1


@dataclass
class PreparedImage:
    original_path: Path
    content: bytes
    sha256: str
    format: str
    original_size: tuple[int, int]
    image: Image.Image
    exif_orientation: int | None
    exif_datetime_original: str | None
    exif_offset_original: str | None


def prepare_image(path: str | Path) -> PreparedImage:
    """Hash and decode the same bytes once; normalize before validating any ROI."""
    original_path = Path(path).expanduser().resolve()
    suffix = original_path.suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png"):
        raise ImageLoadError("Only JPEG (.jpg/.jpeg) and PNG (.png) inputs are supported")
    if not original_path.is_file():
        raise ImageLoadError(f"Input is not a regular file: {original_path}")
    try:
        content = original_path.read_bytes()
    except OSError as exc:
        raise ImageLoadError(f"Cannot read input: {original_path} ({exc})") from exc
    img = _decode_image(BytesIO(content), str(original_path))
    try:
        expected_format = "PNG" if suffix == ".png" else "JPEG"
        if img.format != expected_format:
            raise ImageLoadError("Image content does not match its JPEG/PNG extension")
        image_format, original_size = img.format, img.size
        exif = img.getexif()
        try:
            exif_ifd = exif.get_ifd(34665)
        except (KeyError, TypeError, ValueError):
            exif_ifd = {}
        original_time = exif_ifd.get(36867, exif.get(36867))
        original_offset = exif_ifd.get(36881, exif.get(36881))
        orientation = exif.get(274)
        normalized = ImageOps.exif_transpose(img)
        try:
            # Preserve PNG transparency; orientation is already applied.
            has_alpha = "A" in normalized.getbands() or "transparency" in normalized.info
            rgb = normalized.convert("RGBA" if has_alpha else "RGB")
            rgb.info.clear()
        finally:
            normalized.close()
        return PreparedImage(
            original_path, content, hashlib.sha256(content).hexdigest(),
            image_format, original_size, rgb,
            orientation if type(orientation) is int else None,
            original_time if isinstance(original_time, str) else None,
            original_offset if isinstance(original_offset, str) else None,
        )
    finally:
        img.close()


def save_png_exclusive(image: Image.Image, path: Path) -> None:
    """Fail on existing targets, including symlinks; never overwrite a source."""
    with path.open("xb") as output:
        image.save(output, format="PNG")
