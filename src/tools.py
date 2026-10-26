"""Versioned, deterministic correction recipes. Never execute saved code or commands."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import platform
from types import MappingProxyType

import cv2
import numpy as np
import PIL
from PIL import Image

from schemas import ProcessingStep

PROCESSING_VERSION = "fixed-recipes-v1"
MAX_OUTPUT_PIXELS = 64_000_000
MAX_CORRECTION_CANDIDATES = 2
RECIPES = MappingProxyType({
    "resize_2x": ("resize",),
    "gamma_mild": ("gamma",),
    "clahe_mild": ("clahe",),
    "gamma_clahe": ("gamma", "clahe"),
    "resize_gamma": ("gamma", "resize"),
    "resize_clahe": ("clahe", "resize"),
    "resize_gamma_clahe": ("gamma", "clahe", "resize"),
})


class CorrectionError(ValueError):
    pass


def validate_recipes(recipe_ids) -> tuple[str, ...]:
    if not isinstance(recipe_ids, (tuple, list)):
        raise CorrectionError("recipes must be a list of registered recipe IDs")
    if len(recipe_ids) > MAX_CORRECTION_CANDIDATES:
        raise CorrectionError("At most two correction candidates are allowed per run")
    if any(not isinstance(recipe, str) or recipe not in RECIPES for recipe in recipe_ids):
        raise CorrectionError("Unknown correction recipe")
    if len(set(recipe_ids)) != len(recipe_ids):
        raise CorrectionError("Duplicate correction recipes are not allowed")
    return tuple(recipe_ids)


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "machine": platform.machine(),
        "opencv": cv2.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
    }


def pixel_sha256(image: Image.Image) -> str:
    header = json.dumps([image.mode, *image.size], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + image.tobytes()).hexdigest()


def recipe_steps(recipe_id: str, size: tuple[int, int]) -> list[ProcessingStep]:
    if recipe_id not in RECIPES:
        raise CorrectionError("Unknown correction recipe")
    width, height = size
    if width <= 0 or height <= 0:
        raise CorrectionError("ROI must be nonempty")
    result = []
    for operation in RECIPES[recipe_id]:
        if operation == "resize":
            parameters = {
                "factor": 2, "interpolation": "Pillow.BICUBIC",
                "input_size": [width, height], "output_size": [width * 2, height * 2],
                "max_output_pixels": MAX_OUTPUT_PIXELS,
            }
            width, height = width * 2, height * 2
        elif operation == "gamma":
            parameters = {
                "exponent": 0.75, "channel": "OpenCV.RGB2YCrCb.Y",
                "formula": "round(255 * (Y / 255) ** 0.75)", "rounding": "half_up_uint8",
            }
        else:
            parameters = {
                "clip_limit": 2.0, "requested_tile_grid": [8, 8],
                "effective_tile_grid": [min(8, max(1, width // 32)), min(8, max(1, height // 32))],
                "minimum_tile_side": 32, "channel": "OpenCV.RGB2YCrCb.Y",
            }
        result.append(ProcessingStep(operation=operation, parameters=parameters))
    return result


def gamma_luminance(channel: np.ndarray) -> np.ndarray:
    """The stated exponent, not its reciprocal. Testable independently of color conversion."""
    lut = np.floor(255 * (np.arange(256, dtype=np.float64) / 255) ** 0.75 + 0.5).astype(np.uint8)
    return cv2.LUT(channel, lut)


@dataclass
class Correction:
    image: Image.Image
    steps: list[ProcessingStep]


def apply_recipe(original_roi: Image.Image, recipe_id: str) -> Correction:
    steps = recipe_steps(recipe_id, original_roi.size)
    if original_roi.mode not in ("RGB", "RGBA"):
        raise CorrectionError("Correction input must be the normalized RGB/RGBA ROI")
    # Preflight allocations before any processing. No silent shrinking.
    for step in steps:
        if step.operation == "resize":
            width, height = step.parameters["output_size"]
            if width * height > MAX_OUTPUT_PIXELS:
                raise CorrectionError("Correction output exceeds the pixel limit")
    image = original_roi.copy()
    try:
        for step in steps:
            if step.operation == "resize":
                updated = image.resize(tuple(step.parameters["output_size"]), Image.Resampling.BICUBIC)
            else:
                rgb = np.array(image.convert("RGB"))
                ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
                if step.operation == "gamma":
                    ycc[:, :, 0] = gamma_luminance(ycc[:, :, 0])
                else:
                    clahe = cv2.createCLAHE(
                        clipLimit=step.parameters["clip_limit"],
                        tileGridSize=tuple(step.parameters["effective_tile_grid"]),
                    )
                    ycc[:, :, 0] = clahe.apply(ycc[:, :, 0])
                updated = Image.fromarray(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB))
                if image.mode == "RGBA":
                    with image.getchannel("A") as alpha:
                        updated.putalpha(alpha)
            image.close()
            image = updated
        image.info.clear()
        return Correction(image=image, steps=steps)
    except cv2.error as exc:
        image.close()
        raise CorrectionError("OpenCV could not apply the registered correction") from exc
    except Exception:
        image.close()
        raise
