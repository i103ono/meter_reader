from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)


class OcrError(RuntimeError):
    pass


_READERS = {}


def resolve_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "easyocr_models"


def resolve_user_network_dir() -> Path:
    return resolve_model_dir() / "user_network"


def load_ocr_image(path: str | Path, max_size: int):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
        return np.array(img)


def read_text(
    path: str | Path,
    digits_only: bool = False,
    gpu: bool = False,
    canvas_size: int = 1280,
) -> tuple[str, float | None]:
    try:
        import easyocr
    except ModuleNotFoundError as exc:
        raise OcrError(
            "EasyOCR is not installed. Update the environment from environment.yml."
        ) from exc

    allowlist = "0123456789." if digits_only else None
    try:
        model_dir = resolve_model_dir()
        user_network_dir = resolve_user_network_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        user_network_dir.mkdir(parents=True, exist_ok=True)
        reader_key = (gpu, str(model_dir), str(user_network_dir))
        if reader_key not in _READERS:
            _READERS[reader_key] = easyocr.Reader(
                ["en"],
                gpu=gpu,
                model_storage_directory=str(model_dir),
                user_network_directory=str(user_network_dir),
                verbose=False,
            )
        reader = _READERS[reader_key]
        image = load_ocr_image(path, canvas_size)
        results = reader.readtext(
            image,
            detail=1,
            allowlist=allowlist,
            canvas_size=canvas_size,
        )
    except Exception as exc:
        raise OcrError(f"EasyOCR failed: {exc}") from exc

    texts: list[str] = []
    confidences: list[float] = []
    for _bbox, text, confidence in results:
        text = str(text).strip()
        if text:
            texts.append(text)
        try:
            confidences.append(float(confidence))
        except (TypeError, ValueError):
            continue

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )
    return " ".join(texts), average_confidence
