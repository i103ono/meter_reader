from __future__ import annotations

import os
import warnings
from pathlib import Path


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)


class OcrError(RuntimeError):
    pass


def resolve_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "outputs" / "easyocr_models"


def resolve_user_network_dir() -> Path:
    return resolve_model_dir() / "user_network"


def read_text(
    path: str | Path,
    digits_only: bool = False,
    gpu: bool = False,
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
        reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            model_storage_directory=str(model_dir),
            user_network_directory=str(user_network_dir),
            verbose=False,
        )
        results = reader.readtext(str(path), detail=1, allowlist=allowlist)
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
