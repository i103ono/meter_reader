#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import ImageOps

from image_io import ImageLoadError, load_image


API_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-4.1-mini"


class AgentError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate meter settings with OpenAI Vision and read one meter image."
    )
    parser.add_argument("input", help="Input image path (.jpg/.jpeg/.png)")
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
    parser.add_argument(
        "--digits-only",
        action="store_true",
        help="Kept for compatibility; digital values are read by OpenAI.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated ROI crops.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise AgentError(f"Env path is not a file: {path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AgentError(f"Failed to read env file: {path} ({exc})") from exc

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("'\"")
        os.environ[key] = value


def validate_input(path: Path) -> None:
    if not path.exists():
        raise AgentError(f"Input file not found: {path}")
    if not path.is_file():
        raise AgentError(f"Input path is not a file: {path}")


def encode_image(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise AgentError(f"Failed to read image: {path} ({exc})") from exc


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"
    elif suffix == ".png":
        mime_type = "image/png"
    else:
        raise AgentError("Input image must be .jpg, .jpeg, or .png.")
    return f"data:{mime_type};base64,{encode_image(path)}"


def build_prompt() -> str:
    return (
        "You are helping configure a meter-reading CLI. Inspect the image and return "
        "only a JSON object. Do not wrap it in markdown. The JSON schema is: "
        '{"meter_type":"digital|analog","roi":[x,y,w,h] or null,'
        '"raw_text":string,"value":number or null,'
        '"unit":string,"min_value":number or null,"max_value":number or null,'
        '"min_angle":number or null,"max_angle":number or null,'
        '"direction":"clockwise|counterclockwise" or null,'
        '"confidence":number between 0 and 1,"notes":string}. '
        "Use roi for the main display or dial region in original image pixels. "
        "For digital meters, read the main numeric display directly from the image, "
        "set raw_text to the visible reading text, and set value to the numeric reading. "
        "For analog meters, estimate the needle reading directly from the image by "
        "looking at the thin measuring pointer, tick marks, scale labels, and unit. "
        "If the needle has a wide triangular hub or counterweight, do not use that "
        "wide part as the reading direction; use the thin pointer end that indicates "
        "the scale value. Also estimate min_value and max_value from visible scale "
        "labels. For analog roi, include the whole circular dial and outer rim, not "
        "only the center hub, label text, or needle base. "
        "If a value cannot be estimated, use null."
    )


def call_openai(image_path: Path, model: str, detail: str) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise AgentError(
            "OPENAI_API_KEY is not set. Create .env in the repository root and set "
            "OPENAI_API_KEY there."
        )

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_prompt()},
                    {
                        "type": "input_image",
                        "image_url": image_data_url(image_path),
                        "detail": detail,
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.reason
        try:
            error_data = json.loads(exc.read().decode("utf-8"))
            message = error_data.get("error", {}).get("message", message)
        except (OSError, json.JSONDecodeError):
            pass
        raise AgentError(f"OpenAI API request failed: {message}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AgentError(f"OpenAI API request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AgentError("OpenAI API returned invalid JSON.") from exc

    text = extract_response_text(data)
    try:
        config = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentError(f"OpenAI response was not valid config JSON: {text}") from exc
    if not isinstance(config, dict):
        raise AgentError("OpenAI response must be a JSON object.")
    return config


def extract_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    texts: list[str] = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                texts.append(text)
    if not texts:
        raise AgentError("OpenAI response did not contain text output.")
    return "\n".join(texts).strip()


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    meter_type = config.get("meter_type")
    if meter_type not in ("digital", "analog"):
        raise AgentError("meter_type must be 'digital' or 'analog'.")

    normalized: dict[str, Any] = {
        "meter_type": meter_type,
        "roi": config.get("roi"),
        "raw_text": str(config.get("raw_text") or ""),
        "value": config.get("value"),
        "unit": str(config.get("unit") or ""),
        "min_value": config.get("min_value"),
        "max_value": config.get("max_value"),
        "min_angle": config.get("min_angle"),
        "max_angle": config.get("max_angle"),
        "direction": config.get("direction"),
        "confidence": config.get("confidence"),
        "notes": str(config.get("notes") or ""),
    }
    if normalized["direction"] not in (None, "clockwise", "counterclockwise"):
        raise AgentError("direction must be clockwise, counterclockwise, or null.")
    return normalized


def validate_roi(roi: Any, image_path: Path) -> tuple[int, int, int, int] | None:
    if roi is None:
        return None
    if not isinstance(roi, list) or len(roi) != 4:
        raise AgentError("roi must be null or a list of [x, y, w, h].")
    try:
        x, y, w, h = (int(value) for value in roi)
    except (TypeError, ValueError) as exc:
        raise AgentError("roi values must be integers.") from exc
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise AgentError("roi must have x/y >= 0 and width/height > 0.")

    try:
        img = load_image(image_path)
    except ImageLoadError as exc:
        raise AgentError(str(exc)) from exc
    try:
        transposed = ImageOps.exif_transpose(img)
        if transposed is not img:
            img.close()
            img = transposed
        if x + w > img.width or y + h > img.height:
            raise AgentError(
                f"roi is outside the image bounds (image size: {img.width}x{img.height})."
            )
    finally:
        img.close()
    return x, y, w, h


def crop_roi(image_path: Path, roi: tuple[int, int, int, int], output_dir: Path) -> Path:
    try:
        img = load_image(image_path)
    except ImageLoadError as exc:
        raise AgentError(str(exc)) from exc

    try:
        transposed = ImageOps.exif_transpose(img)
        if transposed is not img:
            img.close()
            img = transposed
        x, y, w, h = roi
        cropped = img.crop((x, y, x + w, y + h))
    finally:
        img.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_agent_roi.png"
    try:
        cropped.save(output_path)
    except OSError as exc:
        raise AgentError(f"Failed to save ROI crop: {output_path} ({exc})") from exc
    finally:
        cropped.close()
    return output_path


def as_float(config: dict[str, Any], key: str) -> float | None:
    value = config.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AgentError(f"{key} must be a number or null.") from exc


def build_read_command(
    script_path: Path,
    image_path: Path,
    config: dict[str, Any],
    digits_only: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path),
        str(image_path),
        "--type",
        config["meter_type"],
    ]
    unit = config.get("unit")
    if unit:
        command.extend(["--unit", str(unit)])

    if config["meter_type"] == "digital":
        if digits_only:
            command.append("--digits-only")
        return command

    min_value = as_float(config, "min_value")
    max_value = as_float(config, "max_value")
    if min_value is None or max_value is None:
        raise AgentError("Analog meters require min_value and max_value.")
    command.extend(["--min-value", f"{min_value:g}", "--max-value", f"{max_value:g}"])

    for key, flag in (("min_angle", "--min-angle"), ("max_angle", "--max-angle")):
        value = as_float(config, key)
        if value is not None:
            command.extend([flag, f"{value:g}"])
    if config.get("direction"):
        command.extend(["--direction", str(config["direction"])])
    return command


def build_analog_local_command(
    script_path: Path,
    image_path: Path,
    config: dict[str, Any],
) -> list[str]:
    min_value = as_float(config, "min_value")
    max_value = as_float(config, "max_value")
    if min_value is None or max_value is None:
        raise AgentError("Analog meters require min_value and max_value.")

    command = [
        sys.executable,
        str(script_path),
        str(image_path),
        "--type",
        "analog",
        "--min-value",
        f"{min_value:g}",
        "--max-value",
        f"{max_value:g}",
    ]
    unit = config.get("unit")
    if unit:
        command.extend(["--unit", str(unit)])
    return command


def render_api_reading(config: dict[str, Any]) -> str:
    meter_type = config["meter_type"]
    value = config.get("value")
    raw_text = config.get("raw_text") or ""
    if value is None or value == "":
        raise AgentError(f"OpenAI did not return a {meter_type} meter value.")

    lines = [
        f"type: {meter_type}",
        f"value: {value}",
        f"unit: {config.get('unit') or ''}",
    ]
    if raw_text:
        lines.append(f"raw_text: {raw_text}")
    confidence = config.get("confidence")
    if confidence is not None:
        lines.append(f"confidence: {confidence}")
    notes = config.get("notes")
    if notes:
        lines.append(f"notes: {notes}")
    return "\n".join(lines)


def run_reader(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, text=True, capture_output=True, check=False)
    except OSError as exc:
        raise AgentError(f"Failed to run reader: {exc}") from exc


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    env_path = Path(args.env_file).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    try:
        validate_input(input_path)
        load_env_file(env_path)
        config = normalize_config(call_openai(input_path, args.model, args.detail))
        if config["meter_type"] == "digital":
            reading = render_api_reading(config)
            reading_source = "openai"
            reader_input = input_path
            result_returncode = 0
            result_stderr = ""
        else:
            reader_input = input_path
            command = build_analog_local_command(
                Path(__file__).resolve().with_name("read_meter.py"),
                reader_input,
                config,
            )
            result = run_reader(command)
            if result.returncode == 0:
                reading = result.stdout.rstrip()
                reading_source = "local_analog_reader"
                result_returncode = 0
                result_stderr = result.stderr.strip()
            elif config.get("value") not in (None, ""):
                reading = render_api_reading(config)
                reading_source = "openai_fallback"
                result_returncode = 0
                result_stderr = (
                    "Warning: Local analog reader failed; using OpenAI value. "
                    f"{result.stderr.strip()}"
                ).strip()
            else:
                reading = result.stdout.rstrip()
                reading_source = "local_analog_reader"
                result_returncode = result.returncode
                result_stderr = result.stderr.strip()
    except AgentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("config:")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(f"reading_source: {reading_source}")
    print(f"reader_input: {reader_input}")
    print("reading:")
    if reading:
        print(reading)
    if result_stderr:
        print(result_stderr, file=sys.stderr)
    return result_returncode


if __name__ == "__main__":
    raise SystemExit(main())
