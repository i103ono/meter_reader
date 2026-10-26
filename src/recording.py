"""A single observation read, saved in its own exclusively created run directory."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from image_io import ImageLoadError, prepare_image, save_png_exclusive, validate_roi_xyxy
from reader import Reader, ReaderError, ReaderInput
from schemas import ObservationInput, ReaderOutput, RunResult, SourceImage
from timestamps import resolve_capture_time


def run_observation(
    request: ObservationInput,
    reader: Reader,
    output_dir: str | Path = "outputs/runs",
    *,
    exif_timezone: str | None = None,
) -> Path:
    prepared = prepare_image(request.image_path)
    try:
        roi = validate_roi_xyxy(request.roi, prepared.image.size) if request.roi is not None else None
        capture = resolve_capture_time(
            request, prepared.exif_datetime_original, prepared.exif_offset_original, exif_timezone,
        )
        if reader.mode not in ("mock", "inference"):
            raise ValueError("Reader mode must be mock or inference")
        run_id = "run_" + uuid4().hex
        observation_id = request.observation_id or "obs_" + uuid4().hex
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        # IDs from input are never used as directory names. A collision fails safely.
        run_dir = root / run_id
        run_dir.mkdir(exist_ok=False)
        source_name = "source.jpg" if prepared.format == "JPEG" else "source.png"
        with (run_dir / source_name).open("xb") as handle:
            handle.write(prepared.content)
        save_png_exclusive(prepared.image, run_dir / "normalized.png")
        with prepared.image.crop(roi or (0, 0, *prepared.image.size)) as crop:
            save_png_exclusive(crop, run_dir / "roi.png")
        reading = None
        error = None
        status = "needs_review"
        reasons = ["mock_result"] if reader.mode == "mock" else ["acceptance_policy_not_calibrated"]
        try:
            reading = ReaderOutput.model_validate(reader.read(ReaderInput(
                image_path=run_dir / "roi.png",
                target_type=request.target_type,
                question=request.question,
            )))
        except (ReaderError, ValidationError) as exc:
            status, reasons, error = "error", ["reader_error"], str(exc)
        # Detect a source modified during the run; it must never be silently accepted.
        if hashlib.sha256(prepared.original_path.read_bytes()).hexdigest() != prepared.sha256:
            raise ImageLoadError("Input image changed during processing")
        result = RunResult(
            run_id=run_id, observation_id=observation_id, meter_id=request.meter_id,
            captured_at=capture.captured_at, captured_timezone=capture.timezone,
            timestamp_source=capture.source, timestamp_metadata=capture.metadata,
            processed_at=datetime.now(timezone.utc),
            input_sha256=prepared.sha256,
            source_image=SourceImage(
                original_path=str(prepared.original_path), stored_path=source_name,
                format=prepared.format, original_size=prepared.original_size,
                normalized_size=prepared.image.size, exif_orientation=prepared.exif_orientation,
            ),
            roi=roi, target_type=request.target_type, question=request.question,
            dataset=request.dataset, acceptance_policy=request.acceptance_policy,
            mode=reader.mode, reader_id=reader.reader_id, status=status,
            reading=reading, reason_codes=reasons,
            api_calls=0 if reader.mode == "mock" else None, error=error,
        )
        path = run_dir / "result.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result.model_dump(mode="json"), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        return path
    finally:
        prepared.image.close()
