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
from schemas import CandidateResult, InferenceTrace, ObservationInput, ReaderOutput, ROIProposal, RunResult, SourceImage
from timestamps import resolve_capture_time


def _read_asset(reader: Reader, path: Path, request: ObservationInput) -> dict:
    reading, error = None, None
    status = "needs_review"
    reasons = ["mock_result"] if reader.mode == "mock" else ["acceptance_policy_not_calibrated"]
    previous_trace = getattr(reader, "last_trace", None)
    try:
        reading = ReaderOutput.model_validate(reader.read(ReaderInput(
            image_path=path, target_type=request.target_type, question=request.question,
        )))
    except ReaderError as exc:
        status, reasons, error = "error", [exc.code], str(exc)
    except ValidationError:
        status, reasons, error = "error", ["invalid_reading"], "Reader returned an invalid reading."
    if status != "error" and reader.mode == "inference" and reading.value is None and reading.unit is None:
        status, reasons = "unreadable", ["value_and_unit_unreadable"]
    trace = getattr(reader, "last_trace", None)
    if trace is previous_trace:
        trace = None  # A failed custom adapter must not reuse a previous candidate's trace.
    if trace is not None:
        trace = InferenceTrace.model_validate(trace)
    return {
        "reading": reading, "status": status, "reasons": reasons, "error": error,
        "inference": trace,
        "api_calls": len(trace.attempts) if trace is not None else (0 if reader.mode == "mock" else None),
    }


def _candidates(reader, request, run_dir, recipe_ids, baseline):
    from PIL import Image
    from quality import measure_quality
    from tools import CorrectionError, apply_recipe, pixel_sha256, recipe_steps, runtime_versions
    roi_path = run_dir / "roi.png"
    roi_hash = hashlib.sha256(roi_path.read_bytes()).hexdigest()
    runtime = runtime_versions()
    with Image.open(roi_path) as roi_image:
        roi_image.load()
        candidates = [CandidateResult(
            candidate_id="c0", parent_candidate_id=None, recipe_id=None, runtime=runtime,
            input_roi_sha256=roi_hash, image_path="roi.png", image_sha256=roi_hash,
            pixel_sha256=pixel_sha256(roi_image), image_size=roi_image.size, image_mode=roi_image.mode,
            quality=measure_quality(roi_image), status=baseline["status"], reading=baseline["reading"],
            inference=baseline["inference"], api_calls=baseline["api_calls"], error=baseline["error"],
            error_code=baseline["reasons"][0] if baseline["status"] == "error" else None,
        )]
        (run_dir / "candidates").mkdir()
        for index, recipe_id in enumerate(recipe_ids, 1):
            candidate_id = f"c{index}"
            steps = recipe_steps(recipe_id, roi_image.size)
            path = run_dir / "candidates" / f"{candidate_id}_{recipe_id}.png"
            try:
                check_budget = getattr(reader, "check_observation_budget", None)
                if check_budget is not None:
                    check_budget()
                correction = apply_recipe(roi_image, recipe_id)
                try:
                    save_png_exclusive(correction.image, path)
                    image_info = {
                        "image_path": str(path.relative_to(run_dir)), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "pixel_sha256": pixel_sha256(correction.image), "image_size": correction.image.size,
                        "image_mode": correction.image.mode, "quality": measure_quality(correction.image),
                    }
                finally:
                    correction.image.close()
            except (CorrectionError, OSError, ReaderError) as exc:
                candidates.append(CandidateResult(
                    candidate_id=candidate_id, parent_candidate_id="c0", recipe_id=recipe_id,
                    steps=steps, runtime=runtime, input_roi_sha256=roi_hash,
                    image_path=None, image_sha256=None, pixel_sha256=None, image_size=None,
                    image_mode=None, quality=None, status="error", reading=None, inference=None,
                    api_calls=0, error_code=exc.code if isinstance(exc, ReaderError) else "correction_error",
                    error=str(exc),
                ))
                continue
            observation = _read_asset(reader, path, request)
            candidates.append(CandidateResult(
                candidate_id=candidate_id, parent_candidate_id="c0", recipe_id=recipe_id,
                steps=steps, runtime=runtime, input_roi_sha256=roi_hash, **image_info,
                status=observation["status"], reading=observation["reading"],
                inference=observation["inference"], api_calls=observation["api_calls"],
                error=observation["error"],
                error_code=observation["reasons"][0] if observation["status"] == "error" else None,
            ))
    if hashlib.sha256(roi_path.read_bytes()).hexdigest() != roi_hash:
        raise ImageLoadError("Original ROI changed during candidate processing")
    return candidates


def run_observation(
    request: ObservationInput,
    reader: Reader,
    output_dir: str | Path = "outputs/runs",
    *,
    exif_timezone: str | None = None,
    recipes: tuple[str, ...] = (),
    roi_locator=None,
    roi_proposal: ROIProposal | None = None,
    roi_proposal_reference: str | None = None,
    roi_only: bool = False,
) -> Path:
    if not isinstance(recipes, (list, tuple)):
        raise ValueError("recipes must be a list of registered recipe IDs")
    if recipes:
        from tools import validate_recipes
        recipes = validate_recipes(recipes)
    if roi_locator is not None and (roi_proposal is not None or request.roi is not None):
        raise ValueError("Automatic ROI cannot be combined with a saved proposal or manual ROI")
    if roi_locator is not None and (
        reader.mode != "inference" or getattr(reader, "provider", None) is not roi_locator.provider
    ):
        raise ValueError("ROI locator and Reader must share one inference provider and budget")
    if roi_only and (recipes or roi_locator is None and roi_proposal is None):
        raise ValueError("ROI-only mode requires an ROI proposal and no correction recipes")
    begin_observation = getattr(reader, "begin_observation", None)
    if begin_observation is not None:
        begin_observation()
    prepared = prepare_image(request.image_path)
    try:
        roi = validate_roi_xyxy(request.roi, prepared.image.size) if request.roi is not None else None
        if roi_proposal is not None:
            from roi_locator import validate_saved_proposal
            validate_saved_proposal(roi_proposal, prepared, target_type=request.target_type, question=request.question)
        capture = resolve_capture_time(
            request, prepared.exif_datetime_original, prepared.exif_offset_original, exif_timezone,
        )
        if reader.mode not in ("mock", "inference"):
            raise ValueError("Reader mode must be mock or inference")
        run_id = "run_" + uuid4().hex
        observation_id = request.observation_id or (roi_proposal.observation_id if roi_proposal else "obs_" + uuid4().hex)
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        # IDs from input are never used as directory names. A collision fails safely.
        run_dir = root / run_id
        run_dir.mkdir(exist_ok=False)
        source_name = "source.jpg" if prepared.format == "JPEG" else "source.png"
        with (run_dir / source_name).open("xb") as handle:
            handle.write(prepared.content)
        save_png_exclusive(prepared.image, run_dir / "normalized.png")
        roi_calls, roi_preview_path = 0, None
        roi_source = "manual" if roi is not None else "full_image"
        if roi_locator is not None:
            roi_proposal = roi_locator.propose(
                prepared, target_type=request.target_type, question=request.question,
                observation_id=observation_id, run_dir=run_dir,
            )
            roi_source = "vlm"
            roi_calls = len(roi_proposal.inference.attempts)
        elif roi_proposal is not None:
            roi_source = "manual_override" if roi is not None else "saved_vlm"
        if roi_proposal is not None:
            from roi_locator import save_preview
            if roi is None:
                roi = roi_proposal.proposed_roi
            with (run_dir / "roi_proposal.json").open("x", encoding="utf-8") as handle:
                handle.write(roi_proposal.model_dump_json(indent=2) + "\n")
            roi_preview_path = "roi_preview.png"
            save_preview(prepared.image, run_dir / roi_preview_path,
                         proposed_roi=roi_proposal.proposed_roi, selected_roi=roi,
                         manual_override=roi_source == "manual_override")
        roi_ready = roi_proposal is None or roi is not None
        if roi_ready:
            with prepared.image.crop(roi or (0, 0, *prepared.image.size)) as crop:
                save_png_exclusive(crop, run_dir / "roi.png")
        reading, trace, error, candidates = None, None, None, []
        status, reasons, api_calls = "needs_review", [], roi_calls
        if not roi_ready:
            reasons = ["roi_" + roi_proposal.status, "reading_skipped_roi_unresolved"]
            if roi_proposal.status == "error":
                status, reasons, error = "error", [roi_proposal.error_code], roi_proposal.error
        elif roi_only:
            reasons = ["roi_only_reading_not_requested"]
        else:
            saved_roi_sha256 = hashlib.sha256((run_dir / "roi.png").read_bytes()).hexdigest()
            baseline = _read_asset(reader, run_dir / "roi.png", request)
            if hashlib.sha256((run_dir / "roi.png").read_bytes()).hexdigest() != saved_roi_sha256:
                raise ImageLoadError("Original ROI changed during baseline reading")
            reading, status, reasons, error, trace = (
                baseline[key] for key in ("reading", "status", "reasons", "error", "inference")
            )
            candidates = _candidates(reader, request, run_dir, recipes, baseline) if recipes else []
            api_calls = baseline["api_calls"] + roi_calls if baseline["api_calls"] is not None else None
        if candidates:
            counts = [candidate.api_calls for candidate in candidates]
            api_calls = roi_calls + sum(counts) if all(count is not None for count in counts) else None
            reasons = [*reasons, "fixed_candidates_not_selected"]
            if status != "error" and any(candidate.status == "error" for candidate in candidates[1:]):
                status = "needs_review"
                reasons.append("correction_candidate_error")
            if status == "unreadable" and any(
                candidate.reading is not None and (
                    candidate.reading.value is not None or candidate.reading.unit is not None
                ) for candidate in candidates[1:]
            ):
                status = "needs_review"
        if roi_proposal is not None:
            reasons = [*reasons, "roi_manually_overridden" if roi_source == "manual_override" else "roi_proposal_unverified"]
        # Detect a source modified during the run; it must never be silently accepted.
        if hashlib.sha256(prepared.original_path.read_bytes()).hexdigest() != prepared.sha256:
            raise ImageLoadError("Input image changed during processing")
        result = RunResult(
            stage=3 if recipes else getattr(reader, "stage", 1),
            run_id=run_id, observation_id=observation_id, meter_id=request.meter_id,
            captured_at=capture.captured_at, captured_timezone=capture.timezone,
            timestamp_source=capture.source, timestamp_metadata=capture.metadata,
            processed_at=datetime.now(timezone.utc),
            input_sha256=prepared.sha256,
            source_image=SourceImage(
                original_path=str(prepared.original_path), stored_path=source_name,
                format=prepared.format, original_size=prepared.original_size,
                normalized_size=prepared.image.size, exif_orientation=prepared.exif_orientation,
                roi_path="roi.png" if roi_ready else None,
            ),
            roi=roi, target_type=request.target_type, question=request.question,
            roi_source=roi_source, roi_proposal=roi_proposal, roi_proposal_reference=roi_proposal_reference,
            roi_preview_path=roi_preview_path, read_requested=not roi_only,
            dataset=request.dataset, acceptance_policy=request.acceptance_policy,
            mode=reader.mode, reader_id=reader.reader_id, status=status,
            reading=reading, reason_codes=reasons,
            api_calls=api_calls,
            error=error, inference=trace,
            baseline_candidate_id="c0" if recipes else None,
            correction_candidates_generated=sum(candidate.image_path is not None for candidate in candidates[1:]),
            candidates=candidates,
        )
        path = run_dir / "result.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(result.model_dump(mode="json"), handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        return path
    finally:
        prepared.image.close()
