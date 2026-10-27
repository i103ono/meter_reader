"""Prepare/reuse a frozen ROI per image, without loading labels or measurements."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

from PIL import Image, ImageDraw, ImageOps

from cli import configured_reader, load_inputs
from image_io import prepare_image, save_png_exclusive, validate_roi_xyxy
from recording import run_observation
from repository import asset_in_run
from roi_locator import ROILocator, load_proposal, save_preview, validate_saved_proposal
from schemas import DatasetReference, ObservationInput


def load_source(path):
    path = Path(path).expanduser().resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Input list is empty")
    if all("image_path" in row for row in rows):
        return load_inputs(path)
    provenance_path = path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    records = []
    for row in rows:
        if set(row) != {"filename", "question_id", "question"}:
            raise ValueError("Use input-only JSONL: question_id/filename/question or typed ObservationInput; never labels")
        relative = Path(row["filename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Dataset filenames must be relative to inputs.jsonl")
        records.append(ObservationInput(
            image_path=(path.parent / relative).resolve(), question=row["question"],
            observation_id="measurebench_" + str(row["question_id"]),
            dataset=DatasetReference(question_id=row["question_id"], name=provenance.get("dataset_id"),
                                     split=provenance.get("split"), revision=provenance.get("dataset_revision"), seed=provenance.get("seed")),
        ))
    return records


def _snapshot(requests):
    cases, ids = [], set()
    for request in requests:
        if request.dataset is None or request.dataset.question_id is None:
            raise ValueError("Every image requires dataset.question_id")
        key = str(request.dataset.question_id)
        if key in ids:
            raise ValueError("Question IDs must be unique")
        ids.add(key)
        image = prepare_image(request.image_path)
        try:
            if request.roi is not None:
                validate_roi_xyxy(request.roi, image.image.size)
            request = request.model_copy(update={"image_path": image.original_path})
            cases.append({"question_id": key, "request": request.model_dump(mode="json"), "input_sha256": image.sha256,
                          "observation_id": request.observation_id or "obs_batch_" + uuid4().hex, "attempts": [],
                          "interrupted_attempts": 0,
                          "roi": list(request.roi) if request.roi else None,
                          "status": "manual" if request.roi else "pending", "result_path": None,
                          "preview_path": None, "error_code": None})
        finally:
            image.image.close()
    return cases


def _fingerprint(cases):
    return hashlib.sha256(json.dumps([{k: row[k] for k in ("question_id", "request", "input_sha256")} for row in cases],
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _write(root, document):
    temporary = root / "batch.json.tmp"
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(root / "batch.json")


def _contact_sheet(root, document):
    columns, width, height = 4, 300, 230
    count = len(document["cases"])
    with Image.new("RGB", (columns * width, ((count + columns - 1) // columns) * height), "white") as sheet:
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(document["cases"]):
            x, y = (index % columns) * width, (index // columns) * height
            if row.get("preview_path"):
                with Image.open(asset_in_run(root, row["preview_path"])) as image:
                    thumbnail = ImageOps.contain(image.convert("RGB"), (width, height - 30))
            else:
                prepared = prepare_image(row["request"]["image_path"])
                try:
                    thumbnail = ImageOps.contain(prepared.image.convert("RGB"), (width, height - 30))
                finally:
                    prepared.image.close()
            try:
                sheet.paste(thumbnail, (x, y + 30))
            finally:
                thumbnail.close()
            draw.text((x + 3, y + 3), row["question_id"][:38], fill="black")
            draw.text((x + 3, y + 16), row["status"], fill="green" if row["roi"] else "red")
        temporary = root / "roi_contact_sheet.tmp.png"
        sheet.save(temporary)
        temporary.replace(root / "roi_contact_sheet.png")


def prepare_batch(requests, reader, workspace, *, overrides=None, expected_count=None, progress=None, retry_unresolved=False):
    cases = _snapshot(requests)
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} images; found {len(cases)}")
    if not cases:
        raise ValueError("No images to prepare")
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError("ROI overrides must be a JSON object mapping question IDs to coordinates")
    if set(overrides) - {row["question_id"] for row in cases}:
        raise ValueError("ROI override refers to an unknown question ID")
    # Validate every override before any API call or workspace write.
    for row in cases:
        if row["question_id"] in overrides:
            prepared = prepare_image(row["request"]["image_path"])
            try:
                validate_roi_xyxy(overrides[row["question_id"]], prepared.image.size)
            finally:
                prepared.image.close()
    root = Path(workspace).expanduser().resolve()
    state = root / "batch.json"
    fingerprint = _fingerprint(cases)
    if state.exists():
        document = json.loads(state.read_text())
        if document.get("version") != "batch-roi-v1" or document.get("fingerprint") != fingerprint:
            raise ValueError("Workspace belongs to different inputs, questions or image hashes; choose a new workspace")
        if _fingerprint(document["cases"]) != fingerprint:
            raise ValueError("Prepared source records were modified; refusing to send changed inputs")
    else:
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise ValueError("Choose an empty batch workspace")
        document = {"version": "batch-roi-v1", "fingerprint": fingerprint, "cases": cases}
        _write(root, document)
    locator = ROILocator(reader.provider) if reader.mode == "inference" else None
    for index, row in enumerate(document["cases"], 1):
        request = ObservationInput.model_validate(row["request"])
        request = request.model_copy(update={"observation_id": row["observation_id"]})
        key = row["question_id"]
        if progress:
            progress(index, len(cases), key, row["status"])
        if row["status"] == "running":
            row["interrupted_attempts"] = row.get("interrupted_attempts", 0) + 1
            row["status"] = "pending"
        if key in overrides or (row["status"] == "manual" and row["roi"] is not None):
            roi = overrides.get(key, row["roi"])
            prepared = prepare_image(request.image_path)
            try:
                validate_roi_xyxy(roi, prepared.image.size)
                if list(roi) != row["roi"] or row["preview_path"] is None:
                    name = "manual_" + uuid4().hex + ".png"
                    save_preview(prepared.image, root / name, proposed_roi=None, selected_roi=roi, manual_override=True)
                    row["preview_path"] = name
                row.update(status="manual", roi=list(roi), error_code=None)
            finally:
                prepared.image.close()
        elif row["status"] == "located":
            saved = load_proposal(asset_in_run(root, row["result_path"]).parent / "roi_proposal.json")
            prepared = prepare_image(request.image_path)
            try:
                validate_saved_proposal(saved, prepared, target_type=request.target_type, question=request.question)
                if list(saved.proposed_roi) != row["roi"]:
                    raise ValueError("Prepared ROI differs from saved proposal; use an explicit override")
            finally:
                prepared.image.close()
        elif row["status"] in ("ambiguous", "not_found") and not retry_unresolved and row["error_code"] != "mock_does_not_locate_roi":
            pass
        elif locator is None:
            row.update(status="ambiguous", roi=None, error_code="mock_does_not_locate_roi")
        else:
            row["status"] = "running"
            _write(root, document)
            path = run_observation(request, reader, root / "roi_runs", roi_locator=locator, roi_only=True)
            result = json.loads(path.read_text())
            proposal = result["roi_proposal"]
            row.update(status=proposal["status"], roi=proposal["proposed_roi"], error_code=proposal["error_code"],
                       result_path=str(path.relative_to(root)), preview_path=str((path.parent / "roi_preview.png").relative_to(root)))
            row["attempts"].append({"result_path": row["result_path"], "status": row["status"], "api_calls": result["api_calls"]})
        _write(root, document)
        if row["error_code"] in ("credential_missing", "authentication_error", "permission_error"):
            break  # Keep pending cases, instead of repeating a broken credential for every image.
    _contact_sheet(root, document)
    return state


def load_prepared_batch(state):
    state = Path(state).expanduser().resolve()
    document = json.loads(state.read_text())
    if document.get("version") != "batch-roi-v1":
        raise ValueError("Unsupported prepared batch")
    requests = [ObservationInput.model_validate(row["request"]) for row in document["cases"]]
    if _fingerprint(document["cases"]) != document["fingerprint"]:
        raise ValueError("Prepared source records were modified")
    snapshot = _snapshot(requests)
    if _fingerprint(snapshot) != document["fingerprint"]:
        raise ValueError("Prepared batch inputs changed")
    if [row["question_id"] for row in snapshot] != [row["question_id"] for row in document["cases"]]:
        raise ValueError("Prepared case IDs changed")
    ready, unresolved = [], {}
    for request, row in zip(requests, document["cases"]):
        request = request.model_copy(update={"observation_id": row["observation_id"]})
        if row["status"] in ("located", "manual"):
            image = prepare_image(request.image_path)
            try:
                roi = validate_roi_xyxy(row["roi"], image.image.size)
                if row["status"] == "located":
                    proposal = load_proposal(asset_in_run(state.parent, row["result_path"]).parent / "roi_proposal.json")
                    validate_saved_proposal(proposal, image, target_type=request.target_type, question=request.question)
                    if roi != proposal.proposed_roi:
                        raise ValueError("Prepared coordinates differ from saved ROI proposal")
            finally:
                image.image.close()
            request = request.model_copy(update={"roi": roi})
        else:
            request = request.model_copy(update={"roi": None})
            unresolved[row["question_id"]] = row["status"] if row["status"] in ("ambiguous", "not_found") else (row["error_code"] or "roi_not_prepared")
        ready.append(request)
    return ready, unresolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, help="JSON object: question_id -> EXIF-normalized full-image xyxy ROI")
    parser.add_argument("--expect-count", type=int, default=20)
    parser.add_argument("--retry-unresolved", action="store_true", help="Request another ROI proposal for ambiguous/not_found cases")
    parser.add_argument("--mode", choices=("mock", "inference"), default="mock")
    parser.add_argument("--provider-config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--api-key-env")
    args = parser.parse_args(argv)
    if args.mode == "mock" and any((args.provider_config, args.model, args.api_key_env)):
        parser.error("provider options require inference mode")
    try:
        requests = load_source(args.inputs)
        overrides = json.loads(args.overrides.read_text()) if args.overrides else None
        path = prepare_batch(requests, configured_reader(args), args.workspace, overrides=overrides, expected_count=args.expect_count,
                             retry_unresolved=args.retry_unresolved,
                             progress=lambda i,n,q,s: print(f"[ROI {i}/{n}] {q}: {s}", file=sys.stderr))
        print(path)
        statuses = Counter(row["status"] for row in json.loads(path.read_text())["cases"])
        print("ROI states: " + json.dumps(statuses), file=sys.stderr)
        return 1 if statuses["error"] or statuses["pending"] else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
