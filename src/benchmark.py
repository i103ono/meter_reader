"""Run the predeclared A-D conditions from inputs only. Never load labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import sqlite3
import time
from uuid import uuid4

from cli import configured_reader, load_inputs
from image_io import prepare_image
from recording import run_observation
from schemas import DatasetReference, ObservationInput
from tools import RECIPES
from result_status import technical_failure
from repository import asset_in_run


def run_benchmark(requests, reader, output_dir, *, fixed_recipe="gamma_mild", allow_full_image_roi=False,
                  database=None, allow_perspective=False, unresolved_rois=None, progress=None, preparation_reference=None, resume=None):
    unresolved_rois = unresolved_rois or {}
    resume_path = Path(resume).expanduser().resolve() if resume else None
    if resume_path is not None and resume_path.name != "experiment.json":
        raise ValueError("Resume requires the original experiment.json path")
    previous = json.loads(resume_path.read_text()) if resume_path else None
    previous_ids = {row["case_id"]: row["observation_id"] for row in previous["conditions"]["A"]} if previous else {}
    if fixed_recipe not in RECIPES:
        raise ValueError("C requires exactly one predeclared parameter-free fixed recipe")
    if not requests:
        raise ValueError("Benchmark requires at least one input")
    if database is not None:
        from repository import Repository
        with Repository(database):
            pass
    prepared = []
    ids = set()
    # Validate all cases before the first outbound call. No labels are available here.
    for request in requests:
        request = ObservationInput.model_validate(request)
        question_id = request.dataset.question_id if request.dataset else None
        if question_id is None:
            raise ValueError("Evaluation cases require an input-only dataset.question_id")
        case_id = str(question_id)
        if request.roi is None and not allow_full_image_roi and case_id not in unresolved_rois:
            raise ValueError("B-D require a shared ROI; use --allow-full-image-roi only for plumbing checks")
        if case_id in unresolved_rois and request.roi is not None:
            raise ValueError("An unresolved ROI cannot also contain coordinates")
        if case_id in ids:
            raise ValueError("Benchmark question IDs must be unique")
        ids.add(case_id)
        image = prepare_image(request.image_path)
        try:
            if request.roi is not None:
                from image_io import validate_roi_xyxy
                validate_roi_xyxy(request.roi, image.image.size)
            source_hash = image.sha256
        finally:
            image.image.close()
        shared = request.model_copy(update={"observation_id": request.observation_id or previous_ids.get(case_id) or "obs_" + uuid4().hex})
        prepared.append((case_id, source_hash, shared))
    root = resume_path.parent if resume_path else Path(output_dir).expanduser().resolve() / ("experiment_" + uuid4().hex)
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    provider = getattr(reader, "provider", None)
    profile = {"reader_id": reader.reader_id, "mode": reader.mode}
    if provider:
        from vlm_reader import PROMPT_VERSION
        profile.update(model=provider.config.model, image_detail=provider.config.image_detail,
                       max_output_tokens=provider.config.max_output_tokens, prompt_version=PROMPT_VERSION)
    document = {"version": "meter-experiment-v1", "mode": reader.mode, "fixed_recipe": fixed_recipe,
                "allow_full_image_roi": allow_full_image_roi, "allow_perspective": allow_perspective,
                "database": str(Path(database).expanduser().resolve()) if database else None,
                "reader_profile": profile, "conditions": {name: [] for name in "ABCD"}}
    if preparation_reference is not None and not resume:
        raw_preparation = Path(preparation_reference).read_bytes()
        preparation_cases = json.loads(raw_preparation)["cases"]
        known_calls = sum(a["api_calls"] for row in preparation_cases for a in row["attempts"])
        (root / "roi_preparation.json").write_bytes(raw_preparation)
        document["roi_preparation"] = {"source": str(Path(preparation_reference).resolve()),
                                       "snapshot": "roi_preparation.json", "sha256": hashlib.sha256(raw_preparation).hexdigest(),
                                       "known_api_calls": known_calls,
                                       "api_calls": None if any(row["status"] == "running" or row.get("interrupted_attempts", 0) for row in preparation_cases) else known_calls}
    def save():
        temporary = root / "experiment.json.tmp"
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(root / "experiment.json")
    # Declare every expected case first so failures cannot shrink the denominator.
    for condition in "ABCD":
        for case_id, source_hash, request in prepared:
            document["conditions"][condition].append({
                "case_id": case_id, "question_id": case_id, "observation_id": request.observation_id,
                "input_sha256": source_hash, "roi": list(request.roi) if request.roi is not None and condition != "A" else None,
                "question": request.question, "target_type": request.target_type,
                "result_path": None, "result_sha256": None, "evaluation_candidate_id": None,
                "wall_elapsed_ms": None, "error": "not_started",
                "read_status": "not_started", "technical_error": True,
                "skip_reason": None,
                "state": "pending", "interrupted_attempts": 0,
            })
    if resume:
        for key in ("version", "mode", "fixed_recipe", "allow_full_image_roi", "allow_perspective", "reader_profile", "database"):
            if previous.get(key) != document[key]:
                raise ValueError("Resume settings differ from the original experiment: " + key)
        for condition in "ABCD":
            old_rows, new_rows = previous["conditions"][condition], document["conditions"][condition]
            if len(old_rows) != len(new_rows):
                raise ValueError("Resume case count changed")
            for old, new in zip(old_rows, new_rows):
                if any(old[k] != new[k] for k in ("case_id", "question_id", "observation_id", "input_sha256", "roi", "question", "target_type")):
                    raise ValueError("Resume inputs/ROI/target differ; start a new experiment")
        document = previous
    save()
    if progress:
        print("Experiment: " + str(root / "experiment.json"), file=sys.stderr)
    for condition in "ABCD":
        for index, (_, _, request) in enumerate(prepared):
            entry = document["conditions"][condition][index]
            if progress:
                progress(condition, index + 1, len(prepared), entry["question_id"])
            if resume and entry["state"] == "completed":
                if entry["result_path"]:
                    saved = asset_in_run(root, entry["result_path"])
                    if hashlib.sha256(saved.read_bytes()).hexdigest() != entry["result_sha256"]:
                        raise ValueError("Completed result changed; refusing resume")
                continue
            if resume and entry["state"] == "running" and entry["result_path"]:
                saved = asset_in_run(root, entry["result_path"])
                if hashlib.sha256(saved.read_bytes()).hexdigest() != entry["result_sha256"]:
                    raise ValueError("Interrupted result hash mismatch")
                if database:
                    from repository import Repository
                    with Repository(database) as repository:
                        repository.import_run(saved, select=False)
                entry.update(state="completed", error=None)
                save()
                continue
            if resume and entry["state"] == "running":
                entry["interrupted_attempts"] += 1
            if condition != "A" and entry["case_id"] in unresolved_rois:
                failure = unresolved_rois[entry["case_id"]]
                technical = failure not in ("ambiguous", "not_found")
                entry.update(skip_reason="roi_error" if technical else "roi_unresolved",
                             read_status="error" if technical else "needs_review", technical_error=technical,
                             error=failure if technical else None, wall_elapsed_ms=0, state="completed")
                save()
                continue
            selected_request = request.model_copy(update={"roi": None}) if condition == "A" else request
            started = time.monotonic()
            entry["state"] = "running"
            save()
            try:
                kwargs = {}
                if condition == "C":
                    kwargs["recipes"] = (fixed_recipe,)
                if condition == "D":
                    from planner import MockPlanner, MockComparator, VLMPlanner, VLMComparator
                    if reader.mode == "mock":
                        kwargs.update(planner=MockPlanner(), comparator=MockComparator())
                    else:
                        kwargs.update(planner=VLMPlanner(reader.provider), comparator=VLMComparator(reader.provider))
                    if allow_perspective:
                        from geometry_locator import MockGeometryLocator, VLMGeometryLocator
                        kwargs["geometry_locator"] = MockGeometryLocator() if reader.mode == "mock" else VLMGeometryLocator(reader.provider)
                path = run_observation(selected_request, reader, root / "runs", **kwargs)
                raw = path.read_bytes()
                result = json.loads(raw)
                entry.update(result_path=str(path.relative_to(root)), result_sha256=hashlib.sha256(raw).hexdigest(),
                             evaluation_candidate_id="c0" if condition in "AB" else ("c1" if condition == "C" else result["selected_candidate_id"]),
                             error=None, read_status=result["status"], technical_error=technical_failure(result))
                if database is not None:
                    from repository import Repository
                    with Repository(database) as repository:
                        repository.import_run(path, select=False)
            except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
                # Credentials and HTTP error bodies never reach these sanitized exceptions.
                entry["error"] = str(exc)
                entry["technical_error"] = True
            finally:
                entry["wall_elapsed_ms"] = max(0, (time.monotonic() - started) * 1000)
                # KeyboardInterrupt/SystemExit leave the checkpoint running.
                if sys.exc_info()[0] is None or isinstance(sys.exc_info()[1], Exception):
                    entry["state"] = "completed"
                save()
    return root / "experiment.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--prepared-batch", type=Path, help="batch_prepare.py state, including unresolved ROI cases")
    parser.add_argument("--resume", type=Path, help="Resume this experiment with identical inputs/settings; keep completed errors")
    parser.add_argument("--roi", nargs=4, type=int)
    parser.add_argument("--roi-proposal", type=Path)
    parser.add_argument("--question-id")
    parser.add_argument("--question")
    parser.add_argument("--target-type", choices=("auto", "analog", "digital"))
    parser.add_argument("--meter-id")
    parser.add_argument("--observation-id")
    parser.add_argument("--mode", choices=("mock", "inference"), default="mock")
    parser.add_argument("--provider-config", type=Path)
    parser.add_argument("--api-key-env")
    parser.add_argument("--model")
    parser.add_argument("--fixed-recipe", choices=tuple(RECIPES), default="gamma_mild")
    parser.add_argument("--allow-full-image-roi", action="store_true")
    parser.add_argument("--allow-perspective", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args(argv)
    if sum(v is not None for v in (args.image, args.inputs, args.prepared_batch)) != 1:
        parser.error("provide one image, --inputs, or --prepared-batch")
    if args.prepared_batch and any(v is not None for v in (args.roi, args.roi_proposal, args.question_id, args.question, args.target_type, args.meter_id, args.observation_id)):
        parser.error("prepared batches already contain per-image settings")
    if args.inputs and any(v is not None for v in (args.roi, args.roi_proposal, args.question_id, args.question, args.target_type, args.meter_id, args.observation_id)):
        parser.error("set per-image fields in the input-only JSONL")
    if args.roi and args.roi_proposal:
        parser.error("provide manual ROI or a saved ROI proposal")
    if args.mode == "mock" and any((args.model, args.provider_config, args.api_key_env)):
        parser.error("provider options require inference mode")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        unresolved = {}
        if args.prepared_batch:
            from batch_prepare import load_prepared_batch
            requests, unresolved = load_prepared_batch(args.prepared_batch)
        elif args.inputs:
            requests = load_inputs(args.inputs)
        else:
            if args.question_id is None:
                raise ValueError("Single-image benchmark requires --question-id for later label matching")
            roi, target_type, question, observation_id = args.roi, args.target_type or "auto", args.question, args.observation_id
            if args.roi_proposal:
                from roi_locator import load_proposal, validate_saved_proposal
                proposal = load_proposal(args.roi_proposal)
                target_type = args.target_type or proposal.target_type
                question = args.question if args.question is not None else proposal.question
                image = prepare_image(args.image)
                try:
                    validate_saved_proposal(proposal, image, target_type=target_type, question=question)
                finally:
                    image.image.close()
                if proposal.status != "located":
                    raise ValueError("Benchmark requires a resolved saved ROI")
                roi, observation_id = proposal.proposed_roi, observation_id or proposal.observation_id
            requests = [ObservationInput(image_path=args.image, roi=roi, question=question, target_type=target_type,
                                         observation_id=observation_id, meter_id=args.meter_id,
                                         dataset=DatasetReference(question_id=args.question_id))]
        if args.limit:
            requests = requests[:args.limit]
        path = run_benchmark(requests, configured_reader(args), args.output_dir, fixed_recipe=args.fixed_recipe,
                             allow_full_image_roi=args.allow_full_image_roi, database=args.database,
                             allow_perspective=args.allow_perspective, unresolved_rois=unresolved,
                             preparation_reference=args.prepared_batch,
                             resume=args.resume,
                             progress=lambda c,i,n,q: print(f"[{c} {i}/{n}] {q}", file=sys.stderr))
        print(path)
        document = json.loads(path.read_text())
        return 1 if any(row["technical_error"] for entries in document["conditions"].values() for row in entries) else 0
    except (OSError, ValueError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
