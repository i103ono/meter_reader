#!/usr/bin/env python3
"""Mock or explicitly configured VLM reads, with no OCR model downloads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pydantic import ValidationError

from image_io import ImageLoadError, xywh_to_xyxy
from reader import MockReader
from recording import run_observation
from schemas import ObservationInput


def load_inputs(path: Path) -> list[ObservationInput]:
    """Relative image paths are relative to the manifest, never the shell cwd."""
    records = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                record = ObservationInput.model_validate(data)
                if not record.image_path.is_absolute():
                    record = record.model_copy(update={"image_path": path.resolve().parent / record.image_path})
                records.append(record)
            except ValidationError as exc:
                fields = ["/".join(map(str, item["loc"])) for item in exc.errors(include_input=False)]
                raise ValueError(f"{path}:{line_number}: invalid input fields: {', '.join(fields)}") from None
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError("inputs.jsonl contains no input records")
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Save a v0.2 mock run or an uncorrected-ROI VLM read.")
    parser.add_argument("image", nargs="?", type=Path, help="One JPEG/PNG input")
    parser.add_argument("--inputs", type=Path, help="Input-only JSONL manifest (never labels)")
    roi = parser.add_mutually_exclusive_group()
    roi.add_argument("--roi", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    roi.add_argument("--legacy-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                     help="Explicitly convert the legacy xywh ROI to new xyxy coordinates")
    proposal = parser.add_mutually_exclusive_group()
    proposal.add_argument("--auto-roi", action="store_true",
                          help="Ask the configured VLM to locate the target before independent reads")
    proposal.add_argument("--roi-proposal", type=Path,
                          help="Reuse roi_proposal.json for the same image/target, without another locator call")
    parser.add_argument("--roi-only", action="store_true",
                        help="Save the ROI proposal, crop and review preview without calling the Reader")
    parser.add_argument("--target-type", choices=("auto", "analog", "digital"))
    parser.add_argument("--meter-id")
    parser.add_argument("--observation-id")
    parser.add_argument("--captured-at", help="Timezone-aware ISO 8601 capture time; unknown if omitted")
    parser.add_argument("--question", help="Which display/scale to read; retained and forwarded to Reader")
    parser.add_argument("--exif-timezone", help="Explicit IANA zone for timezone-less DateTimeOriginal")
    parser.add_argument("--mode", choices=("mock", "inference"), default="mock")
    parser.add_argument("--provider-config", type=Path, help="Non-secret OpenAI provider JSON configuration")
    parser.add_argument("--model", help="Explicit OpenAI model ID; required for inference unless in provider config")
    parser.add_argument("--api-key-env", help="Credential variable name, never the credential value")
    parser.add_argument("--limit", type=int, help="Process at most N records")
    parser.add_argument("--recipes", nargs="+", metavar="RECIPE",
                        help="Generate up to two registered fixed corrections from the original ROI")
    parser.add_argument("--agent", action="store_true",
                        help="Let a bounded Planner choose at most two corrections (mock mode stays offline)")
    parser.add_argument("--comparison-policy", type=Path,
                        help="JSON containing analog_absolute_tolerance and unit; comparison only, no automatic acceptance")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    args = parser.parse_args(argv)
    if (args.image is None) == (args.inputs is None):
        parser.error("provide exactly one image path or --inputs")
    if args.inputs and any(value is not None for value in (
        args.roi, args.legacy_roi, args.target_type, args.meter_id,
        args.observation_id, args.captured_at, args.question, args.roi_proposal, args.comparison_policy,
    )):
        parser.error("with --inputs, set per-image options in JSONL instead of CLI overrides")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.mode == "mock" and any((args.provider_config, args.model, args.api_key_env)):
        parser.error("provider/model/credential settings require --mode inference")
    if args.auto_roi and args.mode != "inference":
        parser.error("--auto-roi requires --mode inference; mock mode never calls a locator API")
    if args.auto_roi and (args.roi is not None or args.legacy_roi is not None):
        parser.error("--auto-roi cannot be combined with manual ROI coordinates")
    if args.roi_only and (not (args.auto_roi or args.roi_proposal) or args.recipes is not None):
        parser.error("--roi-only requires --auto-roi or --roi-proposal, without --recipes")
    if args.agent and (args.recipes is not None or args.roi_only):
        parser.error("--agent cannot be combined with --recipes or --roi-only")
    if args.comparison_policy is not None and not args.agent:
        parser.error("--comparison-policy requires --agent")
    return args


def configured_reader(args):
    if args.mode == "mock":
        return MockReader()
    # Keep mock mode independent of provider initialization and credentials.
    from provider import OpenAIProvider, ProviderConfig
    from vlm_reader import VLMReader
    settings = json.loads(args.provider_config.read_text()) if args.provider_config else {}
    if not isinstance(settings, dict):
        raise ValueError("provider config must be a JSON object")
    if args.model is not None:
        settings["model"] = args.model
    if args.api_key_env is not None:
        settings["api_key_env"] = args.api_key_env
    return VLMReader(OpenAIProvider(ProviderConfig.model_validate(settings)))


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        policy = None
        if args.comparison_policy:
            from schemas import ComparisonPolicy
            policy = ComparisonPolicy.model_validate_json(args.comparison_policy.read_bytes())
        proposal = None
        if args.roi_proposal:
            from roi_locator import load_proposal
            proposal = load_proposal(args.roi_proposal)
        if args.inputs:
            requests = load_inputs(args.inputs)
        else:
            roi = xywh_to_xyxy(args.legacy_roi) if args.legacy_roi is not None else args.roi
            requests = [ObservationInput(
                image_path=args.image, roi=roi,
                target_type=args.target_type or (proposal.target_type if proposal else "auto"),
                meter_id=args.meter_id, observation_id=args.observation_id,
                captured_at=args.captured_at,
                question=args.question if args.question is not None else (proposal.question if proposal else None),
                acceptance_policy={"comparison": policy.model_dump(mode="json")} if policy else None,
            )]
        if args.limit is not None:
            requests = requests[:args.limit]
        if args.recipes is not None:
            from tools import validate_recipes
            validate_recipes(args.recipes)
        if args.auto_roi and any(request.roi is not None for request in requests):
            raise ValueError("--auto-roi cannot override a manual ROI in an input manifest")
        if args.agent:
            from validator import comparison_policy
            for request in requests:
                comparison_policy(request)
        reader = configured_reader(args)
        planner, comparator = None, None
        if args.agent:
            from planner import MockPlanner, MockComparator, VLMPlanner, VLMComparator
            if reader.mode == "mock":
                planner, comparator = MockPlanner(), MockComparator()
            else:
                planner, comparator = VLMPlanner(reader.provider), VLMComparator(reader.provider)
        locator = None
        if args.auto_roi:
            from roi_locator import ROILocator
            locator = ROILocator(reader.provider)
    except ValidationError as exc:
        # A mistakenly supplied secret field must not be echoed by validation errors.
        locations = ["/".join(map(str, item["loc"])) for item in exc.errors(include_input=False)]
        print("Error: invalid input/configuration fields: " + ", ".join(locations), file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    failed, saved = 0, 0
    for index, request in enumerate(requests, 1):
        try:
            result = run_observation(
                request, reader, args.output_dir, exif_timezone=args.exif_timezone,
                recipes=tuple(args.recipes or ()),
                roi_locator=locator, roi_proposal=proposal, roi_only=args.roi_only,
                roi_proposal_reference=str(args.roi_proposal.expanduser().resolve()) if args.roi_proposal else None,
                planner=planner, comparator=comparator,
            )
            saved += 1
            print(result, flush=True)
            saved_result = json.loads(result.read_text())
            budget_codes = set()
            if args.agent:
                from agent import BUDGET_CODES
                budget_codes = BUDGET_CODES
            candidate_error = any(
                candidate["status"] == "error" and (not args.agent or candidate.get("error_code") not in budget_codes)
                for candidate in saved_result.get("candidates", [])
            )
            role_error = any(event["status"] == "error" and event.get("error_code") not in budget_codes
                             for event in (saved_result.get("agent") or {}).get("events", []))
            if saved_result["status"] == "error" or candidate_error or role_error:
                failed += 1
        except (OSError, ValueError, ImageLoadError) as exc:
            failed += 1
            print(f"Error: input {index} ({request.image_path}): {exc}", file=sys.stderr)
    print(f"inputs: {len(requests)}, saved: {saved}, errors: {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
