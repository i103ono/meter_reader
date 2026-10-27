"""Import and inspect durable history, append human reviews, select read runs."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

from history_models import MeterRecord, ReviewInput
from repository import Repository


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("outputs/history/history.sqlite3"))
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("meter")
    register.add_argument("json_file", type=Path, help="MeterRecord JSON")
    importing = commands.add_parser("import")
    importing.add_argument("result", type=Path)
    importing.add_argument("--meter-id")
    importing.add_argument("--keep-selected", action="store_true")
    listing = commands.add_parser("list")
    listing.add_argument("--meter-id")
    detail = commands.add_parser("show")
    detail.add_argument("observation_id")
    run = commands.add_parser("run")
    run.add_argument("run_id")
    selecting = commands.add_parser("select")
    selecting.add_argument("observation_id")
    selecting.add_argument("run_id")
    reviewing = commands.add_parser("review")
    reviewing.add_argument("run_id")
    reviewing.add_argument("--action", required=True, choices=("confirm", "correct", "reject"))
    reviewing.add_argument("--candidate-id")
    reviewing.add_argument("--value")
    reviewing.add_argument("--unit")
    reviewing.add_argument("--scale-json", type=Path)
    reviewing.add_argument("--reason", required=True)
    reviewing.add_argument("--reviewer", required=True)
    reviewing.add_argument("--review-id")
    asset = commands.add_parser("asset")
    asset.add_argument("asset_id")
    args = parser.parse_args(argv)
    try:
        with Repository(args.database) as repository:
            if args.command == "meter":
                output = repository.register_meter(MeterRecord.model_validate_json(args.json_file.read_bytes()))
            elif args.command == "import":
                output = {"run_id": repository.import_run(args.result, meter_id=args.meter_id, select=not args.keep_selected)}
            elif args.command == "list":
                output = repository.list_observations(args.meter_id)
            elif args.command == "show":
                output = repository.get_observation(args.observation_id)
            elif args.command == "run":
                output = repository.get_run(args.run_id)
            elif args.command == "select":
                repository.select_run(args.observation_id, args.run_id)
                output = repository.get_observation(args.observation_id)
            elif args.command == "review":
                request = ReviewInput(run_id=args.run_id, action=args.action, candidate_id=args.candidate_id,
                                      value=args.value, unit=args.unit, reason=args.reason, reviewer=args.reviewer,
                                      review_id=args.review_id,
                                      scale=json.loads(args.scale_json.read_text()) if args.scale_json else None)
                output = {"review_id": repository.add_review(request)}
            else:
                output = {"path": str(repository.get_asset(args.asset_id))}
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
