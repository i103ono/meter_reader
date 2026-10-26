#!/usr/bin/env python3
"""Download a small DEVELOPMENT set from official FlagEval/MeasureBench.

Install:
    python -m pip install huggingface_hub pyarrow pillow
Run:
    python download_measurebench_samples.py --pressure 20 --digital 20
Or download data/real_world-00000-of-00001.parquet from the dataset page, then:
    python download_measurebench_samples.py --parquet PATH_TO_FILE

The real_world parquet is about 99.5 MB; this script fetches that file, not
the repository's Blender assets or synthetic split. Outputs are raw image
files, inputs.jsonl, labels.jsonl, labels.csv, and provenance.json.
Only public development data are processed. No VLM or training is run.

Important:
* real_world can include drawings; visually review the selected samples.
* interval labels are retained as bounds, never replaced by their midpoint.
* do not pass labels to the reading model.
* grouping by source URL is a convenience, not proof of distinct physical gauges.
* CC BY-SA 4.0 is the dataset card's stated license; retain attribution.
Sources (checked 2026-10-27):
https://huggingface.co/datasets/FlagEval/MeasureBench
https://huggingface.co/docs/huggingface_hub/guides/download
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATASET_ID = "FlagEval/MeasureBench"
DATA_FILE = "data/real_world-00000-of-00001.parquet"
FIELDS = [
    "question_id", "question", "image", "image_type", "design",
    "evaluator", "evaluator_kwargs", "meta_info",
]


def json_object(value):
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object in an annotation field.")
    return value


def group_name(row):
    design = str(row.get("design", "")).lower()
    kind = re.sub(r"[^a-z0-9]+", "_", str(row.get("image_type", "")).lower())
    qid = str(row.get("question_id", "")).lower()
    if design == "digital":
        return "digital"
    if design == "dial" and (
        ("pressure" in kind and "gauge" in kind)
        or qid.startswith("pressure_gauge_")
    ):
        return "pressure"
    return None


def source_key(row):
    meta = json_object(row.get("meta_info"))
    return str(meta.get("source") or row["question_id"])


def select_rows(rows, pressure, digital, seed, max_per_source):
    candidates = sorted(
        (r for r in rows if group_name(r)),
        key=lambda r: str(r["question_id"]),
    )
    random.Random(seed).shuffle(candidates)
    targets = {"pressure": pressure, "digital": digital}
    counts, sources = Counter(), Counter()
    selected = []
    for row in candidates:
        group = group_name(row)
        key = (group, source_key(row))
        if counts[group] >= targets[group] or sources[key] >= max_per_source:
            continue
        json_object(row.get("evaluator_kwargs"))
        selected.append(row)
        counts[group] += 1
        sources[key] += 1
    return selected, counts


def numeric_interval(annotation):
    interval = annotation.get("interval")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return None, None
    valid = lambda x: (
        isinstance(x, (int, float)) and not isinstance(x, bool)
        and math.isfinite(x)
    )
    if not all(valid(x) for x in interval):
        return None, None
    if interval[0] > interval[1]:
        raise ValueError("Ground-truth interval bounds are reversed.")
    return interval[0], interval[1]


def encode_image(row):
    from PIL import Image
    payload = row["image"]
    if not isinstance(payload, dict) or not payload.get("bytes"):
        raise ValueError(
            f"{row['question_id']}: no embedded image bytes; inspect this parquet schema."
        )
    raw = bytes(payload["bytes"])
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        size = image.size
        fmt = image.format
        if fmt in ("JPEG", "PNG"):
            return raw, ".jpg" if fmt == "JPEG" else ".png", size, []
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG")
        return output.getvalue(), ".png", size, ["format_conversion_to_png"]


def save_rows(selected, output):
    # Refuse overwrites so an earlier manual selection remains intact.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --output.")
    prepared = []
    for row in selected:
        data, suffix, size, transforms = encode_image(row)
        qid = str(row["question_id"])
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", qid).strip(".") or "sample"
        stem += "_" + hashlib.sha256(qid.encode()).hexdigest()[:8]
        prepared.append((row, f"images/{stem}{suffix}", data, size, transforms))
    names = [item[1] for item in prepared]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate question IDs; inspect the dataset before exporting.")
    (output / "images").mkdir(parents=True, exist_ok=True)
    labels, inputs = [], []
    for row, relative, data, size, transforms in prepared:
        (output / relative).write_bytes(data)
        annotation = json_object(row.get("evaluator_kwargs"))
        lower, upper = numeric_interval(annotation)
        meta = json_object(row.get("meta_info"))
        labels.append({
            "question_id": row["question_id"],
            "filename": relative,
            "group": group_name(row),
            "image_type": row.get("image_type"),
            "design": row.get("design"),
            "evaluator": row.get("evaluator"),
            "evaluator_kwargs": annotation,
            "truth_lower": lower,
            "truth_upper": upper,
            "truth_units": annotation.get("units"),
            "source_group": source_key(row),
            "meta_info": meta,
            "width": size[0],
            "height": size[1],
            "image_sha256": hashlib.sha256(data).hexdigest(),
            "transformations": transforms,
            "use_for": "development",
            "needs_visual_review": True,
        })
        inputs.append({
            "question_id": row["question_id"],
            "filename": relative,
            "question": row.get("question"),
        })
    for name, records in (("inputs.jsonl", inputs), ("labels.jsonl", labels)):
        with (output / name).open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    csv_fields = [
        "question_id", "filename", "group", "image_type", "design",
        "evaluator", "truth_lower", "truth_upper", "truth_units_json",
        "evaluator_kwargs_json", "source_group", "source_url", "uploader",
    ]
    with (output / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for label in labels:
            record = {key: label.get(key) for key in csv_fields}
            record["truth_units_json"] = json.dumps(label["truth_units"], ensure_ascii=False)
            record["evaluator_kwargs_json"] = json.dumps(
                label["evaluator_kwargs"], ensure_ascii=False
            )
            record["source_url"] = label["meta_info"].get("source", "")
            record["uploader"] = label["meta_info"].get("uploader", "")
            writer.writerow(record)
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure", type=int, default=20, help="Maximum pressure-gauge samples.")
    parser.add_argument("--digital", type=int, default=20, help="Maximum digital samples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-source", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/dev_external/measurebench"))
    parser.add_argument("--parquet", type=Path, help="Use an already-downloaded real_world parquet.")
    parser.add_argument("--revision", help="Dataset revision; online mode resolves it to a commit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pressure < 0 or args.digital < 0 or args.pressure + args.digital == 0:
        raise ValueError("Request at least one sample; counts must be nonnegative.")
    if args.max_per_source < 1:
        raise ValueError("--max-per-source must be positive.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty or new --output directory.")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "Install dependencies: python -m pip install huggingface_hub pyarrow pillow"
        ) from exc
    if args.parquet:
        path = args.parquet
        revision = args.revision or "local-file-unpinned"
    else:
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().dataset_info(DATASET_ID, revision=args.revision)
        revision = info.sha
        print("Downloading the real_world parquet (about 99.5 MB).", file=sys.stderr)
        path = Path(hf_hub_download(
            repo_id=DATASET_ID, filename=DATA_FILE, repo_type="dataset",
            revision=revision,
        ))
    rows = pq.read_table(path, columns=FIELDS).to_pylist()
    selected, counts = select_rows(
        rows, args.pressure, args.digital, args.seed, args.max_per_source
    )
    if not selected:
        raise ValueError("No requested categories found. Inspect image_type and design.")
    labels = save_rows(selected, args.output)
    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_page": "https://huggingface.co/datasets/FlagEval/MeasureBench",
        "dataset_revision": revision,
        "split": "real_world",
        "license_as_stated_by_dataset": "CC-BY-SA-4.0",
        "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "max_per_source": args.max_per_source,
        "requested": {"pressure": args.pressure, "digital": args.digital},
        "saved": {key: counts[key] for key in ("pressure", "digital")},
        "question_ids": [label["question_id"] for label in labels],
        "purpose": "development_only; visually select actual photographs",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "saved": provenance["saved"]}, ensure_ascii=False))
    for group, wanted in (("pressure", args.pressure), ("digital", args.digital)):
        if counts[group] < wanted:
            print(
                f"Note: saved {counts[group]}/{wanted} {group} samples with the source limit.",
                file=sys.stderr,
            )
    print("Review photographs manually; keep labels separate from model inputs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

