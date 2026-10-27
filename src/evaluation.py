"""Offline v0.2 evaluation. Labels never enter the reading/benchmark runner."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import sys
from uuid import uuid4

from repository import asset_in_run
from schemas import RunResult
from result_status import technical_failure
from analog_diagnostics import diagnose_analog

ALIASES = {"m³": "m^3", "kW·h": "kWh", "kW h": "kWh", "m³/h": "m^3/h"}


def number(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("Evaluation numbers must be finite decimals")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("Evaluation numbers must be finite decimals") from None
    if not result.is_finite():
        raise ValueError("Evaluation numbers must be finite decimals")
    return result


def unit(value):
    return ALIASES.get(value.strip(), value.strip()) if isinstance(value, str) else None


def label_alternatives(row):
    kwargs = row.get("evaluator_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError("Ground truth requires original evaluator_kwargs")
    if row.get("evaluator", "interval_matching") == "interval_matching" and set(kwargs) == {"interval", "units"}:
        intervals, groups = [kwargs["interval"]], [kwargs["units"]]
    elif row.get("evaluator") == "multi_interval_matching" and set(kwargs) == {"intervals", "units"}:
        intervals, groups = kwargs["intervals"], kwargs["units"]
    else:
        raise ValueError("Unsupported evaluator/kwargs; supported: interval_matching and multi_interval_matching")
    if not isinstance(intervals, list) or not isinstance(groups, list) or not intervals or len(intervals) != len(groups):
        raise ValueError("Each ground-truth interval needs its corresponding unit group")
    result = []
    for interval, units in zip(intervals, groups):
        if not isinstance(interval, list) or len(interval) != 2 or number(interval[0]) > number(interval[1]):
            raise ValueError("Invalid closed ground-truth interval")
        if not isinstance(units, list) or not units or any(not isinstance(u, str) or not u.strip() for u in units):
            raise ValueError("Ground truth requires explicit unit candidates for each interval")
        result.append((number(interval[0]), number(interval[1]), units))
    return result


def load_labels(path, *, content=None):
    path = Path(path)
    text = (path.read_bytes() if content is None else content).decode("utf-8-sig")
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
        for row in rows:
            if "evaluator_kwargs_json" in row:
                row["evaluator_kwargs"] = json.loads(row["evaluator_kwargs_json"])
            if "truth_units_json" in row:
                row["truth_units"] = json.loads(row["truth_units_json"])
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    indexed = {}
    for row in rows:
        key = str(row.get("question_id", ""))
        if not key or key in indexed:
            raise ValueError("Labels require unique question_id values")
        if "evaluator" not in row:
            raise ValueError("Original evaluator name must be explicit")
        alternatives = label_alternatives(row)
        kwargs = row["evaluator_kwargs"]
        units = kwargs["units"]
        if row.get("truth_units") is not None and row["truth_units"] != units:
            raise ValueError("Label unit candidates disagree with original evaluator_kwargs")
        for index, field in enumerate(("truth_lower", "truth_upper")):
            if row.get(field) not in (None, "") and (len(alternatives) != 1 or number(row[field]) != alternatives[0][index]):
                raise ValueError("Label convenience bounds disagree with original evaluator_kwargs")
        indexed[key] = row
    if not indexed:
        raise ValueError("No evaluation labels")
    return indexed


def score_reading(reading, label):
    alternatives = label_alternatives(label)
    value = number(reading.value) if reading is not None and reading.value is not None else None
    predicted_unit = reading.unit if reading is not None else None
    matched = [i for i, (_, _, units) in enumerate(alternatives) if predicted_unit is not None and unit(predicted_unit) in {unit(u) for u in units}]
    unit_correct = bool(matched)
    numeric_intervals = alternatives if len(alternatives) == 1 else [alternatives[i] for i in matched]
    value_correct = value is not None and any(lower <= value <= upper for lower, upper, _ in numeric_intervals)
    distance = min(max(alternatives[i][0] - value, value - alternatives[i][1], Decimal(0)) for i in matched) if value is not None and matched else None
    point = alternatives[matched[0]][0] if len(matched) == 1 and alternatives[matched[0]][0] == alternatives[matched[0]][1] else None
    relative = distance / abs(point) if distance is not None and point is not None and point != 0 else None
    return {"value_correct": value_correct, "unit_correct": unit_correct, "both_correct": value_correct and unit_correct,
            "predicted_value": str(value) if value is not None else None, "predicted_unit": predicted_unit,
            "truth_interval": [str(alternatives[0][0]), str(alternatives[0][1])] if len(alternatives) == 1 else None,
            "truth_alternatives": [{"interval": [str(lower), str(upper)], "units": units} for lower, upper, units in alternatives],
            "matched_unit_interval_indexes": matched,
            "absolute_distance_to_interval": str(distance) if distance is not None else None,
            "relative_error": str(relative) if relative is not None else None}


def trace_totals(result):
    if result is None:
        return {"api_calls": 0, "input_tokens": None, "output_tokens": None, "estimated_cost_usd": None, "api_elapsed_ms": None}
    traces = []
    if result.candidates:
        for candidate in result.candidates:
            if candidate.inference is not None:
                traces.append(candidate.inference)
            if candidate.geometry and candidate.geometry.inference:
                traces.append(candidate.geometry.inference)
    elif result.inference:
        traces.append(result.inference)
    if result.roi_source == "vlm" and result.roi_proposal:
        traces.append(result.roi_proposal.inference)
    if result.agent:
        traces.extend(event.inference for event in result.agent.events if event.inference is not None)
    if result.mode == "mock":
        return {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": None, "api_elapsed_ms": 0}
    def total(field):
        values = [getattr(trace, field) for trace in traces]
        return sum(values) if values and all(v is not None for v in values) else None
    costs = [trace.estimated_cost_usd for trace in traces]
    cost = str(sum(map(number, costs))) if costs and all(c is not None for c in costs) else None
    return {"api_calls": result.api_calls, "input_tokens": total("input_tokens"), "output_tokens": total("output_tokens"),
            "estimated_cost_usd": cost, "api_elapsed_ms": total("elapsed_ms")}


def run_diagnostics(result, evaluation_candidate_id):
    """Label-free evidence for every candidate, including ones D did not select."""
    if result is None:
        return {"agent_stop_reason": None, "run_reason_codes": [], "candidate_diagnostics": []}
    candidates = result.candidates or [None]
    rows = []
    for candidate in candidates:
        reading = candidate.reading if candidate is not None else result.reading
        candidate_id = candidate.candidate_id if candidate is not None else "c0"
        if candidate is not None:
            image_size = candidate.image_size
        elif result.roi is not None:
            x1, y1, x2, y2 = result.roi
            image_size = (x2 - x1, y2 - y1)
        else:
            image_size = result.source_image.normalized_size
        rows.append({
            "candidate_id": candidate_id,
            "recipe_id": candidate.recipe_id if candidate is not None else None,
            "is_evaluation_candidate": candidate_id == evaluation_candidate_id,
            "status": candidate.status if candidate is not None else result.status,
            "error_code": candidate.error_code if candidate is not None else None,
            "value": reading.value if reading is not None else None,
            "unit": reading.unit if reading is not None else None,
            "value_status": reading.value_status if reading is not None else None,
            "unit_status": reading.unit_status if reading is not None else None,
            "meter_type": reading.meter_type if reading is not None else None,
            "issue_codes": reading.issue_codes if reading is not None else [],
            "analog_trace": reading.analog_trace.model_dump(mode="json") if reading is not None and reading.analog_trace else None,
            "analog_diagnostic": diagnose_analog(reading, image_size),
            "geometry_status": candidate.geometry.status if candidate is not None and candidate.geometry else None,
        })
    return {"agent_stop_reason": result.agent.stop_reason if result.agent is not None else None,
            "run_reason_codes": result.reason_codes, "candidate_diagnostics": rows}


def evaluate_experiment(experiment, labels_path):
    path = Path(experiment).expanduser().resolve()
    experiment_bytes = path.read_bytes()
    document = json.loads(experiment_bytes)
    if document.get("version") != "meter-experiment-v1" or set(document.get("conditions", {})) != set("ABCD"):
        raise ValueError("Expected an A-D experiment manifest")
    if document.get("mode") not in ("mock", "inference"):
        raise ValueError("Experiment mode must be explicit")
    labels_bytes = Path(labels_path).read_bytes()
    labels = load_labels(labels_path, content=labels_bytes)
    conditions = document["conditions"]
    keys = [[row["case_id"] for row in conditions[name]] for name in "ABCD"]
    if not keys[0] or any(len(set(k)) != len(k) or set(k) != set(keys[0]) for k in keys):
        raise ValueError("A-D must contain the same unique cases, including failed/missing results")
    cases = {condition: {row["case_id"]: row for row in conditions[condition]} for condition in "ABCD"}
    rows = {name: [] for name in "ABCD"}
    signatures = set()
    for case_id in keys[0]:
        base = cases["B"][case_id]
        for condition in "ABCD":
            entry = cases[condition][case_id]
            if entry["question_id"] != base["question_id"] or entry["input_sha256"] != base["input_sha256"]:
                raise ValueError("Conditions refer to different inputs or questions")
            if (entry["question"], entry["target_type"]) != (base["question"], base["target_type"]):
                raise ValueError("A-D must read the same display/scale question and target type")
            if condition != "A" and entry["roi"] != base["roi"]:
                raise ValueError("B-D must use exactly the same ROI")
            if condition == "A" and entry["roi"] is not None:
                raise ValueError("Condition A must read the full image")
            label = labels.get(str(entry["question_id"]))
            if label is None:
                raise ValueError("No label for experiment question_id")
            if label.get("image_sha256") and label["image_sha256"] != entry["input_sha256"]:
                raise ValueError("Ground truth belongs to a different image hash")
            result, reading = None, None
            if entry.get("result_path"):
                result_path = asset_in_run(path.parent, entry["result_path"])
                raw = result_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry["result_sha256"]:
                    raise ValueError("Saved experiment result hash mismatch")
                result = RunResult.model_validate_json(raw)
                if result.input_sha256 != entry["input_sha256"] or (list(result.roi) if result.roi else None) != entry["roi"]:
                    raise ValueError("Result input hash/ROI disagrees with manifest")
                if result.mode != document["mode"]:
                    raise ValueError("Mock and inference results must not be mixed")
                if result.dataset is None or str(result.dataset.question_id) != str(entry["question_id"]):
                    raise ValueError("Result question_id disagrees with the experiment")
                if condition in "AB" and (result.candidates or result.agent):
                    raise ValueError("A/B must use the independent, uncorrected read")
                if condition == "C" and (len(result.candidates) != 2 or result.agent is not None
                                         or result.candidates[1].recipe_id != document["fixed_recipe"]):
                    raise ValueError("C must use exactly the predeclared fixed correction")
                if condition == "D" and (result.stage != 4 or result.agent is None):
                    raise ValueError("D must use the recorded agent run")
                if result.mode == "inference" and result.inference is None:
                    raise ValueError("Reader settings are unavailable; cannot verify comparable conditions")
                if (result.question, result.target_type) != (entry["question"], entry["target_type"]):
                    raise ValueError("Result targets a different display/scale question")
                for candidate in result.candidates:
                    if candidate.inference:
                        t = candidate.inference
                        signatures.add((t.requested_model, t.prompt_version, t.image_detail, t.max_output_tokens))
                if not result.candidates and result.inference:
                    t = result.inference
                    signatures.add((t.requested_model, t.prompt_version, t.image_detail, t.max_output_tokens))
                candidate_id = entry.get("evaluation_candidate_id")
                if candidate_id is not None:
                    if not result.candidates and candidate_id == "c0":
                        reading = result.reading
                    else:
                        candidate = next((c for c in result.candidates if c.candidate_id == candidate_id), None)
                        if candidate is None:
                            raise ValueError("Evaluation candidate does not exist")
                        reading = candidate.reading
                expected = "c0" if condition in "AB" else ("c1" if condition == "C" else result.selected_candidate_id)
                if candidate_id != expected:
                    raise ValueError("Evaluation must use the predeclared candidate, never the best label match")
            score = score_reading(reading, label)
            if document["mode"] == "mock":
                for key in ("value_correct", "unit_correct", "both_correct", "absolute_distance_to_interval", "relative_error"):
                    score[key] = None
            roi_skip = entry.get("skip_reason") in ("roi_unresolved", "roi_error")
            if roi_skip and (condition == "A" or result is not None or entry["roi"] is not None):
                raise ValueError("ROI-unresolved skips require an unexecuted B-D condition")
            technical = (result is None and not roi_skip) or bool(entry.get("error")) or (result is not None and technical_failure(result.model_dump(mode="json")))
            totals = trace_totals(result)
            if entry.get("interrupted_attempts", 0):
                for key in totals:
                    totals[key] = None  # A request may have reached the API before interruption.
            rows[condition].append({"case_id": case_id, "question_id": entry["question_id"], "technical_error": technical,
                                    "status": result.status if result else (entry["read_status"] if roi_skip else "error"),
                                    "skip_reason": entry.get("skip_reason"), "missing_result": result is None and not roi_skip,
                                    "evaluation_candidate_id": entry.get("evaluation_candidate_id"),
                                    "abstained": reading is None or reading.value is None or reading.unit is None,
                                    "auto_accept": result.auto_accept if result else False,
                                    "raw_value_text": reading.raw_value_text if reading else None,
                                    "wall_elapsed_ms": entry.get("wall_elapsed_ms"), "interrupted_attempts": entry.get("interrupted_attempts", 0), **totals, **score,
                                    **run_diagnostics(result, entry.get("evaluation_candidate_id")),
                                    "label": label})
    if len(signatures) > 1:
        raise ValueError("A-D Reader model, prompt, image detail and output settings must match")
    simulated = document["mode"] == "mock"
    summaries = {name: summarize(items, simulated) for name, items in rows.items()}
    categories = sorted({row["label"].get("image_type", "unknown") for items in rows.values() for row in items})
    by_category = {category: {name: summarize([row for row in items if row["label"].get("image_type", "unknown") == category], simulated)
                              for name, items in rows.items()} for category in categories}
    return {"version": "meter-evaluation-v1", "adapter": "local-closed-interval-v1",
            "official_evaluator_executed": False, "mode": document["mode"], "accuracy_evaluated": not simulated,
            "experiment_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
            "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
            "fixed_recipe": document["fixed_recipe"], "roi_preparation": document.get("roi_preparation"),
            "summaries": summaries, "by_category": by_category, "conditions": rows}


def summarize(items, simulated=False):
    n = len(items)
    adopted = [row for row in items if row["auto_accept"]]
    extracted = [row for row in items if not row["abstained"]]
    review_candidates = [row for row in extracted if row["status"] == "needs_review"]
    statuses = Counter(row["status"] for row in items)
    def rate(field):
        return None if simulated else sum(bool(row[field]) for row in items) / n
    summary = {
        "n": n, "value_accuracy": rate("value_correct"), "unit_accuracy": rate("unit_correct"),
        "combined_accuracy": rate("both_correct"), "auto_accept_rate": len(adopted) / n,
        "auto_accept_error_rate": sum(not row["both_correct"] for row in adopted) / len(adopted) if adopted else None,
        "statuses": dict(statuses), "unreadable_rate": statuses["unreadable"] / n,
        "needs_review_rate": statuses["needs_review"] / n, "technical_error_rate": sum(row["technical_error"] for row in items) / n,
        "missing_results": sum(row["missing_result"] for row in items),
        "abstention_rate": sum(row["abstained"] for row in items) / n,
        "extracted_candidates": sum(not row["abstained"] for row in items),
        "extracted_candidate_accuracy": sum(row["both_correct"] for row in extracted) / len(extracted) if extracted and not simulated else None,
        "needs_review_candidate_accuracy": sum(row["both_correct"] for row in review_candidates) / len(review_candidates) if review_candidates and not simulated else None,
        "api_calls": sum(row["api_calls"] for row in items) if all(row["api_calls"] is not None for row in items) else None,
    }
    for field in ("input_tokens", "output_tokens", "wall_elapsed_ms", "api_elapsed_ms"):
        values = [row[field] for row in items]
        summary[field] = sum(values) if all(v is not None for v in values) else None
    costs = [row["estimated_cost_usd"] for row in items]
    summary["estimated_cost_usd"] = str(sum(map(number, costs))) if all(c is not None for c in costs) else None
    for field in ("absolute_distance_to_interval", "relative_error"):
        values = [number(row[field]) for row in items if row[field] is not None]
        summary["mean_" + field] = str(sum(values) / len(values)) if values and not simulated else None
    summary["roi_skipped_cases"] = sum(row.get("skip_reason") in ("roi_unresolved", "roi_error") for row in items)
    summary["agent_stop_reasons"] = dict(Counter(row["agent_stop_reason"] for row in items if row.get("agent_stop_reason")))
    # Counts describe candidate observations, not verified causes or case rates.
    candidates = [c for row in items for c in row.get("candidate_diagnostics", []) if c["meter_type"] == "analog"]
    summary["analog_diagnostics"] = {
        "candidate_count": len(candidates),
        "trace_available_count": sum(c["analog_diagnostic"]["trace_available"] for c in candidates),
        "reported_problem_stage_counts": dict(Counter(stage for c in candidates for stage in c["analog_diagnostic"]["reported_problem_stages"])),
        "consistency_finding_counts": dict(Counter(code for c in candidates for code in c["analog_diagnostic"]["consistency_findings"])),
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluations"))
    args = parser.parse_args(argv)
    try:
        report = evaluate_experiment(args.experiment, args.labels)
        root = args.output_dir.expanduser().resolve() / ("evaluation_" + uuid4().hex)
        root.mkdir(parents=True, exist_ok=False)
        path = root / "evaluation.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        from evaluation_csv import write_csv_reports
        write_csv_reports(report, root)
        print(path)
        return 0
    except (OSError, ValueError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
