"""Readable evaluation tables; retain JSON as the canonical detailed report."""
import csv
import json

from analog_diagnostics import STAGES


def safe_text(value):
    text = "" if value is None else str(value)
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text


def write_csv_reports(report, root):
    metrics = ["n", "value_accuracy", "unit_accuracy", "combined_accuracy", "abstention_rate",
               "extracted_candidate_accuracy", "needs_review_candidate_accuracy", "technical_error_rate",
               "roi_skipped_cases", "api_calls", "wall_elapsed_ms", "estimated_cost_usd"]
    with (root / "summary.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "condition", *metrics])
        writer.writeheader()
        for category, conditions in [("ALL", report["summaries"]), *report.get("by_category", {}).items()]:
            for name, summary in conditions.items():
                writer.writerow({"category": safe_text(category), "condition": name, **{k: summary[k] for k in metrics}})
    fields = ["condition", "question_id", "category", "status", "predicted_value", "predicted_unit", "raw_value_text",
              "value_correct", "unit_correct", "both_correct", "abstained", "skip_reason", "technical_error",
              "truth_alternatives", "api_calls", "agent_stop_reason", "run_reason_codes"]
    numeric = {"predicted_value", "value_correct", "unit_correct", "both_correct", "abstained", "technical_error", "api_calls"}
    with (root / "cases.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, rows in report["conditions"].items():
            for row in rows:
                data = {**row, "condition": condition, "category": row["label"].get("image_type", "unknown"),
                        "truth_alternatives": json.dumps(row["truth_alternatives"], ensure_ascii=False),
                        "run_reason_codes": json.dumps(row.get("run_reason_codes", []), ensure_ascii=False)}
                writer.writerow({key: data.get(key) if key in numeric else safe_text(data.get(key)) for key in fields})
    fields = ["condition", "question_id", "category", "candidate_id", "recipe_id", "is_evaluation_candidate",
              "status", "error_code", "meter_type", "value", "unit", "value_status", "unit_status", "agent_stop_reason",
              "geometry_status", "trace_available", "first_reported_problem_stage", "reported_problem_stages",
              "consistency_findings", "interpolated_value", "issue_codes", "stage_observations", "analog_trace"]
    fields += [stage + "_stage_status" for stage in STAGES] + [stage + "_stage_evidence" for stage in STAGES]
    numeric = {"value", "is_evaluation_candidate", "trace_available"}
    with (root / "diagnostics.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, rows in report["conditions"].items():
            for row in rows:
                for candidate in row.get("candidate_diagnostics", []):
                    diagnostic = candidate["analog_diagnostic"]
                    data = {**candidate, **diagnostic, "condition": condition, "question_id": row["question_id"],
                            "category": row["label"].get("image_type", "unknown"), "agent_stop_reason": row.get("agent_stop_reason")}
                    for stage in STAGES:
                        observation = diagnostic["stage_observations"].get(stage, {})
                        data[stage + "_stage_status"] = observation.get("status")
                        data[stage + "_stage_evidence"] = observation.get("evidence")
                    for key in ("reported_problem_stages", "consistency_findings", "issue_codes", "stage_observations", "analog_trace"):
                        data[key] = json.dumps(data.get(key), ensure_ascii=False)
                    writer.writerow({key: data.get(key) if key in numeric else safe_text(data.get(key)) for key in fields})
