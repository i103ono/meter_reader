"""Readable evaluation tables; retain JSON as the canonical detailed report."""
import csv
import json


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
              "truth_alternatives", "api_calls"]
    numeric = {"predicted_value", "value_correct", "unit_correct", "both_correct", "abstained", "technical_error", "api_calls"}
    with (root / "cases.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, rows in report["conditions"].items():
            for row in rows:
                data = {**row, "condition": condition, "category": row["label"].get("image_type", "unknown"),
                        "truth_alternatives": json.dumps(row["truth_alternatives"], ensure_ascii=False)}
                writer.writerow({key: data.get(key) if key in numeric else safe_text(data.get(key)) for key in fields})
