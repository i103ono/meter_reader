"""Shared CLI/evaluation distinction between technical failure and review stops."""
REVIEW_STOP_CODES = {"observation_api_call_limit", "deadline_exceeded", "cost_budget_exceeded", "retry_budget_exceeded",
                     "geometry_ambiguous", "geometry_not_found"}


def technical_failure(result):
    if result["status"] == "error":
        return True
    ignored = REVIEW_STOP_CODES if result["stage"] == 4 else set()
    return any(c["status"] == "error" and c.get("error_code") not in ignored for c in result.get("candidates", [])) or any(
        event["status"] == "error" and event.get("error_code") not in REVIEW_STOP_CODES
        for event in (result.get("agent") or {}).get("events", [])
    )
