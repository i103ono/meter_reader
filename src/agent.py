"""At most two planned corrections under the observation's existing budget."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from planner import ComparisonInput, PlannerInput, validate_decision
from reader import ReaderError
from schemas import AgentEvent, AgentTrace, ComparisonOutput, PlannerOutput
from validator import compare_candidates, comparison_policy, eligible, raw_mismatch, scale_unit_conflict, AMBIGUOUS

BUDGET_CODES = {"observation_api_call_limit", "deadline_exceeded", "cost_budget_exceeded", "retry_budget_exceeded"}


@dataclass
class AgentOutcome:
    candidates: list
    trace: AgentTrace
    selected_candidate_id: str | None
    status: str
    reasons: list[str]
    next_action: str
    api_calls: int | None
    error: str | None


def _event(client, callback, role, turn, *, candidate_id=None):
    before = getattr(client, "last_trace", None)
    result, error, code = None, None, None
    try:
        check = getattr(client, "check_observation_budget", None)
        if check is not None:
            check()
        result = callback()
    except ReaderError as exc:
        code, error = exc.code, str(exc)
    except ValidationError:
        code, error = "invalid_" + role, "Role returned an invalid structured output."
    trace = getattr(client, "last_trace", None)
    if trace is before:
        trace = None
    calls = len(trace.attempts) if trace is not None else (0 if code or client.mode == "mock" else None)
    return AgentEvent(
        role=role, turn=turn, candidate_id=candidate_id,
        status="skipped" if code in BUDGET_CODES and not calls else ("error" if code else "completed"),
        decision=result if role == "planner" else None,
        comparison=result if role == "comparison" else None,
        inference=trace, api_calls=calls, error_code=code, error=error,
        reason_codes=[code] if code else [],
    )


def run_agent(request, reader, planner, comparator, run_dir, baseline, executor):
    policy = comparison_policy(request)
    candidates = [executor.baseline(baseline)]
    events, stop, selected = [], "candidate_limit", None
    recapture = False
    if baseline["status"] == "error":
        stop = "baseline_error"
        events.append(AgentEvent(role="planner", turn=1, status="skipped", reason_codes=[stop]))
    else:
        for turn in (1, 2):
            provider = getattr(reader, "provider", None)
            remaining = (max(0, provider.config.max_observation_api_calls - provider.observation_api_calls)
                         if provider is not None else None)
            plan_input = PlannerInput(
                image_path=run_dir / "roi.png", target_type=request.target_type, question=request.question,
                quality=candidates[0].quality, candidates=tuple(candidates), remaining_api_calls=remaining,
            )
            def plan():
                decision = PlannerOutput.model_validate(planner.plan(plan_input))
                return validate_decision(decision, executor.tried)
            event = _event(planner, plan, "planner", turn)
            events.append(event)
            executor.verify()
            if event.status != "completed":
                stop = event.error_code
                break
            decision = event.decision
            if decision.action != "apply":
                stop = "planner_" + decision.action
                recapture = decision.action == "recommend_recapture"
                break
            candidate = executor.apply(decision.recipe_id, turn)
            candidates.append(candidate)
            if candidate.status == "error":
                stop = candidate.error_code
                events.append(AgentEvent(
                    role="correction", turn=turn, candidate_id=candidate.candidate_id,
                    status="skipped" if stop in BUDGET_CODES else "error", error_code=stop,
                    error=candidate.error, reason_codes=[stop],
                ))
                break
            events.append(AgentEvent(role="correction", turn=turn, candidate_id=candidate.candidate_id,
                                     status="completed"))
            if any(row.conflicts for row in compare_candidates(candidates, policy)):
                stop = "candidate_conflict"
                break
            if any(eligible(c) for c in candidates[1:]):
                stop = "readable_candidate_available"
                break
    comparisons = compare_candidates(candidates, policy)
    conflicts = sorted({code for row in comparisons for code in row.conflicts})
    readable = [c for c in candidates if eligible(c)]
    reasons = ["acceptance_policy_not_calibrated"] if reader.mode == "inference" else ["mock_result"]
    reasons.extend(conflicts)
    # Baseline preference is only a display selection. It cannot bypass conflicts.
    baseline_safe = eligible(candidates[0]) and not conflicts
    corrected = next((c for c in readable if c.candidate_id != "c0"), None)
    if baseline_safe and not recapture:
        selected = "c0"
        events.append(AgentEvent(role="comparison", turn=1, status="skipped",
                                 reason_codes=["original_candidate_preferred"]))
    elif corrected is not None and not conflicts and not recapture and stop not in BUDGET_CODES:
        pair = ComparisonInput(
            original_path=run_dir / "roi.png", corrected_path=run_dir / corrected.image_path,
            original=candidates[0], corrected=corrected, target_type=request.target_type, question=request.question,
        )
        def compare():
            result = ComparisonOutput.model_validate(comparator.compare(pair))
            if result.supported_candidate_id not in (None, "c0", corrected.candidate_id):
                raise ReaderError("Final comparison refers to an unavailable candidate.", code="invalid_comparison")
            return result
        event = _event(comparator, compare, "comparison", 1, candidate_id=corrected.candidate_id)
        events.append(event)
        executor.verify()
        if event.status == "completed":
            result = event.comparison
            if result.contradictions:
                reasons.append("final_comparison_conflict")
            if result.supported_candidate_id in {c.candidate_id for c in readable}:
                selected = result.supported_candidate_id
                reasons.append("final_comparison_support_unverified")
            else:
                reasons.append("final_comparison_not_supported")
        else:
            reasons.extend(["final_comparison_unavailable", event.error_code])
    else:
        reason = "comparison_skipped_conflict" if conflicts else (
            "comparison_skipped_recapture" if recapture else (
                "final_comparison_unavailable" if corrected is not None else "no_readable_corrected_candidate"
            )
        )
        events.append(AgentEvent(role="comparison", turn=1, status="skipped", reason_codes=[reason]))
    executor.verify()
    has_partial = any(c.reading is not None and (c.reading.value is not None or c.reading.unit is not None)
                      for c in candidates)
    valid_observation = any(c.reading is not None for c in candidates)
    technical = any(event.status == "error" for event in events) or any(
        c.status == "error" and c.error_code not in BUDGET_CODES for c in candidates
    )
    blocked = stop in BUDGET_CODES or any(event.error_code in BUDGET_CODES for event in events)
    status = "needs_review" if has_partial or conflicts or technical or blocked or reader.mode == "mock" else "unreadable"
    if not valid_observation and technical:
        status = "error"
    if any(c.reading is not None and AMBIGUOUS.intersection(c.reading.issue_codes) for c in candidates):
        reasons.append("target_or_scale_ambiguous")
        selected = None
        status = "needs_review"
    if any(c.reading is not None and raw_mismatch(c.reading) for c in candidates):
        reasons.append("raw_value_mismatch")
        selected = None
        status = "needs_review"
    if any(c.reading is not None and scale_unit_conflict(c.reading) for c in candidates):
        reasons.append("scale_unit_conflict")
        selected = None
        status = "needs_review"
    if recapture:
        reasons.append("recapture_recommended")
    reasons.append(stop)
    reasons.extend(event.error_code for event in events if event.error_code)
    counts = [c.api_calls for c in candidates] + [e.api_calls for e in events]
    calls = sum(counts) if all(count is not None for count in counts) else None
    # Preserve a useful original observation even if a later role fails.
    error = baseline["error"] if status == "error" else None
    return AgentOutcome(
        candidates=candidates, trace=AgentTrace(planner_id=planner.planner_id, stop_reason=stop,
                                               events=events, comparisons=comparisons, comparison_policy=policy),
        selected_candidate_id=None if recapture else selected, status=status,
        reasons=list(dict.fromkeys(reasons)), next_action="recapture" if recapture else "human_review",
        api_calls=calls, error=error,
    )
