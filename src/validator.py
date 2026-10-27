"""Conservative candidate comparisons. Never vote, estimate a value or accept it."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from itertools import combinations
import re

from schemas import ComparisonPolicy, ReadingComparison

AMBIGUOUS = {"ambiguous_target", "ambiguous_scale", "multiple_meters", "no_meter"}


def comparison_policy(request):
    policy = (request.acceptance_policy or {}).get("comparison")
    return ComparisonPolicy.model_validate(policy) if policy is not None else None


def raw_mismatch(reading):
    if reading.meter_type != "digital" or reading.value is None or reading.raw_value_text is None:
        return False
    raw = reading.raw_value_text.strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw):
        return True  # Unparsed display text is not interchangeable with its number.
    try:
        return Decimal(raw) != Decimal(str(reading.value))
    except InvalidOperation:
        return True


def eligible(candidate):
    reading = candidate.reading
    return bool(candidate.status != "error" and reading is not None
                and reading.value_status == "readable" and reading.unit_status == "readable"
                and reading.value is not None and reading.unit is not None
                and not AMBIGUOUS.intersection(reading.issue_codes) and not raw_mismatch(reading)
                and not scale_unit_conflict(reading))


def scale_unit_conflict(reading):
    return reading.unit is not None and reading.scale.unit is not None and reading.unit != reading.scale.unit


def compare_candidates(candidates, policy: ComparisonPolicy | None):
    comparisons = []
    for left, right in combinations(candidates, 2):
        a, b = left.reading, right.reading
        if a is None or b is None:
            continue
        conflicts = []
        if a.unit is not None and b.unit is not None and a.unit != b.unit:
            conflicts.append("unit_conflict")
        if a.meter_type != "unknown" and b.meter_type != "unknown" and a.meter_type != b.meter_type:
            conflicts.append("meter_type_conflict")
        if a.value is not None and b.value is not None:
            tolerance = Decimal(0)
            if (policy is not None and a.meter_type == b.meter_type == "analog"
                    and a.unit == b.unit == policy.unit):
                tolerance = policy.analog_absolute_tolerance
            if abs(Decimal(str(a.value)) - Decimal(str(b.value))) > tolerance:
                conflicts.append("value_conflict")
        for field in ("min_value", "max_value", "minor_division", "unit"):
            va, vb = getattr(a.scale, field), getattr(b.scale, field)
            if va is not None and vb is not None and va != vb:
                conflicts.append("scale_conflict")
                break
        if AMBIGUOUS.intersection(a.issue_codes) or AMBIGUOUS.intersection(b.issue_codes):
            conflicts.append("target_or_scale_ambiguous")
        if raw_mismatch(a) or raw_mismatch(b):
            conflicts.append("raw_value_mismatch")
        comparisons.append(ReadingComparison(candidate_ids=(left.candidate_id, right.candidate_id), conflicts=conflicts))
    return comparisons
