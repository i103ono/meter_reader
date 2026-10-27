"""Stage observations and deterministic consistency checks, never label attribution.

These checks cannot prove that VLM landmarks or scale labels match the pixels.
They retain the proposed value; contradictions downgrade it rather than repair it.
"""
from __future__ import annotations

from decimal import Decimal
import math
import re

STAGES = ("target", "geometry", "scale", "needle", "interpolation")
ALIASES = {"m³": "m^3", "m³/h": "m^3/h", "kW·h": "kWh", "kW h": "kWh"}


def normalized_unit(value):
    return ALIASES.get(value, value)


def diagnose_analog(reading, image_size=None):
    result = {"basis": "model_observations_and_local_checks_v1", "trace_available": False,
              "stage_observations": {}, "reported_problem_stages": [],
              "consistency_findings": [], "interpolated_value": None,
              "first_reported_problem_stage": None}
    if reading is None or reading.meter_type != "analog":
        return result
    trace = reading.analog_trace
    if trace is None:
        result["consistency_findings"] = ["analog_trace_missing"]
        return result
    result["trace_available"] = True
    result["stage_observations"] = {stage: getattr(trace, stage).model_dump(mode="json") for stage in STAGES}
    problems = [stage for stage in STAGES if getattr(trace, stage).status != "visible"]
    result["reported_problem_stages"] = problems
    result["first_reported_problem_stage"] = next(iter(problems), None)
    findings = result["consistency_findings"]
    if trace.center is not None and trace.tip is not None:
        vector = (trace.tip[0] - trace.center[0], trace.tip[1] - trace.center[1])
        if math.hypot(*vector) < 1e-6:
            findings.append("analog_degenerate_pointer")
        if trace.counterweight is not None and image_size is not None:
            width, height = image_size
            vector = (vector[0] * width, vector[1] * height)
            back = ((trace.counterweight[0] - trace.center[0]) * width,
                    (trace.counterweight[1] - trace.center[1]) * height)
            # The two identified ends cannot point along the same ray.
            norm = math.hypot(*vector) * math.hypot(*back)
            if norm > 1e-12 and sum(a * b for a, b in zip(vector, back)) / norm > 0.9:
                findings.append("analog_pointer_counterweight_conflict")
    units = {normalized_unit(u) for u in (reading.unit, reading.scale.unit, trace.scale_unit) if u is not None}
    if len(units) > 1:
        findings.append("analog_scale_unit_conflict")
    if trace.anchors is not None:
        for anchor in trace.anchors:
            raw = anchor.raw_text.strip().replace("−", "-")
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw) and Decimal(raw) != Decimal(str(anchor.value)):
                findings.append("analog_anchor_text_value_conflict")
                break
    if trace.interpolation.status == "visible" and trace.interpolation_mode == "linear":
        if trace.anchors is None or trace.fraction is None:
            findings.append("analog_interpolation_evidence_missing")
        else:
            a, b = trace.anchors
            lower, upper, fraction = Decimal(str(a.value)), Decimal(str(b.value)), Decimal(str(trace.fraction))
            if lower == upper:
                findings.append("analog_degenerate_anchors")
            else:
                computed = lower + fraction * (upper - lower)
                result["interpolated_value"] = str(computed)
                # Rounding to half of the observed minor division is allowed;
                # absent a division this only tolerates floating-point noise.
                tolerance = abs(upper - lower) * Decimal("0.000001")
                if reading.scale.minor_division is not None and len(units) == 1:
                    tolerance = max(tolerance, Decimal(str(reading.scale.minor_division)) / 2)
                if reading.value is not None and abs(Decimal(str(reading.value)) - computed) > tolerance:
                    findings.append("analog_interpolation_value_conflict")
                # Only check circular angle ordering when the model explicitly
                # reports front-facing geometry. Never use oblique image angles
                # as physical angles or infer perspective from an ellipse.
                if (image_size is not None and trace.view == "front_facing" and trace.geometry.status == "visible"
                        and trace.center is not None and trace.tip is not None):
                    points = (a.tick_point, b.tick_point, trace.tip)
                    width, height = image_size
                    vectors = [((p[0] - trace.center[0]) * width, (p[1] - trace.center[1]) * height) for p in points]
                    if any(math.hypot(*v) < 1e-6 for v in vectors):
                        findings.append("analog_degenerate_anchor_geometry")
                    else:
                        angles = [math.atan2(y, x) for x, y in vectors]
                        wrap = lambda angle: (angle + math.pi) % (2 * math.pi) - math.pi
                        arc = wrap(angles[1] - angles[0])
                        offset = wrap(angles[2] - angles[0])
                        # Adjacent bracketing labels must define a local arc.
                        # A reversed needle lies outside it. Allow 2 degrees
                        # of coordinate noise at either boundary.
                        if 1e-6 < abs(arc) <= math.pi / 2:
                            position = offset / arc
                            margin = math.radians(2) / abs(arc)
                            if not -margin <= position <= 1 + margin:
                                findings.append("analog_tip_outside_anchor_arc")
    return result


def guard_analog_reading(reading, image_size=None):
    if reading.meter_type != "analog":
        return reading
    diagnostic = diagnose_analog(reading, image_size)
    issues = list(reading.issue_codes)
    for code in diagnostic["consistency_findings"]:
        if code not in issues:
            issues.append(code)
    # Geometry can be uncertain in an oblique view even when local tick-count
    # interpolation is supported. Do not reject every oblique image by default.
    blocking_stages = set(diagnostic["reported_problem_stages"]) - {"geometry"}
    for stage in sorted(blocking_stages):
        code = "analog_" + stage + "_unresolved"
        if code not in issues:
            issues.append(code)
    changes = {"issue_codes": issues}
    if diagnostic["consistency_findings"] or blocking_stages:
        if reading.value_status == "readable":
            changes["value_status"] = "uncertain"
    if "analog_scale_unit_conflict" in issues and reading.unit_status == "readable":
        changes["unit_status"] = "uncertain"
    return reading.model_copy(update=changes)
