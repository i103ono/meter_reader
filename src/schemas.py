"""Validated contracts for specification v0.2, stages 1 through 4 and ROI proposals."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictInt, StrictStr,
    field_validator, model_validator,
)

Identifier = Annotated[StrictStr, Field(min_length=1, max_length=256)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
ROI = tuple[StrictInt, StrictInt, StrictInt, StrictInt]
ImageSize = tuple[Annotated[StrictInt, Field(gt=0)], Annotated[StrictInt, Field(gt=0)]]
SHA256 = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
TargetType = Literal["auto", "analog", "digital"]
RecipeId = Literal[
    "resize_2x", "gamma_mild", "clahe_mild", "gamma_clahe",
    "resize_gamma", "resize_clahe", "resize_gamma_clahe",
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class DatasetReference(Contract):
    """Provenance only. Evaluator definitions and answers stay in label files."""
    name: StrictStr | None = None
    split: StrictStr | None = None
    revision: StrictStr | None = None
    seed: StrictInt | None = None
    question_id: StrictStr | StrictInt | None = None
    category: StrictStr | None = None
    source: JsonValue = None


class ObservationInput(Contract):
    image_path: Path
    roi: ROI | None = None
    target_type: TargetType = "auto"
    meter_id: Identifier | None = None
    observation_id: Identifier | None = None
    captured_at: AwareDatetime | None = None
    timestamp_source: Literal["user", "exif", "unknown"] = "unknown"
    question: StrictStr | None = None
    dataset: DatasetReference | None = None
    acceptance_policy: dict[str, JsonValue] | None = None

    @model_validator(mode="before")
    @classmethod
    def infer_user_timestamp_source(cls, data):
        if isinstance(data, dict) and "timestamp_source" not in data:
            data = dict(data)
            data["timestamp_source"] = "user" if data.get("captured_at") is not None else "unknown"
        return data

    @field_validator("captured_at", mode="before")
    @classmethod
    def iso_timestamp_only(cls, value):
        if value is not None and not isinstance(value, (str, datetime)):
            raise ValueError("captured_at must be a timezone-aware ISO 8601 timestamp")
        if isinstance(value, str):
            # Pydantic otherwise also accepts numeric strings as Unix timestamps.
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("captured_at must be a timezone-aware ISO 8601 timestamp") from exc
        return value

    @field_validator("meter_id", "observation_id")
    @classmethod
    def nonblank_id(cls, value):
        if value is not None and not value.strip():
            raise ValueError("IDs must not be blank")
        return value

    @model_validator(mode="after")
    def consistent_capture_source(self):
        if (self.captured_at is None) != (self.timestamp_source == "unknown"):
            raise ValueError("captured_at and timestamp_source must agree")
        return self


class MeterObservationInput(ObservationInput):
    """Future history endpoints can require a meter ID without changing the CLI."""
    meter_id: Identifier


class ScaleInfo(Contract):
    min_value: Number | None = None
    max_value: Number | None = None
    minor_division: Number | None = None
    unit: StrictStr | None = None
    status: Literal["readable", "partial", "unreadable", "not_applicable"] = "unreadable"
    source: Literal["image", "manual", "registry", "none"] = "none"

    @field_validator("unit")
    @classmethod
    def nonblank_unit(cls, value):
        if value is not None and not value.strip():
            raise ValueError("unknown scale unit must be null, not an empty string")
        return value

    @model_validator(mode="after")
    def coherent_scale(self):
        known = any(v is not None for v in (
            self.min_value, self.max_value, self.minor_division, self.unit,
        ))
        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ValueError("scale min_value must not exceed max_value")
        if self.minor_division is not None and self.minor_division <= 0:
            raise ValueError("minor_division must be positive")
        if self.status in ("unreadable", "not_applicable") and known:
            raise ValueError("unknown/not-applicable scales must not contain inferred values")
        if self.source == "none" and known:
            raise ValueError("scale values require an explicit source")
        if self.status == "readable" and (self.min_value is None or self.max_value is None):
            raise ValueError("readable scale requires min_value and max_value")
        return self


class ReaderOutput(Contract):
    meter_type: Literal["analog", "digital", "unknown"] = "unknown"
    value: Number | None = None
    raw_value_text: StrictStr | None = None
    unit: StrictStr | None = None
    raw_unit_text: StrictStr | None = None
    value_status: Literal["readable", "uncertain", "unreadable"] = "unreadable"
    unit_status: Literal["readable", "uncertain", "unreadable", "not_present"] = "unreadable"
    value_evidence: StrictStr | None = None
    unit_evidence: StrictStr | None = None
    issue_codes: list[StrictStr] = Field(default_factory=list)
    scale: ScaleInfo = Field(default_factory=ScaleInfo)

    @model_validator(mode="after")
    def coherent_reading(self):
        if self.value_status == "readable" and self.value is None:
            raise ValueError("readable value must not be null")
        if self.value_status == "unreadable" and self.value is not None:
            raise ValueError("unreadable value must be null; use uncertain for candidates")
        if self.unit is not None and not self.unit.strip():
            raise ValueError("unknown unit must be null, not an empty string")
        if self.unit_status == "readable" and self.unit is None:
            raise ValueError("readable unit must not be null")
        if self.unit_status in ("unreadable", "not_present") and self.unit is not None:
            raise ValueError("unreadable/absent unit must be null")
        if self.scale.source not in ("image", "none"):
            raise ValueError("Readers cannot supply manual or registry scale information")
        return self


class TimestampMetadata(Contract):
    exif_datetime_original: StrictStr | None = None
    exif_offset_original: StrictStr | None = None
    assumed_timezone: StrictStr | None = None
    notes: list[StrictStr] = Field(default_factory=list)


class SourceImage(Contract):
    original_path: StrictStr
    stored_path: StrictStr
    format: Literal["JPEG", "PNG"]
    original_size: tuple[StrictInt, StrictInt]
    normalized_size: tuple[StrictInt, StrictInt]
    exif_orientation: StrictInt | None = None
    normalized_path: StrictStr = "normalized.png"
    roi_path: StrictStr | None = "roi.png"


class TokenUsage(Contract):
    input_tokens: Annotated[StrictInt, Field(ge=0)] | None = None
    output_tokens: Annotated[StrictInt, Field(ge=0)] | None = None
    total_tokens: Annotated[StrictInt, Field(ge=0)] | None = None


class InferenceAttempt(Contract):
    number: Annotated[StrictInt, Field(ge=1)]
    status: Literal["completed", "http_error", "network_error", "invalid_response", "refused", "incomplete"]
    http_status: StrictInt | None = None
    request_id: StrictStr | None = None
    response_id: StrictStr | None = None
    reported_model: StrictStr | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    raw_output: StrictStr | None = None
    error_code: StrictStr | None = None


class InferenceTrace(Contract):
    provider: StrictStr
    requested_model: StrictStr
    prompt_version: StrictStr
    image_detail: Literal["low", "high", "auto"]
    image_width: StrictInt
    image_height: StrictInt
    max_output_tokens: StrictInt
    timeout_seconds: Number
    deadline_seconds: Number
    api_call_limit: StrictInt
    observation_api_call_limit: StrictInt | None = None
    attempts: list[InferenceAttempt] = Field(default_factory=list)
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    input_tokens: StrictInt | None = None
    output_tokens: StrictInt | None = None
    estimated_cost_usd: StrictStr | None = None
    cost_basis: Literal["unconfigured", "configured_token_rates"] = "unconfigured"
    pricing_source: StrictStr | None = None
    batch_cost_limit_usd: StrictStr | None = None
    batch_cost_reserved_usd: StrictStr | None = None
    error_code: StrictStr | None = None
    unit_normalization_version: Literal["unit-aliases-v1"] = "unit-aliases-v1"


class ROIProposalOutput(Contract):
    """Locator coordinates refer to the exact image sent to the model."""
    status: Literal["located", "ambiguous", "not_found"]
    roi: ROI | None
    target_description: StrictStr | None
    reason: StrictStr | None
    issue_codes: list[StrictStr]

    @model_validator(mode="after")
    def coherent_proposal(self):
        if (self.status == "located") != (self.roi is not None):
            raise ValueError("only located proposals may have a non-null ROI")
        if self.status == "located" and not (self.target_description or "").strip():
            raise ValueError("located proposals require a target description")
        if self.status == "located" and {"ambiguous_target", "ambiguous_scale"}.intersection(self.issue_codes):
            raise ValueError("unresolved target/scale ambiguity cannot be located")
        if self.status != "located" and not (self.reason or "").strip():
            raise ValueError("unresolved proposals require a reason")
        return self


class ROIProposal(Contract):
    schema_version: Literal["roi-proposal-v1"] = "roi-proposal-v1"
    observation_id: Identifier
    input_sha256: SHA256
    normalized_pixel_sha256: SHA256
    normalized_size: ImageSize
    model_image_size: ImageSize
    model_image_sha256: SHA256
    coordinate_system: Literal["exif_normalized_xyxy_exclusive"] = "exif_normalized_xyxy_exclusive"
    mapping: Literal["floor_left_top_ceil_right_bottom-v1"] = "floor_left_top_ceil_right_bottom-v1"
    resize: Literal["longest-side-1600-lanczos-v1"] = "longest-side-1600-lanczos-v1"
    target_type: TargetType
    question: StrictStr | None
    status: Literal["located", "ambiguous", "not_found", "error"]
    response: ROIProposalOutput | None
    proposed_roi: ROI | None
    inference: InferenceTrace
    error_code: StrictStr | None = None
    error: StrictStr | None = None

    @model_validator(mode="after")
    def coherent_record(self):
        if self.status == "error":
            if (self.proposed_roi is not None or not self.error_code or self.response is not None
                    or self.inference.error_code != self.error_code):
                raise ValueError("failed proposals require an error code and no ROI")
        elif (self.response is None or self.response.status != self.status
              or self.error_code is not None or self.error is not None or self.inference.error_code is not None):
            raise ValueError("proposal status and response must agree")
        if (self.status == "located") != (self.proposed_roi is not None):
            raise ValueError("only located records may have a proposed ROI")
        return self


class ProcessingStep(Contract):
    operation: Literal["gamma", "clahe", "resize"]
    parameters: dict[str, JsonValue]


class QualityMetrics(Contract):
    width: StrictInt
    height: StrictInt
    luminance_p5: Number
    luminance_p50: Number
    luminance_p95: Number
    luminance_spread: Number
    dark_fraction: Number
    saturated_fraction: Number
    local_mean_spread: Number
    parameters: dict[str, JsonValue]


class CandidateResult(Contract):
    candidate_id: Identifier
    parent_candidate_id: Literal["c0"] | None
    recipe_id: RecipeId | None
    processing_version: Literal["fixed-recipes-v1"] = "fixed-recipes-v1"
    steps: list[ProcessingStep] = Field(default_factory=list)
    runtime: dict[str, StrictStr]
    input_roi_sha256: StrictStr
    image_path: StrictStr | None
    image_sha256: StrictStr | None
    pixel_sha256: StrictStr | None
    image_size: tuple[StrictInt, StrictInt] | None
    image_mode: StrictStr | None
    quality: QualityMetrics | None
    status: Literal["needs_review", "unreadable", "error"]
    reading: ReaderOutput | None
    inference: InferenceTrace | None
    api_calls: Annotated[StrictInt, Field(ge=0)] | None
    error_code: StrictStr | None = None
    error: StrictStr | None = None


class ComparisonPolicy(Contract):
    """Comparison tolerance only. This never enables automatic acceptance."""
    analog_absolute_tolerance: Decimal = Field(ge=0, allow_inf_nan=False)
    unit: StrictStr = Field(min_length=1)

    @field_validator("unit")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("comparison unit must not be blank")
        return value


class PlannerOutput(Contract):
    action: Literal["apply", "stop", "recommend_recapture"]
    recipe_id: RecipeId | None
    reason: StrictStr = Field(min_length=1)
    reason_codes: list[StrictStr]

    @model_validator(mode="after")
    def coherent_action(self):
        if (self.action == "apply") != (self.recipe_id is not None):
            raise ValueError("only apply requires a recipe ID")
        if not self.reason.strip():
            raise ValueError("planner reason must not be blank")
        return self


class ComparisonOutput(Contract):
    supported_candidate_id: Identifier | None
    readable: StrictBool
    contradictions: list[StrictStr]
    reason: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def coherent_support(self):
        if self.supported_candidate_id is not None and (not self.readable or self.contradictions):
            raise ValueError("support requires readability and no contradictions")
        if not self.reason.strip():
            raise ValueError("comparison reason must not be blank")
        return self


class AgentEvent(Contract):
    role: Literal["planner", "correction", "comparison"]
    turn: Annotated[StrictInt, Field(ge=1, le=2)]
    candidate_id: Identifier | None = None
    status: Literal["completed", "error", "skipped"]
    decision: PlannerOutput | None = None
    comparison: ComparisonOutput | None = None
    inference: InferenceTrace | None = None
    api_calls: Annotated[StrictInt, Field(ge=0)] | None = 0
    reason_codes: list[StrictStr] = Field(default_factory=list)
    error_code: StrictStr | None = None
    error: StrictStr | None = None


class ReadingComparison(Contract):
    candidate_ids: tuple[Identifier, Identifier]
    conflicts: list[StrictStr]


class AgentTrace(Contract):
    version: Literal["meter-agent-v1"] = "meter-agent-v1"
    planner_id: StrictStr
    stop_reason: StrictStr
    events: list[AgentEvent]
    comparisons: list[ReadingComparison]
    comparison_policy: ComparisonPolicy | None = None


class RunResult(Contract):
    schema_version: Literal["0.2"] = "0.2"
    stage: Literal[1, 2, 3, 4] = 1
    run_id: Identifier
    observation_id: Identifier
    meter_id: Identifier | None
    captured_at: AwareDatetime | None
    captured_timezone: StrictStr | None
    timestamp_source: Literal["user", "exif", "unknown"]
    timestamp_metadata: TimestampMetadata
    processed_at: AwareDatetime
    input_sha256: StrictStr
    source_image: SourceImage
    roi: ROI | None
    roi_coordinate_system: Literal["exif_normalized_xyxy_exclusive"] = "exif_normalized_xyxy_exclusive"
    roi_source: Literal["full_image", "manual", "vlm", "saved_vlm", "manual_override"] | None = None
    roi_proposal: ROIProposal | None = None
    roi_proposal_reference: StrictStr | None = None
    roi_preview_path: StrictStr | None = None
    read_requested: StrictBool = True
    target_type: TargetType
    question: StrictStr | None
    dataset: DatasetReference | None
    acceptance_policy: dict[str, JsonValue] | None
    mode: Literal["mock", "inference"]
    reader_id: StrictStr
    status: Literal["needs_review", "unreadable", "error"]
    reading: ReaderOutput | None
    auto_accept: Literal[False] = False
    effective_value: None = None
    effective_unit: None = None
    review_status: Literal["unreviewed"] = "unreviewed"
    reason_codes: list[StrictStr]
    next_action: Literal["human_review", "recapture"] = "human_review"
    api_calls: Annotated[StrictInt, Field(ge=0)] | None
    error: StrictStr | None = None
    inference: InferenceTrace | None = None
    baseline_candidate_id: Literal["c0"] | None = None
    selected_candidate_id: Identifier | None = None
    correction_candidates_generated: StrictInt = Field(default=0, ge=0)
    candidates: list[CandidateResult] = Field(default_factory=list)
    agent: AgentTrace | None = None
