"""Validated contracts for specification v0.2, stage 1."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, StrictInt, StrictStr,
    field_validator, model_validator,
)

Identifier = Annotated[StrictStr, Field(min_length=1, max_length=256)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
ROI = tuple[StrictInt, StrictInt, StrictInt, StrictInt]
TargetType = Literal["auto", "analog", "digital"]


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
    roi_path: StrictStr = "roi.png"


class RunResult(Contract):
    schema_version: Literal["0.2"] = "0.2"
    stage: Literal[1] = 1
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
    target_type: TargetType
    question: StrictStr | None
    dataset: DatasetReference | None
    acceptance_policy: dict[str, JsonValue] | None
    mode: Literal["mock", "inference"]
    reader_id: StrictStr
    status: Literal["needs_review", "error"]
    reading: ReaderOutput | None
    auto_accept: Literal[False] = False
    effective_value: None = None
    effective_unit: None = None
    review_status: Literal["unreviewed"] = "unreviewed"
    reason_codes: list[StrictStr]
    next_action: Literal["human_review"] = "human_review"
    api_calls: Annotated[StrictInt, Field(ge=0)] | None
    error: StrictStr | None = None
