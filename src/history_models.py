"""Human history contracts, separate from immutable model predictions."""
from decimal import Decimal
from typing import Literal

from pydantic import Field, StrictStr, model_validator

from schemas import Contract, Identifier, ScaleInfo


class MeterRecord(Contract):
    meter_id: Identifier
    name: StrictStr = Field(min_length=1)
    meter_type: Literal["analog", "digital", "unknown"] = "unknown"
    location: StrictStr | None = None
    registered_unit: StrictStr | None = None
    registered_scale: ScaleInfo | None = None
    series_version: Identifier = "1"

    @model_validator(mode="after")
    def registry_source(self):
        if not self.name.strip() or not self.meter_id.strip() or not self.series_version.strip():
            raise ValueError("meter name must not be blank")
        if self.registered_scale is not None and self.registered_scale.source not in ("registry", "none"):
            raise ValueError("registered scales must identify their registry source")
        return self


class ReviewInput(Contract):
    run_id: Identifier
    action: Literal["confirm", "correct", "reject"]
    candidate_id: Identifier | None = None
    value: Decimal | None = Field(default=None, allow_inf_nan=False)
    unit: StrictStr | None = None
    scale: ScaleInfo | None = None
    reason: StrictStr = Field(min_length=1)
    reviewer: Identifier
    review_id: Identifier | None = None

    @model_validator(mode="after")
    def coherent_review(self):
        if not self.reason.strip() or not self.reviewer.strip():
            raise ValueError("reviewer and reason must not be blank")
        if self.action == "correct":
            if self.value is None or self.unit is None or not self.unit.strip():
                raise ValueError("corrections require an explicit finite value and unit")
            if self.scale is not None and self.scale.source not in ("manual", "none"):
                raise ValueError("corrected scales must identify their manual source")
        elif self.value is not None or self.unit is not None or self.scale is not None:
            raise ValueError("use correct to supply new values or scale information")
        return self
