"""OpenAI Responses transport, with bounded requests and no credential logging."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
import http.client
import json
import os
import socket
import time
from typing import Callable, Literal
import urllib.error
import urllib.request

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from reader import ReaderError
from schemas import Contract, InferenceAttempt, TokenUsage


class TokenPricing(Contract):
    """User-verified rates and the model's *full context* input-token upper bound."""
    model: StrictStr
    input_usd_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    output_usd_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    max_input_tokens_per_call: StrictInt = Field(gt=0)
    source: StrictStr = Field(min_length=1)

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        return (
            self.input_usd_per_million * input_tokens
            + self.output_usd_per_million * output_tokens
        ) / Decimal(1_000_000)


class ProviderConfig(Contract):
    model: StrictStr = Field(min_length=1)
    api_key_env: StrictStr = Field(default="METER_READER_API_KEY", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    image_detail: Literal["low", "high", "auto"] = "high"
    timeout_seconds: float = Field(default=30, gt=0, le=30, allow_inf_nan=False)
    deadline_seconds: float = Field(default=120, gt=0, le=120, allow_inf_nan=False)
    max_output_tokens: StrictInt = Field(default=2048, ge=128, le=8192)
    max_api_calls: StrictInt = Field(default=2, ge=1, le=2)
    max_observation_api_calls: StrictInt = Field(default=6, ge=1, le=6)
    pricing: TokenPricing | None = None
    max_cost_usd: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("model", "api_key_env")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("model and api_key_env must not be blank")
        return value

    @model_validator(mode="after")
    def coherent_config(self):
        if self.pricing is not None and self.pricing.model != self.model:
            raise ValueError("pricing must describe the requested model")
        if self.max_cost_usd is not None and self.pricing is None:
            raise ValueError("a USD limit requires user-verified token pricing and context bound")
        return self


@dataclass(frozen=True)
class HTTPResult:
    body: bytes
    status: int
    request_id: str | None = None


class TransportFailure(Exception):
    def __init__(self, code, *, status=None, retry_after=None):
        self.code, self.status, self.retry_after = code, status, retry_after


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def http_post(request: urllib.request.Request, timeout: float) -> HTTPResult:
    """TLS verification uses the machine's normal trust configuration."""
    opener = urllib.request.build_opener(NoRedirects())
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(2_097_153)
            if len(body) > 2_097_152:
                raise TransportFailure("response_too_large")
            return HTTPResult(body, response.status, response.headers.get("x-request-id"))
    except urllib.error.HTTPError as exc:
        # Error bodies can echo credentials. Never put them in results or exceptions.
        retry_after = None
        header = exc.headers.get("Retry-After") if exc.headers is not None else None
        if header:
            try:
                retry_after = float(header)
            except ValueError:
                try:
                    retry_after = max(0, (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    retry_after = 6.0  # Unknown server wait: do not retry early.
            if not 0 <= retry_after <= 5:
                retry_after = 6.0
        code = "authentication_error" if exc.code == 401 else (
            "permission_error" if exc.code == 403 else (
                "rate_limit_error" if exc.code == 429 else "http_error"
            )
        )
        raise TransportFailure(code, status=exc.code, retry_after=retry_after) from None
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException):
        raise TransportFailure("network_error") from None


class CostBudget:
    """Shared across a CLI batch. Reserve before each outbound attempt."""
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.reserved = Decimal(0)

    def check(self) -> Decimal | None:
        prices = self.config.pricing
        if prices is None:
            return None
        amount = prices.cost(prices.max_input_tokens_per_call, self.config.max_output_tokens)
        if self.config.max_cost_usd is not None and self.reserved + amount > self.config.max_cost_usd:
            raise ReaderError("Configured batch cost limit reached before API request.", code="cost_budget_exceeded")
        return amount

    def reserve(self) -> Decimal | None:
        amount = self.check()
        if amount is not None:
            self.reserved += amount
        return amount

    def settle(self, reserved: Decimal | None, usage: TokenUsage) -> None:
        prices = self.config.pricing
        if prices is None or reserved is None or usage.input_tokens is None or usage.output_tokens is None:
            return
        # Incomplete/failed attempts keep their reservation: unknown cost is not zero.
        actual = prices.cost(usage.input_tokens, usage.output_tokens)
        self.reserved += actual - reserved


class OpenAIProvider:
    name = "openai"
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(
        self, config: ProviderConfig, *, transport: Callable | None = None,
        clock: Callable = time.monotonic, sleep: Callable = time.sleep,
    ):
        self.config = config
        self.transport = transport or http_post
        self.clock, self.sleep = clock, sleep
        self.budget = CostBudget(config)
        self.observation_deadline: float | None = None
        self.observation_api_calls = 0

    def begin_observation(self) -> None:
        self.observation_deadline = self.clock() + self.config.deadline_seconds
        self.observation_api_calls = 0

    def check_observation_budget(self) -> None:
        if self.observation_deadline is not None and self.clock() >= self.observation_deadline:
            raise ReaderError("Observation deadline reached.", code="deadline_exceeded")
        if self.observation_api_calls >= self.config.max_observation_api_calls:
            raise ReaderError("Observation API call limit reached.", code="observation_api_call_limit")
        self.budget.check()

    def credential(self) -> str:
        value = os.environ.get(self.config.api_key_env)
        if not value:
            raise ReaderError(
                f"Required credential variable {self.config.api_key_env} is not set.",
                code="credential_missing",
            )
        return value

    def generate(self, payload: dict, attempts: list[InferenceAttempt], deadline: float) -> str:
        if self.observation_deadline is not None:
            deadline = min(deadline, self.observation_deadline)
        key = self.credential()
        for number in range(1, self.config.max_api_calls + 1):
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise ReaderError("Reader deadline reached before API request.", code="deadline_exceeded")
            self.check_observation_budget()
            reserved = self.budget.reserve()
            request = urllib.request.Request(
                self.endpoint, data=json.dumps(payload, allow_nan=False).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                method="POST",
            )
            started = self.clock()
            self.observation_api_calls += 1
            try:
                response = self.transport(request, min(self.config.timeout_seconds, remaining))
            except TransportFailure as exc:
                attempts.append(InferenceAttempt(
                    number=number,
                    status="http_error" if exc.status is not None else "network_error",
                    http_status=exc.status, elapsed_ms=max(0, (self.clock() - started) * 1000),
                    error_code=exc.code,
                ))
                transient = exc.status in (429, 500, 502, 503, 504) or exc.code == "network_error"
                wait = exc.retry_after if exc.retry_after is not None else 0.5
                if not transient or number == self.config.max_api_calls:
                    raise ReaderError("VLM request failed; see the recorded error code and HTTP status.", code=exc.code) from None
                if wait > 5 or self.clock() + wait >= deadline:
                    raise ReaderError("Retry wait exceeds the remaining reader budget.", code="retry_budget_exceeded") from None
                self.sleep(wait)
                continue
            elapsed = max(0, (self.clock() - started) * 1000)
            try:
                document = json.loads(response.body)
                if not isinstance(document, dict):
                    raise ValueError("response must be an object")
                usage_document = document.get("usage") or {}
                if not isinstance(usage_document, dict):
                    raise ValueError("usage must be an object")
                usage = TokenUsage.model_validate({
                    key: usage_document.get(key) for key in TokenUsage.model_fields
                })
                for field in ("id", "model"):
                    if document.get(field) is not None and not isinstance(document[field], str):
                        raise ValueError("invalid response identity")
                if not isinstance(document.get("output"), list):
                    raise ValueError("output must be an array")
                for item in document["output"]:
                    if not isinstance(item, dict):
                        raise ValueError("output items must be objects")
                    if item.get("type") == "message" and not isinstance(item.get("content"), list):
                        raise ValueError("message content must be an array")
            except (ValueError, TypeError, UnicodeDecodeError):
                attempts.append(InferenceAttempt(
                    number=number, status="invalid_response", http_status=response.status,
                    elapsed_ms=elapsed, error_code="invalid_response",
                ))
                raise ReaderError("VLM returned invalid response metadata.", code="invalid_response") from None
            self.budget.settle(reserved, usage)
            output = []
            refusal = False
            for item in document.get("output", []):
                if isinstance(item, dict) and item.get("type") == "message":
                    for part in item.get("content", []):
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            output.append(part.get("text", ""))
                        elif isinstance(part, dict) and part.get("type") == "refusal":
                            refusal = True
            raw_output = "".join(output) if all(isinstance(text, str) for text in output) else None
            code = None
            if refusal:
                code = "refused"
            elif document.get("status") == "incomplete":
                code = "incomplete"
            elif document.get("status") != "completed" or not raw_output:
                code = "invalid_response"
            attempts.append(InferenceAttempt(
                number=number, status=code or "completed", http_status=response.status,
                request_id=response.request_id if isinstance(response.request_id, str) else None,
                response_id=document.get("id"),
                reported_model=document.get("model"), usage=usage, elapsed_ms=elapsed,
                raw_output=raw_output, error_code=code,
            ))
            if self.clock() > deadline:
                raise ReaderError("Reader deadline reached after the API response.", code="deadline_exceeded")
            if code:
                raise ReaderError("VLM output was refused, incomplete or invalid.", code=code)
            return raw_output
        raise ReaderError("API call limit reached.", code="api_call_limit")
