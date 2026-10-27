"""Shared transport and accounting for independent structured image requests."""
from __future__ import annotations

import base64
import json

from PIL import Image

from provider import OpenAIProvider
from reader import ReaderError
from schemas import InferenceTrace


def strict_schema(contract) -> dict:
    schema = contract.model_json_schema()

    def strict(node):
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            for child in list(node.values()):
                strict(child)
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
                node["additionalProperties"] = False
            # Responses schemas express homogeneous coordinates as arrays, not tuples.
            if "prefixItems" in node:
                items = node.pop("prefixItems")
                if not items or any(item != items[0] for item in items):
                    raise ValueError("Structured image schemas require homogeneous arrays")
                node["items"] = items[0]
        elif isinstance(node, list):
            for child in node:
                strict(child)

    strict(schema)
    return schema


class StructuredImageClient:
    def __init__(self, provider: OpenAIProvider):
        self.provider = provider
        self.last_trace: InferenceTrace | None = None

    def begin_observation(self) -> None:
        self.provider.begin_observation()

    def check_observation_budget(self) -> None:
        self.provider.check_observation_budget()

    def request(self, image_path, *, instructions, prompt_version, schema, schema_name,
                target_data, parse):
        provider, config = self.provider, self.provider.config
        started = provider.clock()
        attempts = []
        error_code = None
        with Image.open(image_path) as image:
            width, height = image.size
        try:
            payload = {
                "model": config.model, "instructions": instructions,
                "store": False, "truncation": "disabled",
                "max_output_tokens": config.max_output_tokens,
                "text": {"format": {
                    "type": "json_schema", "name": schema_name,
                    "strict": True, "schema": schema,
                }},
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": json.dumps(target_data, ensure_ascii=False)},
                    {"type": "input_image", "detail": config.image_detail,
                     "image_url": "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")},
                ]}],
            }
            raw = provider.generate(payload, attempts, started + config.deadline_seconds)
            return parse(raw)
        except ReaderError as exc:
            error_code = exc.code
            raise
        finally:
            complete_usage = bool(attempts) and all(
                item.usage.input_tokens is not None and item.usage.output_tokens is not None
                for item in attempts
            )
            input_tokens = sum(item.usage.input_tokens for item in attempts) if complete_usage else None
            output_tokens = sum(item.usage.output_tokens for item in attempts) if complete_usage else None
            cost = config.pricing.cost(input_tokens, output_tokens) if config.pricing is not None and complete_usage else None
            self.last_trace = InferenceTrace(
                provider=provider.name, requested_model=config.model, prompt_version=prompt_version,
                image_detail=config.image_detail, image_width=width, image_height=height,
                max_output_tokens=config.max_output_tokens, timeout_seconds=config.timeout_seconds,
                deadline_seconds=config.deadline_seconds, api_call_limit=config.max_api_calls,
                observation_api_call_limit=config.max_observation_api_calls,
                attempts=attempts, elapsed_ms=max(0, (provider.clock() - started) * 1000),
                input_tokens=input_tokens, output_tokens=output_tokens,
                estimated_cost_usd=str(cost) if cost is not None else None,
                cost_basis="configured_token_rates" if config.pricing is not None else "unconfigured",
                pricing_source=config.pricing.source if config.pricing is not None else None,
                batch_cost_limit_usd=str(config.max_cost_usd) if config.max_cost_usd is not None else None,
                batch_cost_reserved_usd=str(provider.budget.reserved) if config.pricing is not None else None,
                error_code=error_code,
            )
