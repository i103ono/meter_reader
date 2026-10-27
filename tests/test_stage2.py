"""Fabricated HTTP fixtures only: no live API, dataset answers, or accuracy claims."""
from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli import main
from provider import (
    HTTPResult, OpenAIProvider, ProviderConfig, TokenPricing, TransportFailure, http_post,
)
from reader import ReaderError
from recording import run_observation
from schemas import AnalogStage, AnalogTrace, ObservationInput, ReaderOutput, ScaleInfo
from vlm_reader import VLMReader, parse_reading, structured_schema


MODEL = "test-openai-vision-not-a-real-model"


def reading(kind="digital"):
    if kind == "unknown":
        return ReaderOutput().model_dump()
    if kind == "analog":
        return ReaderOutput(
            meter_type="analog", value=0.42, raw_value_text="0.42", unit="MPa",
            raw_unit_text="MPa", value_status="readable", unit_status="readable",
            value_evidence="Fabricated test: needle lies between visible marks.",
            unit_evidence="Fabricated test: MPa is visible.",
            scale=ScaleInfo(min_value=0, max_value=1, minor_division=0.02,
                            unit="MPa", status="readable", source="image"),
            analog_trace=AnalogTrace(
                **{stage: AnalogStage(status="visible", evidence="Synthetic stage observation, not image evidence.")
                   for stage in ("target", "geometry", "scale", "needle", "interpolation")},
                view="unknown", center=None, tip=None, counterweight=None, anchors=None,
                scale_unit="MPa", interpolation_mode="unknown", fraction=None,
            ),
        ).model_dump()
    return ReaderOutput(
        meter_type="digital", value=123.4, raw_value_text="00123.40",
        unit="m³", raw_unit_text="m³", value_status="readable", unit_status="readable",
        value_evidence="Fabricated test digits, not an actual measurement.",
        unit_evidence="Fabricated test unit.",
        scale=ScaleInfo(status="not_applicable"),
    ).model_dump()


def response(reading_output=None, **changes):
    document = {
        "id": "resp_fixture", "model": MODEL, "status": "completed",
        "usage": {
            "input_tokens": 111, "output_tokens": 22, "total_tokens": 133,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": json.dumps(reading_output if reading_output is not None else reading())},
        ]}],
    }
    document.update(changes)
    return HTTPResult(json.dumps(document).encode(), 200, "req_fixture")


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "meter.png"
        with Image.new("RGB", (12, 8), "white") as image:
            image.save(self.image)
        self.environment = patch.dict(os.environ, {"METER_READER_API_KEY": "fixture-secret-not-real"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.sent = []
        self.clock = Clock()

    def execute(self, outcomes=None, config=None, **request_fields):
        queue = list(outcomes or [response()])

        def transport(request, timeout):
            self.sent.append((request, timeout))
            outcome = queue.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        provider = OpenAIProvider(
            config or ProviderConfig(model=MODEL), transport=transport,
            clock=self.clock, sleep=self.clock.sleep,
        )
        reader = VLMReader(provider)
        path = run_observation(ObservationInput(image_path=self.image, **request_fields),
                               reader, self.root / "runs")
        return path, json.loads(path.read_text()), reader

    def test_analog_and_digital_structured_output(self):
        for kind in ("analog", "digital"):
            with self.subTest(kind=kind):
                path, result, _ = self.execute([response(reading(kind))])
                self.assertEqual(result["stage"], 2)
                self.assertEqual(result["mode"], "inference")
                self.assertEqual(result["status"], "needs_review")
                self.assertFalse(result["auto_accept"])
                self.assertIsNone(result["effective_value"])
                self.assertEqual(result["reading"]["meter_type"], kind)
                self.assertEqual(result["api_calls"], 1)
                self.assertEqual(result["inference"]["input_tokens"], 111)
                self.assertEqual(result["inference"]["output_tokens"], 22)
                self.assertIsNone(result["inference"]["estimated_cost_usd"])
                self.assertEqual(result["inference"]["attempts"][0]["request_id"], "req_fixture")
                self.assertIn("Fabricated test", result["inference"]["attempts"][0]["raw_output"])
                self.assertEqual(self.image.read_bytes(), (path.parent / "source.png").read_bytes())

    def test_request_is_independent_roi_only_and_contains_no_answers_or_history(self):
        path, result, _ = self.execute(
            roi=[1, 2, 6, 7], question="Read the upper display only",
            meter_id="not-forwarded", captured_at="2026-10-27T10:00:00+09:00",
            dataset={"name": "not-forwarded", "source": {"secret_truth": 987654}},
            acceptance_policy={"previous_value": 654321},
        )
        request, timeout = self.sent[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(timeout, 30)
        self.assertEqual(payload["model"], MODEL)
        self.assertFalse(payload["store"])
        self.assertEqual(payload["truncation"], "disabled")
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])
        text, image = payload["input"][0]["content"]
        self.assertEqual(json.loads(text["text"]), {"target_type": "auto", "question": "Read the upper display only"})
        self.assertEqual(base64.b64decode(image["image_url"].split(",", 1)[1]), (path.parent / "roi.png").read_bytes())
        self.assertEqual((result["inference"]["image_width"], result["inference"]["image_height"]), (5, 5))
        self.assertNotIn("987654", json.dumps(payload))
        self.assertNotIn("654321", json.dumps(payload))
        self.assertNotIn("fixture-secret-not-real", path.read_text())

    def test_schema_requires_explicit_nulls_and_image_only_scale_sources(self):
        schema = structured_schema()
        for obj in (schema, schema["$defs"]["ScaleInfo"]):
            self.assertEqual(set(obj["required"]), set(obj["properties"]))
            self.assertFalse(obj["additionalProperties"])
        self.assertEqual(schema["$defs"]["ScaleInfo"]["properties"]["source"]["enum"], ["image", "none"])
        self.assertNotIn('"default"', json.dumps(schema))

    def test_unit_aliases_preserve_value_magnitude_and_raw_text(self):
        digital = parse_reading(json.dumps(reading()))
        self.assertEqual(digital.unit, "m^3")
        self.assertEqual(digital.raw_unit_text, "m³")
        self.assertEqual(digital.raw_value_text, "00123.40")
        self.assertEqual(digital.value, 123.4)
        analog = reading("analog")
        analog.update(unit="Pa", raw_unit_text="Pa")
        self.assertEqual(parse_reading(json.dumps(analog)).unit, "Pa")
        analog["scale"]["unit"] = "m³"
        self.assertEqual(parse_reading(json.dumps(analog)).scale.unit, "m^3")

    def test_invalid_readings_are_errors_and_raw_output_remains_available(self):
        candidates = [{"meter_type": "digital"}, reading(), reading(), reading(), reading()]
        candidates[1]["scale"]["source"] = "registry"
        candidates[2]["value"] = None
        candidates[3]["value"] = True
        candidates[4]["value"] = float("nan")
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                _, result, _ = self.execute([response(candidate)])
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["reading"])
                self.assertEqual(result["reason_codes"], ["invalid_reading"])
                self.assertEqual(result["api_calls"], 1)
                self.assertIsNotNone(result["inference"]["attempts"][0]["raw_output"])

    def test_no_values_is_unreadable_and_partial_values_are_preserved(self):
        _, result, _ = self.execute([response(reading("unknown"))])
        self.assertEqual(result["status"], "unreadable")
        self.assertIsNone(result["reading"]["value"])
        partial = reading()
        partial.update(unit=None, raw_unit_text=None, unit_status="unreadable")
        _, result, _ = self.execute([response(partial)])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["reading"]["value"], 123.4)
        self.assertIsNone(result["reading"]["unit"])

    def test_auth_errors_do_not_retry_or_become_unreadable(self):
        for status, code in ((401, "authentication_error"), (403, "permission_error")):
            with self.subTest(status=status):
                before = len(self.sent)
                _, result, _ = self.execute([TransportFailure(code, status=status)])
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["reason_codes"], [code])
                self.assertEqual(result["inference"]["attempts"][0]["http_status"], status)
                self.assertEqual(len(self.sent) - before, 1)

    def test_one_transient_retry_is_counted_and_unknown_usage_is_not_zero(self):
        _, result, _ = self.execute([TransportFailure("rate_limit_error", status=429), response()])
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(self.clock.waits, [0.5])
        self.assertIsNone(result["inference"]["input_tokens"])
        self.assertIsNone(result["inference"]["estimated_cost_usd"])
        self.assertEqual(result["inference"]["attempts"][1]["usage"]["input_tokens"], 111)

    def test_retry_limit_and_large_retry_after(self):
        _, result, _ = self.execute([
            TransportFailure("network_error"), TransportFailure("network_error"),
        ])
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["status"], "error")
        self.clock.waits.clear()
        _, result, _ = self.execute([TransportFailure("rate_limit_error", status=429, retry_after=60)])
        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(result["reason_codes"], ["retry_budget_exceeded"])
        self.assertEqual(self.clock.waits, [])

    def test_incomplete_and_refused_are_technical_errors_without_retry(self):
        for document, code in (
            (response(status="incomplete"), "incomplete"),
            (response(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "fixture"}]}]), "refused"),
        ):
            with self.subTest(code=code):
                _, result, _ = self.execute([document])
                self.assertEqual(result["reason_codes"], [code])
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["api_calls"], 1)

    def test_malformed_response_metadata_is_safely_recorded(self):
        for body in (b"not-json", b"[]", b'{"output":null}', b'{"output":[{"type":"message","content":5}]}',
                     b'{"output":[],"usage":{"input_tokens":true}}'):
            with self.subTest(body=body):
                _, result, _ = self.execute([HTTPResult(body, 200)])
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["reason_codes"], ["invalid_response"])
                self.assertEqual(result["api_calls"], 1)

    def test_missing_credential_makes_zero_requests_and_has_a_trace(self):
        with patch.dict(os.environ, {}, clear=True):
            _, result, _ = self.execute()
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["reason_codes"], ["credential_missing"])
        self.assertEqual(result["status"], "error")
        self.assertEqual(len(self.sent), 0)

    def test_custom_and_proxy_placeholder_credentials_are_used_without_inspection(self):
        with patch.dict(os.environ, {"CUSTOM_MODEL_KEY": "proxy-placeholder"}):
            _, _, _ = self.execute(config=ProviderConfig(model=MODEL, api_key_env="CUSTOM_MODEL_KEY"))
        self.assertEqual(self.sent[0][0].get_header("Authorization"), "Bearer proxy-placeholder")

    def test_models_are_selected_by_configuration_only(self):
        for model in ("test-model-a", "test-model-b"):
            with self.subTest(model=model):
                _, result, _ = self.execute(config=ProviderConfig(model=model))
                payload = json.loads(self.sent[-1][0].data)
                self.assertEqual(payload["model"], model)
                self.assertEqual(result["inference"]["requested_model"], model)
                self.assertEqual(result["inference"]["attempts"][0]["reported_model"], MODEL)

    def test_gpt6_luna_example_is_sent_verbatim_without_live_api(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "openai_gpt6_luna.json"
        config = ProviderConfig.model_validate_json(example.read_text())
        _, result, _ = self.execute(config=config)
        self.assertEqual(json.loads(self.sent[0][0].data)["model"], "gpt-6-luna")
        self.assertEqual(result["inference"]["requested_model"], "gpt-6-luna")
        self.assertEqual(result["inference"]["max_output_tokens"], 4096)
        self.assertIsNone(result["inference"]["estimated_cost_usd"])

    def test_http_retry_after_date_and_incomplete_body(self):
        import http.client
        import urllib.request
        request = urllib.request.Request("https://api.openai.com/v1/responses", data=b"{}")
        error = urllib.error.HTTPError(
            request.full_url, 429, "fixture",
            {"Retry-After": "Wed, 01 Jan 2100 00:00:00 GMT"}, io.BytesIO(b""),
        )
        for failure in (error, http.client.IncompleteRead(b"partial")):
            with self.subTest(failure=type(failure).__name__), patch("provider.urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = failure
                with self.assertRaises(TransportFailure) as caught:
                    http_post(request, 1)
                if failure is error:
                    self.assertGreater(caught.exception.retry_after, 5)
                else:
                    self.assertEqual(caught.exception.code, "network_error")

    def test_mock_mode_never_uses_provider_even_with_a_credential_present(self):
        with (
            patch("provider.http_post", side_effect=AssertionError("mock must not call HTTP")) as transport,
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()),
        ):
            status = main([str(self.image), "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 0)
        transport.assert_not_called()
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["mode"], "mock")
        self.assertEqual(result["api_calls"], 0)

    def pricing_config(self, **changes):
        return ProviderConfig(
            model=MODEL, max_output_tokens=128,
            pricing=TokenPricing(model=MODEL, input_usd_per_million="1",
                                 output_usd_per_million="2", max_input_tokens_per_call=1000,
                                 source="fabricated-test-ratecard-not-real-pricing"),
            **changes,
        )

    def test_cost_limit_is_checked_before_network_and_exact_usage_is_recorded(self):
        _, blocked, _ = self.execute(config=self.pricing_config(max_cost_usd="0.001"))
        self.assertEqual(blocked["api_calls"], 0)
        self.assertEqual(blocked["reason_codes"], ["cost_budget_exceeded"])
        _, accepted, _ = self.execute(config=self.pricing_config(max_cost_usd="0.01"))
        self.assertEqual(Decimal(accepted["inference"]["estimated_cost_usd"]), Decimal("0.000155"))
        self.assertEqual(accepted["inference"]["cost_basis"], "configured_token_rates")

    def test_batch_budget_is_shared_between_observations(self):
        _, first, reader = self.execute(config=self.pricing_config(max_cost_usd="0.0014"))
        path = run_observation(ObservationInput(image_path=self.image), reader, self.root / "runs")
        second = json.loads(path.read_text())
        self.assertEqual(first["status"], "needs_review")
        self.assertEqual(second["reason_codes"], ["cost_budget_exceeded"])
        self.assertEqual(second["api_calls"], 0)
        self.assertEqual(len(self.sent), 1)

    def test_failed_attempt_keeps_cost_reservation(self):
        _, result, _ = self.execute(
            [TransportFailure("rate_limit_error", status=429)],
            config=self.pricing_config(max_cost_usd="0.002"),
        )
        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(result["reason_codes"], ["cost_budget_exceeded"])
        self.assertIsNone(result["inference"]["estimated_cost_usd"])
        self.assertEqual(Decimal(result["inference"]["batch_cost_reserved_usd"]), Decimal("0.001256"))

    def test_pricing_model_and_config_validation(self):
        for changes in ({"max_cost_usd": "1"}, {"timeout_seconds": 31}, {"max_api_calls": 3},
                        {"model": " "}, {"api_key_env": "not a variable"}):
            values = {"model": MODEL, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                ProviderConfig(**values)
        with self.assertRaises(ValidationError):
            ProviderConfig(model="different", pricing=self.pricing_config().pricing)

    def test_deadline_clamps_timeout_and_rejects_late_results(self):
        def slow(request, timeout):
            self.assertEqual(timeout, 0.25)
            self.clock.now += 0.3
            return response()
        reader = VLMReader(OpenAIProvider(
            ProviderConfig(model=MODEL, deadline_seconds=0.25), transport=slow,
            clock=self.clock, sleep=self.clock.sleep,
        ))
        path = run_observation(ObservationInput(image_path=self.image), reader, self.root / "runs")
        result = json.loads(path.read_text())
        self.assertEqual(result["api_calls"], 1)
        self.assertEqual(result["reason_codes"], ["deadline_exceeded"])

    def test_http_error_body_is_never_echoed_and_redirects_are_disabled(self):
        import urllib.request
        request = urllib.request.Request("https://api.openai.com/v1/responses", data=b"{}")
        error = urllib.error.HTTPError(
            request.full_url, 401, "fixture-secret-not-real", {},
            io.BytesIO(b'{"error":{"message":"fixture-secret-not-real"}}'),
        )
        with patch("provider.urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(TransportFailure) as caught:
                http_post(request, 1)
            self.assertEqual(caught.exception.code, "authentication_error")
            self.assertNotIn("fixture-secret-not-real", str(caught.exception))
            self.assertEqual(type(opener.call_args.args[0]).__name__, "NoRedirects")

    def test_cli_inference_errors_exit_nonzero_while_saving_error_json(self):
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
            status = main([str(self.image), "--mode", "inference", "--model", MODEL,
                           "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 1)
        self.assertIn("inputs: 1, saved: 1, errors: 1", stderr.getvalue())
        result = json.loads(Path(stdout.getvalue().strip()).read_text())
        self.assertEqual(result["reason_codes"], ["credential_missing"])

    def test_cli_inference_success_with_http_fixture_and_limit(self):
        manifest = self.root / "inputs.jsonl"
        manifest.write_text('{"image_path":"meter.png"}\n' * 2)
        with (
            patch("provider.http_post", return_value=response()) as transport,
            patch("socket.socket", side_effect=AssertionError("no network in tests")),
            redirect_stdout(io.StringIO()) as stdout,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            status = main(["--inputs", str(manifest), "--mode", "inference", "--model", MODEL,
                           "--limit", "1", "--output-dir", str(self.root / "runs")])
        self.assertEqual(status, 0, stderr.getvalue())
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(json.loads(Path(stdout.getvalue().strip()).read_text())["stage"], 2)

    def test_cli_requires_explicit_model_and_never_echoes_config_secrets(self):
        config = self.root / "provider.json"
        config.write_text(json.dumps({"model": MODEL, "api_key": "fixture-secret-not-real"}))
        for args in ([], ["--provider-config", str(config)]):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                status = main([str(self.image), "--mode", "inference", *args])
            self.assertEqual(status, 1)
            self.assertNotIn("fixture-secret-not-real", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
