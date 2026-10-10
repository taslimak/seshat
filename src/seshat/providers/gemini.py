"""Google Gemini adapter (official ``google-genai`` SDK, ``models.generate_content``).

Seshat ``effort`` maps to ``thinking_config.thinking_level``. SDK retries are left
off (no ``retry_options``). Preflight uses ``models.count_tokens``; when the system
instruction or tools cannot be included the count is marked incomplete.
"""

from __future__ import annotations

import importlib
import time
from datetime import UTC, datetime
from typing import Any, ClassVar

from seshat.data.schemas.enums import PreflightMethod
from seshat.data.schemas.request import LLMRequest, Message
from seshat.providers._common import (
    classify_client_side,
    public_options,
    read_attachment,
    require_max_output,
)
from seshat.providers.base import (
    ErrorCategory,
    PreflightResult,
    ProviderAdapter,
    ProviderError,
    ProviderResponse,
    classify_status,
)
from seshat.providers.normalization import NormalizedUsage, normalize_gemini
from seshat.providers.payloads import parse_gemini


def _content(message: Message) -> dict[str, Any]:
    role = "model" if message.role == "assistant" else "user"
    if message.role == "tool":
        return {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": message.name or "tool",
                        "response": {"content": message.text()},
                    }
                }
            ],
        }
    parts: list[dict[str, Any]] = []
    for part in message.parts():
        if part.type == "text":
            parts.append({"text": part.text})
        else:
            mime, data = read_attachment(part)
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
    return {"role": role, "parts": parts}


def build_contents(request: LLMRequest) -> list[dict[str, Any]]:
    return [_content(message) for message in request.messages]


def build_config(request: LLMRequest) -> dict[str, Any]:
    params = request.params
    config: dict[str, Any] = {"max_output_tokens": require_max_output(request)}
    if request.system:
        config["system_instruction"] = request.system
    for key in ("temperature", "top_p", "top_k", "seed"):
        value = getattr(params, key)
        if value is not None:
            config[key] = value
    if params.stop_sequences:
        config["stop_sequences"] = list(params.stop_sequences)
    if params.response_format in {"json", "json_schema"}:
        config["response_mime_type"] = "application/json"
        if params.response_format == "json_schema" and params.response_schema:
            config["response_json_schema"] = params.response_schema
    thinking: dict[str, Any] = {}
    if params.effort is not None:
        thinking["thinking_level"] = params.effort.upper()
    if params.thinking_budget_tokens is not None:
        thinking["thinking_budget"] = params.thinking_budget_tokens
    if thinking:
        config["thinking_config"] = thinking
    if request.tools:
        config["tools"] = [
            {
                "function_declarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters_json_schema": t.input_schema,
                    }
                    for t in request.tools
                    if t.type == "function"
                ]
            }
        ]
        # Disable automatic function calling: one request must stay one invocation.
        config["automatic_function_calling"] = {"disable": True}
    if params.tool_choice:
        mode = {"auto": "AUTO", "any": "ANY", "none": "NONE"}.get(params.tool_choice)
        config["tool_config"] = {
            "function_calling_config": (
                {"mode": mode}
                if mode
                else {"mode": "ANY", "allowed_function_names": [params.tool_choice]}
            )
        }
    if params.service_tier:
        config["service_tier"] = params.service_tier
    config.update(public_options(request))
    return config


def classify_exception(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    try:
        errors: Any = importlib.import_module("google.genai.errors")
    except ImportError:  # SDK not installed: only generic exceptions can occur
        errors = None
    if errors is not None and isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        return ProviderError(
            classify_status(code),
            str(exc)[:500],
            status_code=code,
            code=getattr(exc, "status", None),
        )
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return ProviderError(ErrorCategory.TIMEOUT, str(exc)[:500])
    if "connect" in name or "network" in name:
        return ProviderError(ErrorCategory.CONNECTION, str(exc)[:500])
    if isinstance(exc, (TypeError, ValueError)):
        return classify_client_side(exc)  # raised before the HTTP request was sent
    return ProviderError(ErrorCategory.UNKNOWN, f"{type(exc).__name__}: {exc}"[:500])


class GeminiAdapter(ProviderAdapter):
    name: ClassVar[str] = "gemini"
    family: ClassVar[str] = "gemini"
    api_surface: ClassVar[str] = "gemini.generate_content"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    'Install provider SDKs with pip install -e ".[providers]"'
                ) from exc
            client = genai.Client()
        self._client = client

    def sdk_version(self) -> str | None:
        try:
            from google.genai import version

            return f"google-genai/{version.__version__}"
        except ImportError:
            return None

    def count_input_tokens(self, request: LLMRequest) -> PreflightResult:
        contents = build_contents(request)
        started = time.perf_counter()
        includes = ["messages"]
        try:
            result = self._client.models.count_tokens(model=request.model, contents=contents)
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                input_tokens=None,
                method=PreflightMethod.PROVIDER_NATIVE,
                preflight_provider=self.name,
                timestamp_utc=datetime.now(UTC),
                error=str(classify_exception(exc))[:500],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if result.total_tokens is None:
            return PreflightResult(
                input_tokens=None,
                method=PreflightMethod.PROVIDER_NATIVE,
                preflight_provider=self.name,
                timestamp_utc=datetime.now(UTC),
                error="count_tokens returned no total_tokens",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return PreflightResult(
            input_tokens=int(result.total_tokens),
            method=PreflightMethod.PROVIDER_NATIVE,
            preflight_provider=self.name,
            tokenizer_name="gemini.models.count_tokens",
            tokenizer_version=request.model,
            timestamp_utc=datetime.now(UTC),
            latency_ms=(time.perf_counter() - started) * 1000,
            includes=includes,
            # contents only: system instruction and tool declarations are not counted
            complete=not request.system and not request.tools,
        )

    def execute_request(self, request: LLMRequest, *, timeout_seconds: float) -> ProviderResponse:
        config = build_config(request)
        config["http_options"] = {"timeout": int(timeout_seconds * 1000)}
        started = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=request.model,
                contents=build_contents(request),
                config=config,  # type: ignore[arg-type,unused-ignore]  # SDK accepts the dict form
            )
        except Exception as exc:  # noqa: BLE001
            raise classify_exception(exc) from exc
        payload = (
            response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        )
        return parse_gemini(payload, latency_ms=(time.perf_counter() - started) * 1000)

    def normalize_usage(self, response: ProviderResponse) -> NormalizedUsage:
        return normalize_gemini(response.raw_usage)
