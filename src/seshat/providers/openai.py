"""OpenAI Responses API adapter (official ``openai`` SDK).

Sends ``store=False`` so responses are not retained server-side. SDK retries are
disabled; the collector handles retries. Preflight uses
``responses.input_tokens.count`` (POST /v1/responses/input_tokens).
"""

from __future__ import annotations

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
from seshat.providers.normalization import NormalizedUsage, normalize_openai
from seshat.providers.payloads import parse_openai

_COUNT_PARAMS = {
    "model",
    "input",
    "instructions",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning",
    "text",
}


def _input_item(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": message.text(),
        }
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}
    text_type = "output_text" if message.role == "assistant" else "input_text"
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if part.type == "text":
            parts.append({"type": text_type, "text": part.text})
        elif part.type == "image":
            mime, data = read_attachment(part)
            parts.append({"type": "input_image", "image_url": f"data:{mime};base64,{data}"})
        elif part.type == "document":
            mime, data = read_attachment(part)
            parts.append(
                {
                    "type": "input_file",
                    "filename": "document",
                    "file_data": f"data:{mime};base64,{data}",
                }
            )
        else:
            raise ProviderError(
                ErrorCategory.UNSUPPORTED, f"OpenAI adapter does not send {part.type} parts"
            )
    return {"role": message.role, "content": parts}


def build_request_kwargs(request: LLMRequest) -> dict[str, Any]:
    params = request.params
    kwargs: dict[str, Any] = {
        "model": request.model,
        "input": [_input_item(m) for m in request.messages],
        "max_output_tokens": require_max_output(request),
        "store": False,
    }
    if request.system:
        kwargs["instructions"] = request.system
    if params.effort is not None:
        kwargs["reasoning"] = {"effort": params.effort}
    text: dict[str, Any] = {}
    if params.verbosity is not None:
        text["verbosity"] = params.verbosity
    if params.response_format == "json":
        text["format"] = {"type": "json_object"}
    elif params.response_format == "json_schema":
        text["format"] = {
            "type": "json_schema",
            "name": "output",
            "schema": params.response_schema or {"type": "object"},
        }
    if text:
        kwargs["text"] = text
    if request.tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            if t.type == "function"
            else {"type": t.type}
            for t in request.tools
        ]
    if params.tool_choice:
        choice = params.tool_choice
        kwargs["tool_choice"] = (
            {"auto": "auto", "none": "none", "any": "required"}[choice]
            if choice in {"auto", "none", "any"}
            else {"type": "function", "name": choice}
        )
    if params.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = params.parallel_tool_calls
    if params.temperature is not None:
        kwargs["temperature"] = params.temperature
    if params.top_p is not None:
        kwargs["top_p"] = params.top_p
    if params.cache.enabled:
        if params.cache.key:
            kwargs["prompt_cache_key"] = params.cache.key
        if params.cache.ttl:
            kwargs["prompt_cache_retention"] = params.cache.ttl
    if params.service_tier:
        kwargs["service_tier"] = params.service_tier
    kwargs.update(public_options(request))
    return kwargs


def classify_exception(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    try:
        import openai
    except ImportError:  # SDK not installed: only generic exceptions can occur
        if isinstance(exc, (TypeError, ValueError)):
            return classify_client_side(exc)
        return ProviderError(ErrorCategory.UNKNOWN, f"{type(exc).__name__}: {exc}"[:500])
    if isinstance(exc, openai.APITimeoutError):
        return ProviderError(ErrorCategory.TIMEOUT, str(exc))
    if isinstance(exc, openai.APIConnectionError):
        return ProviderError(ErrorCategory.CONNECTION, str(exc))
    if isinstance(exc, openai.APIStatusError):
        retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
        return ProviderError(
            classify_status(exc.status_code),
            str(exc)[:500],
            status_code=exc.status_code,
            code=getattr(exc, "code", None) or type(exc).__name__,
            retry_after_seconds=float(retry_after)
            if retry_after and retry_after.isdigit()
            else None,
        )
    if isinstance(exc, (openai.OpenAIError, TypeError, ValueError)):
        return classify_client_side(exc)  # raised before the HTTP request was sent
    return ProviderError(ErrorCategory.UNKNOWN, f"{type(exc).__name__}: {exc}"[:500])


class OpenAIAdapter(ProviderAdapter):
    name: ClassVar[str] = "openai"
    family: ClassVar[str] = "openai"
    api_surface: ClassVar[str] = "openai.responses"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                import openai
            except ImportError as exc:
                raise RuntimeError(
                    'Install provider SDKs with pip install -e ".[providers]"'
                ) from exc
            client = openai.OpenAI(max_retries=0)
        self._client = client

    def sdk_version(self) -> str | None:
        try:
            import openai

            return f"openai-python/{openai.__version__}"
        except ImportError:
            return None

    def count_input_tokens(self, request: LLMRequest) -> PreflightResult:
        kwargs = {k: v for k, v in build_request_kwargs(request).items() if k in _COUNT_PARAMS}
        started = time.perf_counter()
        try:
            result = self._client.responses.input_tokens.count(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                input_tokens=None,
                method=PreflightMethod.PROVIDER_NATIVE,
                preflight_provider=self.name,
                timestamp_utc=datetime.now(UTC),
                error=str(classify_exception(exc))[:500],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        return PreflightResult(
            input_tokens=int(result.input_tokens),
            method=PreflightMethod.PROVIDER_NATIVE,
            preflight_provider=self.name,
            tokenizer_name="openai.responses.input_tokens",
            tokenizer_version=request.model,
            timestamp_utc=datetime.now(UTC),
            latency_ms=(time.perf_counter() - started) * 1000,
            includes=["instructions", "input"] + (["tools"] if request.tools else []),
            complete=True,
        )

    def execute_request(self, request: LLMRequest, *, timeout_seconds: float) -> ProviderResponse:
        kwargs = build_request_kwargs(request)
        started = time.perf_counter()
        try:
            response = self._client.responses.create(**kwargs, timeout=timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            raise classify_exception(exc) from exc
        payload = (
            response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
        )
        return parse_openai(payload, latency_ms=(time.perf_counter() - started) * 1000)

    def normalize_usage(self, response: ProviderResponse) -> NormalizedUsage:
        return normalize_openai(response.raw_usage)
