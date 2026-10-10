"""Anthropic Messages API adapter (official ``anthropic`` SDK).

Usage semantics are normalized by ``normalization.normalize_anthropic``. The SDK's
own retries are disabled (``max_retries=0``) so the collector alone decides when a
retry is safe; a timeout after dispatch is reported as uncertain, never re-sent.
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
from seshat.providers.normalization import NormalizedUsage, normalize_anthropic
from seshat.providers.payloads import parse_anthropic

_COUNT_TOKENS_PARAMS = {
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "output_config",
    "cache_control",
}


def _content(message: Message) -> Any:
    if isinstance(message.content, str):
        if message.role == "tool":
            return [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
            ]
        return message.content
    blocks: list[dict[str, Any]] = []
    for part in message.content:
        if part.type == "text":
            blocks.append({"type": "text", "text": part.text})
        elif part.type in {"image", "document"}:
            mime, data = read_attachment(part)
            blocks.append(
                {"type": part.type, "source": {"type": "base64", "media_type": mime, "data": data}}
            )
        else:
            raise ProviderError(
                ErrorCategory.UNSUPPORTED, f"Anthropic adapter does not send {part.type} parts"
            )
    if message.role == "tool":
        return [{"type": "tool_result", "tool_use_id": message.tool_call_id, "content": blocks}]
    return blocks


def build_request_kwargs(request: LLMRequest) -> dict[str, Any]:
    """Translate an ``LLMRequest`` into ``messages.create`` keyword arguments."""
    params = request.params
    kwargs: dict[str, Any] = {
        "model": request.model,
        "max_tokens": require_max_output(request),
        "messages": [
            {"role": "user" if m.role == "tool" else m.role, "content": _content(m)}
            for m in request.messages
        ],
    }
    if request.system:
        kwargs["system"] = request.system
    if request.tools:
        kwargs["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            if t.type == "function"
            else {"type": t.type, "name": t.name}
            for t in request.tools
        ]
    if params.tool_choice or params.parallel_tool_calls is not None:
        choice = params.tool_choice or "auto"
        tool_choice: dict[str, Any] = (
            {"type": choice}
            if choice in {"auto", "any", "none"}
            else {"type": "tool", "name": choice}
        )
        if params.parallel_tool_calls is False and choice != "none":
            tool_choice["disable_parallel_tool_use"] = True
        kwargs["tool_choice"] = tool_choice
    if params.thinking_mode == "enabled":
        if params.thinking_budget_tokens is None:
            raise ProviderError(
                ErrorCategory.INVALID_REQUEST, "thinking_mode=enabled needs thinking_budget_tokens"
            )
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": params.thinking_budget_tokens}
    elif params.thinking_mode in {"adaptive", "disabled"}:
        kwargs["thinking"] = {"type": params.thinking_mode}
    output_config: dict[str, Any] = {}
    if params.effort is not None:
        output_config["effort"] = params.effort
    if params.response_format == "json_schema":
        output_config["format"] = {
            "type": "json_schema",
            "schema": params.response_schema or {"type": "object"},
        }
    elif params.response_format == "json":
        raise ProviderError(
            ErrorCategory.UNSUPPORTED, "Anthropic structured output requires a json_schema"
        )
    if output_config:
        kwargs["output_config"] = output_config
    if params.stop_sequences:
        kwargs["stop_sequences"] = list(params.stop_sequences)
    if params.cache.enabled:
        kwargs["cache_control"] = {
            "type": "ephemeral",
            **({"ttl": params.cache.ttl} if params.cache.ttl else {}),
        }
    if params.service_tier:
        kwargs["service_tier"] = params.service_tier
    kwargs.update(public_options(request))
    return kwargs


def classify_exception(exc: Exception) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    try:
        import anthropic
    except ImportError:  # SDK not installed: only generic exceptions can occur
        if isinstance(exc, (TypeError, ValueError)):
            return classify_client_side(exc)
        return ProviderError(ErrorCategory.UNKNOWN, f"{type(exc).__name__}: {exc}"[:500])
    if isinstance(exc, anthropic.APITimeoutError):
        return ProviderError(ErrorCategory.TIMEOUT, str(exc))
    if isinstance(exc, anthropic.APIConnectionError):
        return ProviderError(ErrorCategory.CONNECTION, str(exc))
    if isinstance(exc, anthropic.APIStatusError):
        category = classify_status(exc.status_code)
        retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
        return ProviderError(
            category,
            str(exc)[:500],
            status_code=exc.status_code,
            code=getattr(exc, "type", None) or type(exc).__name__,
            retry_after_seconds=float(retry_after)
            if retry_after and retry_after.isdigit()
            else None,
        )
    if isinstance(exc, (anthropic.AnthropicError, TypeError, ValueError)):
        return classify_client_side(exc)  # raised before the HTTP request was sent
    return ProviderError(ErrorCategory.UNKNOWN, f"{type(exc).__name__}: {exc}"[:500])


class AnthropicAdapter(ProviderAdapter):
    name: ClassVar[str] = "anthropic"
    family: ClassVar[str] = "anthropic"
    api_surface: ClassVar[str] = "anthropic.messages"

    def __init__(self, client: Any | None = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    'Install provider SDKs with pip install -e ".[providers]"'
                ) from exc
            client = anthropic.Anthropic(max_retries=0)
            if not any(
                getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials")
            ):
                raise RuntimeError("no Anthropic credentials configured (set ANTHROPIC_API_KEY)")
        self._client = client

    def sdk_version(self) -> str | None:
        try:
            import anthropic

            return f"anthropic-python/{anthropic.__version__}"
        except ImportError:
            return None

    def count_input_tokens(self, request: LLMRequest) -> PreflightResult:
        kwargs = {
            k: v for k, v in build_request_kwargs(request).items() if k in _COUNT_TOKENS_PARAMS
        }
        started = time.perf_counter()
        try:
            result = self._client.messages.count_tokens(**kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            return PreflightResult(
                input_tokens=None,
                method=PreflightMethod.PROVIDER_NATIVE,
                preflight_provider=self.name,
                timestamp_utc=datetime.now(UTC),
                error=str(classify_exception(exc))[:500],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        includes = (
            ["system", "messages"]
            + (["tools"] if request.tools else [])
            + (["thinking"] if "thinking" in kwargs else [])
        )
        return PreflightResult(
            input_tokens=int(result.input_tokens),
            method=PreflightMethod.PROVIDER_NATIVE,
            preflight_provider=self.name,
            tokenizer_name="anthropic.messages.count_tokens",
            tokenizer_version=request.model,
            timestamp_utc=datetime.now(UTC),
            latency_ms=(time.perf_counter() - started) * 1000,
            includes=includes,
            # Documented as an estimate; server tools / URL sources are not countable.
            complete=all(tool.type == "function" for tool in request.tools),
        )

    def execute_request(self, request: LLMRequest, *, timeout_seconds: float) -> ProviderResponse:
        kwargs = build_request_kwargs(request)
        started = time.perf_counter()
        try:
            message = self._client.messages.create(**kwargs, timeout=timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            raise classify_exception(exc) from exc
        payload = (
            message.model_dump(mode="json") if hasattr(message, "model_dump") else dict(message)
        )
        return parse_anthropic(payload, latency_ms=(time.perf_counter() - started) * 1000)

    def normalize_usage(self, response: ProviderResponse) -> NormalizedUsage:
        return normalize_anthropic(response.raw_usage)
