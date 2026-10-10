"""Anthropic Messages API adapter."""

from __future__ import annotations

import time

from seshat.data.models import EffortLevel, NormalizedUsage, ProviderResult
from seshat.providers.base import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"

    def __init__(self) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                'Install provider dependencies with pip install -e ".[providers]"'
            ) from exc
        self._client = Anthropic()

    def execute(
        self,
        *,
        prompt: str,
        model: str,
        effort: EffortLevel,
        max_output_tokens: int,
        temperature: float | None,
    ) -> ProviderResult:
        if effort not in {"auto", "none"}:
            raise ValueError(
                "Anthropic effort is not normalized to OpenAI reasoning effort. "
                "Use effort=auto unless a provider-specific experimental mapping is added."
            )
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        started = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = response.usage
        raw_usage = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else {}
        raw_response = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
        text_parts = [
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        response_text = "".join(text_parts)
        return ProviderResult(
            provider_request_id=getattr(response, "id", None),
            model_resolved=getattr(response, "model", None),
            usage=NormalizedUsage(
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                cached_input_tokens=getattr(usage, "cache_read_input_tokens", None),
                cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", None),
            ),
            provider_usage_raw=raw_usage,
            provider_response_raw=raw_response,
            finish_reason=getattr(response, "stop_reason", None),
            response_text=response_text,
            latency_ms=latency_ms,
            is_truncated=getattr(response, "stop_reason", None) == "max_tokens",
        )
