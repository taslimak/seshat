"""OpenAI Responses API adapter.

The SDK is imported lazily so dataset tooling that does not execute provider calls
can run without provider extras installed.
"""

from __future__ import annotations

import time
from typing import Any

from seshat.data.models import EffortLevel, NormalizedUsage, ProviderResult
from seshat.providers.base import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'Install provider dependencies with pip install -e ".[providers]"'
            ) from exc
        self._client = OpenAI()

    def execute(
        self,
        *,
        prompt: str,
        model: str,
        effort: EffortLevel,
        max_output_tokens: int,
        temperature: float | None,
    ) -> ProviderResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        if effort != "auto":
            kwargs["reasoning"] = {"effort": effort}
        if temperature is not None:
            kwargs["temperature"] = temperature
        started = time.perf_counter()
        response = self._client.responses.create(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        raw_usage = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else {}
        raw_response = response.model_dump(mode="json") if hasattr(response, "model_dump") else {}
        incomplete = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete, "reason", None)
        return ProviderResult(
            provider_request_id=getattr(response, "id", None),
            model_resolved=getattr(response, "model", None),
            usage=NormalizedUsage(
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
                cached_input_tokens=getattr(input_details, "cached_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ),
            provider_usage_raw=raw_usage,
            provider_response_raw=raw_response,
            response_text=getattr(response, "output_text", None),
            latency_ms=latency_ms,
            is_truncated=incomplete_reason == "max_output_tokens",
        )
