"""Deterministic local adapter for testing the entire data factory without API spend.

Usage numbers are a deterministic function of the request. The provider request ID
is unique per call, like a real provider's, so repeated identical requests are not
mistaken for duplicated records by data-quality validation.
"""

from __future__ import annotations

import hashlib
import uuid

from seshat.data.models import EffortLevel, NormalizedUsage, ProviderResult
from seshat.providers.base import ProviderAdapter


class MockAdapter(ProviderAdapter):
    name = "mock"

    def execute(
        self,
        *,
        prompt: str,
        model: str,
        effort: EffortLevel,
        max_output_tokens: int,
        temperature: float | None,
    ) -> ProviderResult:
        del temperature
        words = max(1, len(prompt.split()))
        jitter = int(hashlib.sha256(f"{prompt}|{model}|{effort}".encode()).hexdigest()[:4], 16) % 17
        multiplier = {"low": 0.8, "medium": 1.0, "high": 1.25}.get(effort, 1.0)
        natural_output = max(8, round((24 + words * 0.7 + jitter) * multiplier))
        output_tokens = min(max_output_tokens, natural_output)
        input_tokens = max(1, round(len(prompt) / 4))
        return ProviderResult(
            provider_request_id=f"mock_{uuid.uuid4().hex}",
            model_resolved=model,
            usage=NormalizedUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            provider_usage_raw={"input_tokens": input_tokens, "output_tokens": output_tokens},
            provider_response_raw={"fixture": True},
            finish_reason="length" if output_tokens == max_output_tokens else "stop",
            response_text="x" * output_tokens,
            latency_ms=1.0,
            is_truncated=output_tokens == max_output_tokens,
        )
