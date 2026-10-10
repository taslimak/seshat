"""Offline mock providers that emit realistic OpenAI / Anthropic / Gemini payloads.

The mocks never touch the network. They produce payloads in each provider's SDK
shape (including cache fields, reasoning/thinking accounting and truncation) and
run them through the same parsers and normalizers as the live adapters, so the
offline pipeline exercises the real normalization code.

Outputs are a deterministic function of the request and of how many times the same
request has been executed by this adapter instance, so repeated observations vary
like real sampling while staying reproducible. Everything they produce is a
**test fixture** (``source_type=test_fixture``), never a real usage label.

``FailurePlan`` injects classified failures for retry/resume/uncertainty tests.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar

from seshat.data.provenance import DataOrigin
from seshat.data.schemas.enums import PreflightMethod
from seshat.data.schemas.request import LLMRequest
from seshat.providers.base import (
    ErrorCategory,
    PreflightResult,
    ProviderAdapter,
    ProviderError,
    ProviderResponse,
)
from seshat.providers.normalization import NORMALIZERS, NormalizedUsage
from seshat.providers.payloads import PARSERS

_EFFORT_REASONING: dict[str | None, float] = {
    None: 0.0,
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.35,
    "medium": 0.8,
    "high": 1.6,
    "xhigh": 2.4,
    "max": 3.2,
}
_CATEGORY_BASE = {
    "short_answer": 40,
    "classification": 30,
    "translation": 90,
    "extraction": 120,
    "summarization": 180,
    "question_answering": 220,
    "structured_json": 160,
    "tool_calling": 60,
    "code_review": 380,
    "debugging": 360,
    "code_generation": 420,
    "mathematics": 300,
    "reasoning": 340,
    "creative_writing": 450,
    "long_context_analysis": 400,
    "multi_turn": 250,
    "verbose_explanation": 650,
}
_FORMAT_FACTOR = {"json": 0.8, "bullet_list": 0.7, "table": 0.9, "code": 1.2}
_STATUS_FOR = {
    ErrorCategory.RATE_LIMIT: 429,
    ErrorCategory.OVERLOADED: 529,
    ErrorCategory.SERVER_ERROR: 500,
    ErrorCategory.INVALID_REQUEST: 400,
    ErrorCategory.AUTHENTICATION: 401,
}


@dataclass
class FailurePlan:
    """Deterministic failure injection: fail the n-th call (1-based) with a category."""

    by_call: dict[int, ErrorCategory] = field(default_factory=dict)
    every: int | None = None
    every_category: ErrorCategory = ErrorCategory.RATE_LIMIT

    def failure_for(self, call_number: int) -> ErrorCategory | None:
        if call_number in self.by_call:
            return self.by_call[call_number]
        if self.every and call_number % self.every == 0:
            return self.every_category
        return None


def _stable_int(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:12], 16)


def _length_target(request: LLMRequest, category: str | None) -> int:
    from seshat.data.features.extract import explicit_length_hint, requested_output_format

    base = _CATEGORY_BASE.get(category or "", 220)
    current = request.current_user_message().text()
    hint = explicit_length_hint(current) or ""
    if hint.startswith("max_words:"):
        base = min(base, int(hint.split(":")[1]) * 4 // 3)
    elif hint.startswith("count_sentences:"):
        base = 30 * int(hint.split(":")[1])
    elif hint.startswith(("count_bullet_points:", "count_items:", "count_steps:")):
        base = 28 * int(hint.split(":")[1])
    elif hint.startswith("count_paragraphs:"):
        base = 110 * int(hint.split(":")[1])
    elif hint == "qualitative:brief":
        base = int(base * 0.5)
    elif hint == "qualitative:detailed":
        base = int(base * 1.8)
    base = int(base * _FORMAT_FACTOR.get(requested_output_format(request), 1.0))
    if "one word" in current.lower() or "single word" in current.lower():
        base = 3
    return max(3, base)


class MockAdapter(ProviderAdapter):
    """Base mock; subclasses set the emulated ``family``."""

    name: ClassVar[str] = "mock-openai"
    family: ClassVar[str] = "openai"
    api_surface: ClassVar[str] = "mock.responses"
    data_origin: ClassVar[DataOrigin] = DataOrigin.SYNTHETIC
    makes_external_calls: ClassVar[bool] = False

    def __init__(self, failures: FailurePlan | None = None) -> None:
        self.failures = failures or FailurePlan()
        self.calls = 0
        self._executions: Counter[str] = Counter()
        self._cached_prefixes: set[str] = set()

    def sdk_version(self) -> str | None:
        return "seshat-mock/1"

    # -- preflight ------------------------------------------------------------
    def _true_input_tokens(self, request: LLMRequest) -> int:
        tools = json.dumps([tool.model_dump() for tool in request.tools]) if request.tools else ""
        overhead = {"anthropic": 8, "openai": 6, "gemini": 2}[self.family]
        return max(
            1,
            round(len(request.all_text()) / 3.8)
            + round(len(tools) / 3.4)
            + 4 * len(request.messages)
            + overhead,
        )

    def count_input_tokens(self, request: LLMRequest) -> PreflightResult:
        return PreflightResult(
            input_tokens=self._true_input_tokens(request),
            method=PreflightMethod.PROVIDER_NATIVE,
            preflight_provider=f"{self.name} (test fixture)",
            tokenizer_name="mock-tokenizer",
            tokenizer_version="1",
            timestamp_utc=datetime.now(UTC),
            latency_ms=0.0,
            includes=["system", "messages", "tools"],
            complete=True,
        )

    # -- execution ------------------------------------------------------------
    def execute_request(self, request: LLMRequest, *, timeout_seconds: float) -> ProviderResponse:
        self.calls += 1
        failure = self.failures.failure_for(self.calls)
        if failure is not None:
            raise ProviderError(
                failure, f"injected {failure.value}", status_code=_STATUS_FOR.get(failure)
            )
        key = request.model_dump_json()
        repetition = self._executions[key]
        self._executions[key] += 1
        rng = random.Random(_stable_int(key, repetition))
        category = request.provider_options.get("_mock_category")
        input_tokens = self._true_input_tokens(request)
        visible = max(1, round(_length_target(request, category) * rng.uniform(0.75, 1.3)))
        effort = request.params.effort
        if request.params.thinking_mode == "disabled":
            effort = None
        elif request.params.thinking_mode in {"enabled", "adaptive"} and effort is None:
            effort = "medium"
        reasoning = round(visible * _EFFORT_REASONING.get(effort, 0.8) * rng.uniform(0.6, 1.5))
        limit = request.params.max_output_tokens
        truncated = limit is not None and visible + reasoning > limit
        if truncated and limit is not None:
            reasoning = min(reasoning, limit)
            visible = limit - reasoning
        prefix_key = hashlib.sha256(((request.system or "") + key[:400]).encode()).hexdigest()
        seen = prefix_key in self._cached_prefixes
        self._cached_prefixes.add(prefix_key)
        tool_call = (
            bool(request.tools)
            and request.params.tool_choice != "none"
            and category == "tool_calling"
        )
        latency = round(
            180 + 0.02 * input_tokens + 9.0 * (visible + reasoning) * rng.uniform(0.8, 1.2), 1
        )
        builder = {
            "anthropic": self._anthropic_payload,
            "openai": self._openai_payload,
            "gemini": self._gemini_payload,
        }[self.family]
        payload = builder(
            request, input_tokens, visible, reasoning, truncated, seen, tool_call, rng
        )
        return PARSERS[self.family](payload, latency_ms=latency)

    def _anthropic_payload(
        self,
        request: LLMRequest,
        input_tokens: int,
        visible: int,
        reasoning: int,
        truncated: bool,
        seen: bool,
        tool_call: bool,
        rng: random.Random,
    ) -> dict[str, Any]:
        cache = request.params.cache
        cache_read = cache_write = five = hour = 0
        if cache.enabled and input_tokens >= 1024:
            cacheable = int(input_tokens * 0.8)
            if seen:
                cache_read = cacheable
            else:
                cache_write = cacheable
                hour, five = (cacheable, 0) if cache.ttl == "1h" else (0, cacheable)
        stop = "max_tokens" if truncated else ("tool_use" if tool_call else "end_turn")
        content: list[dict[str, Any]] = [{"type": "text", "text": "x" * (visible * 4)}]
        if tool_call:
            content.append(
                {"type": "tool_use", "id": "toolu_mock", "name": request.tools[0].name, "input": {}}
            )
        return {
            "id": f"msg_mock_{rng.getrandbits(48):012x}",
            "type": "message",
            "model": request.model,
            "stop_reason": stop,
            "content": content,
            "usage": {
                "input_tokens": input_tokens - cache_read - cache_write,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": five,
                    "ephemeral_1h_input_tokens": hour,
                },
                "output_tokens": visible + reasoning,
                "output_tokens_details": {"thinking_tokens": reasoning},
                "server_tool_use": None,
                "service_tier": "standard",
            },
        }

    def _openai_payload(
        self,
        request: LLMRequest,
        input_tokens: int,
        visible: int,
        reasoning: int,
        truncated: bool,
        seen: bool,
        tool_call: bool,
        rng: random.Random,
    ) -> dict[str, Any]:
        cached = (int(input_tokens * 0.6) // 128) * 128 if seen and input_tokens >= 1024 else 0
        output: list[dict[str, Any]] = [
            {"type": "message", "content": [{"type": "output_text", "text": "x" * (visible * 4)}]}
        ]
        if tool_call:
            output.append(
                {"type": "function_call", "name": request.tools[0].name, "arguments": "{}"}
            )
        return {
            "id": f"resp_mock_{rng.getrandbits(48):012x}",
            "object": "response",
            "model": request.model,
            "status": "incomplete" if truncated else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
            "output": output,
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": cached},
                "output_tokens": visible + reasoning,
                "output_tokens_details": {"reasoning_tokens": reasoning},
                "total_tokens": input_tokens + visible + reasoning,
            },
        }

    def _gemini_payload(
        self,
        request: LLMRequest,
        input_tokens: int,
        visible: int,
        reasoning: int,
        truncated: bool,
        seen: bool,
        tool_call: bool,
        rng: random.Random,
    ) -> dict[str, Any]:
        cached = int(input_tokens * 0.75) if seen and input_tokens >= 1024 else None
        parts: list[dict[str, Any]] = [{"text": "x" * (visible * 4)}]
        if tool_call:
            parts.append({"function_call": {"name": request.tools[0].name, "args": {}}})
        thoughts = reasoning or None
        return {
            "response_id": f"gem_mock_{rng.getrandbits(48):012x}",
            "model_version": request.model,
            "candidates": [
                {
                    "content": {"parts": parts},
                    "finish_reason": "MAX_TOKENS" if truncated else "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": input_tokens,
                "cached_content_token_count": cached,
                "candidates_token_count": visible,
                "thoughts_token_count": thoughts,
                "total_token_count": input_tokens + visible + (thoughts or 0),
            },
        }

    def normalize_usage(self, response: ProviderResponse) -> NormalizedUsage:
        return NORMALIZERS[self.family](response.raw_usage)


class MockAnthropicAdapter(MockAdapter):
    name: ClassVar[str] = "mock-anthropic"
    family: ClassVar[str] = "anthropic"
    api_surface: ClassVar[str] = "mock.messages"


class MockOpenAIAdapter(MockAdapter):
    name: ClassVar[str] = "mock-openai"
    family: ClassVar[str] = "openai"
    api_surface: ClassVar[str] = "mock.responses"


class MockGeminiAdapter(MockAdapter):
    name: ClassVar[str] = "mock-gemini"
    family: ClassVar[str] = "gemini"
    api_surface: ClassVar[str] = "mock.generate_content"
