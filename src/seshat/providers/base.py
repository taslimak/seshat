"""Provider-neutral adapter contract.

Every provider implements five operations:

* ``list_supported_models()`` — model IDs from ``configs/model_catalog.json``.
* ``get_capabilities(model)`` — which settings the model accepts. Requests using an
  unsupported setting are rejected, never silently coerced.
* ``count_input_tokens(request)`` — a preflight measurement (provider-native when
  available, otherwise a clearly-labelled local estimate).
* ``execute_request(request)`` — one billable invocation; returns the raw payload.
* ``normalize_usage(response)`` — provider-specific usage normalization
  (``seshat.providers.normalization``).

Errors are classified into ``ProviderError`` so the collector can decide whether a
retry is safe. Timeouts and dropped connections *after* dispatch are
``outcome_uncertain``: the request may have executed and been billed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from seshat.data.provenance import DataOrigin
from seshat.data.schemas.enums import PreflightMethod, ResponseStatus
from seshat.data.schemas.request import LLMRequest
from seshat.providers.normalization import NormalizedUsage


class ErrorCategory(StrEnum):
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


_RETRYABLE = {ErrorCategory.RATE_LIMIT, ErrorCategory.OVERLOADED, ErrorCategory.SERVER_ERROR}
_UNCERTAIN = {ErrorCategory.TIMEOUT, ErrorCategory.CONNECTION, ErrorCategory.UNKNOWN}
_FATAL = {ErrorCategory.AUTHENTICATION, ErrorCategory.PERMISSION}


class ProviderError(Exception):
    """A classified provider failure."""

    def __init__(
        self,
        category: ErrorCategory,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.code = code
        self.retry_after_seconds = retry_after_seconds

    @property
    def retryable(self) -> bool:
        """Safe to retry: the provider rejected the call before generating."""
        return self.category in _RETRYABLE

    @property
    def outcome_uncertain(self) -> bool:
        """The request may have executed (and been billed); do not blindly repeat."""
        return self.category in _UNCERTAIN

    @property
    def fatal(self) -> bool:
        """Configuration problem affecting every request (stop the run)."""
        return self.category in _FATAL


def classify_status(status_code: int | None) -> ErrorCategory:
    if status_code is None:
        return ErrorCategory.UNKNOWN
    if status_code == 429:
        return ErrorCategory.RATE_LIMIT
    if status_code in {503, 529}:
        return ErrorCategory.OVERLOADED
    if status_code in {408, 504}:
        return ErrorCategory.TIMEOUT
    if status_code >= 500:
        return ErrorCategory.SERVER_ERROR
    if status_code == 401:
        return ErrorCategory.AUTHENTICATION
    if status_code == 403:
        return ErrorCategory.PERMISSION
    if status_code == 404:
        return ErrorCategory.NOT_FOUND
    if 400 <= status_code < 500:
        return ErrorCategory.INVALID_REQUEST
    return ErrorCategory.UNKNOWN


@dataclass(frozen=True)
class ModelCapabilities:
    """What a model accepts. ``None`` collections mean the control does not exist."""

    provider: str
    model: str
    verified: bool = False
    context_window: int | None = None
    max_output_tokens: int | None = None
    efforts: frozenset[str] | None = None
    thinking_modes: frozenset[str] | None = None
    supports_temperature: bool = False
    supports_top_p: bool = False
    supports_top_k: bool = False
    supports_seed: bool = False
    supports_stop_sequences: bool = False
    supports_tools: bool = True
    supports_parallel_tool_calls: bool = False
    supports_verbosity: bool = False
    supports_response_format: bool = False
    supports_caching: bool = False
    cache_ttls: frozenset[str] = field(default_factory=frozenset)
    native_token_counting: bool = False
    notes: str | None = None

    def unsupported_settings(self, request: LLMRequest) -> list[str]:
        """Settings in ``request`` this model does not accept (empty when valid)."""
        params = request.params
        problems: list[str] = []
        if params.effort is not None:
            if self.efforts is None:
                problems.append(f"effort is not supported by {self.model}")
            elif params.effort not in self.efforts:
                problems.append(f"effort={params.effort!r} not in {sorted(self.efforts)}")
        if params.thinking_mode is not None:
            if self.thinking_modes is None:
                problems.append(f"thinking_mode is not supported by {self.model}")
            elif params.thinking_mode not in self.thinking_modes:
                problems.append(
                    f"thinking_mode={params.thinking_mode!r} not in {sorted(self.thinking_modes)}"
                )
        checks = (
            ("temperature", params.temperature is not None, self.supports_temperature),
            ("top_p", params.top_p is not None, self.supports_top_p),
            ("top_k", params.top_k is not None, self.supports_top_k),
            ("seed", params.seed is not None, self.supports_seed),
            ("stop_sequences", bool(params.stop_sequences), self.supports_stop_sequences),
            ("tools", bool(request.tools), self.supports_tools),
            (
                "parallel_tool_calls",
                params.parallel_tool_calls is not None,
                self.supports_parallel_tool_calls,
            ),
            ("verbosity", params.verbosity is not None, self.supports_verbosity),
            ("response_format", params.response_format is not None, self.supports_response_format),
            ("cache", params.cache.enabled, self.supports_caching),
        )
        problems.extend(
            f"{name} is not supported by {self.model}"
            for name, used, ok in checks
            if used and not ok
        )
        if (
            params.cache.enabled
            and params.cache.ttl
            and self.cache_ttls
            and params.cache.ttl not in self.cache_ttls
        ):
            problems.append(f"cache ttl {params.cache.ttl!r} not in {sorted(self.cache_ttls)}")
        if (
            self.max_output_tokens is not None
            and params.max_output_tokens is not None
            and params.max_output_tokens > self.max_output_tokens
        ):
            problems.append(f"max_output_tokens exceeds model limit {self.max_output_tokens}")
        return problems


class PreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None
    method: PreflightMethod
    preflight_provider: str | None = None
    tokenizer_name: str | None = None
    tokenizer_version: str | None = None
    timestamp_utc: datetime
    error: str | None = None
    latency_ms: float | None = None
    includes: list[str] = []
    complete: bool = False


class ProviderResponse(BaseModel):
    """Raw outcome of one invocation, before normalization."""

    model_config = ConfigDict(extra="forbid")

    response_id: str | None = None
    resolved_model: str | None = None
    model_version: str | None = None
    status: ResponseStatus
    stop_reason: str | None = None
    is_truncated: bool | None = None
    latency_ms: float | None = None
    first_token_latency_ms: float | None = None
    client_tool_call_count: int | None = None
    server_tool_call_count: int | None = None
    output_char_count: int | None = None
    raw_usage: dict[str, Any] = {}
    raw_response: dict[str, Any] = {}
    provider_reported_cost_usd: str | None = None
    completed_at_utc: datetime | None = None


class ProviderAdapter(ABC):
    """Base class for provider adapters (real and mock)."""

    name: ClassVar[str]
    family: ClassVar[str]
    api_surface: ClassVar[str]
    data_origin: ClassVar[DataOrigin] = DataOrigin.REAL
    makes_external_calls: ClassVar[bool] = True

    def sdk_version(self) -> str | None:
        return None

    def list_supported_models(self) -> list[str]:
        from seshat.providers.catalog import load_catalog

        return sorted(load_catalog().models_for(self.name))

    def get_capabilities(self, model: str) -> ModelCapabilities:
        from seshat.providers.catalog import load_catalog

        return load_catalog().capabilities(self.name, model)

    @abstractmethod
    def count_input_tokens(self, request: LLMRequest) -> PreflightResult:
        """Preflight input-token measurement for the full request."""

    @abstractmethod
    def execute_request(self, request: LLMRequest, *, timeout_seconds: float) -> ProviderResponse:
        """Send one request. Raise ``ProviderError`` on failure."""

    @abstractmethod
    def normalize_usage(self, response: ProviderResponse) -> NormalizedUsage:
        """Provider-specific usage normalization of ``response.raw_usage``."""


def unavailable_preflight(provider: str, reason: str) -> PreflightResult:
    return PreflightResult(
        input_tokens=None,
        method=PreflightMethod.UNAVAILABLE,
        preflight_provider=provider,
        timestamp_utc=datetime.now(UTC),
        error=reason,
    )


def approximate_preflight(request: LLMRequest, provider: str) -> PreflightResult:
    """Statistical approximation (~4 characters per token). Never authoritative."""
    import json

    started = time.perf_counter()
    tools = json.dumps([tool.model_dump() for tool in request.tools]) if request.tools else ""
    chars = len(request.all_text()) + len(tools)
    estimate = max(1, round(chars / 4)) + 4 * len(request.messages)
    return PreflightResult(
        input_tokens=estimate,
        method=PreflightMethod.STATISTICAL_APPROXIMATION,
        preflight_provider=provider,
        tokenizer_name="chars_div_4_plus_message_overhead",
        tokenizer_version="1",
        timestamp_utc=datetime.now(UTC),
        latency_ms=(time.perf_counter() - started) * 1000,
        includes=["system", "messages", "tools"] if request.tools else ["system", "messages"],
        complete=False,
    )
