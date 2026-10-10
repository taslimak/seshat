"""Canonical schemas for controlled experiment data."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

EffortLevel = Literal["auto", "none", "low", "medium", "high", "xhigh", "max"]


class RunStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"


class QualityDecision(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


class PromptRecord(BaseModel):
    prompt_id: str = Field(min_length=1)
    prompt_family_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    language: str = "en"
    complexity: Literal["low", "medium", "high"] = "medium"
    is_synthetic: bool = False
    license_id: str | None = None
    commercial_use_allowed: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _default_efforts() -> list[EffortLevel]:
    return ["auto"]


class ProviderExperimentConfig(BaseModel):
    models: list[str] = Field(min_length=1)
    efforts: list[EffortLevel] = Field(default_factory=_default_efforts)
    max_output_tokens: int = Field(default=2048, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)


class ExperimentConfig(BaseModel):
    experiment_id: str = Field(min_length=1)
    prompt_sources: list[str] = Field(min_length=1)
    providers: dict[str, ProviderExperimentConfig] = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1)
    max_requests: int | None = Field(default=None, ge=1)
    seed: int = 1729
    sample_models_per_prompt: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManifestRow(BaseModel):
    run_id: str
    condition_id: str
    experiment_id: str
    prompt_id: str
    prompt_family_id: str
    category: str
    provider: str
    model: str
    effort: EffortLevel
    repetition: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    temperature: float | None = None


class NormalizedUsage(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)
    tool_prompt_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    billable_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def totals_are_plausible(self) -> NormalizedUsage:
        if (
            self.total_tokens is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
            and self.total_tokens < self.input_tokens + self.output_tokens
        ):
            raise ValueError("total_tokens cannot be smaller than input_tokens + output_tokens")
        return self


class ProviderResult(BaseModel):
    provider_request_id: str | None = None
    model_resolved: str | None = None
    usage: NormalizedUsage
    provider_usage_raw: dict[str, Any] = Field(default_factory=dict)
    provider_response_raw: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None
    response_text: str | None = None
    latency_ms: float = Field(ge=0)
    is_refusal: bool = False
    is_truncated: bool = False


class RawObservation(BaseModel):
    observation_id: str
    run_id: str
    condition_id: str
    experiment_id: str
    prompt_id: str
    prompt_family_id: str
    category: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provider: str
    model_requested: str
    model_resolved: str | None = None
    effort: EffortLevel
    repetition: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    temperature: float | None = None
    request_features: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
    provider_usage_raw: dict[str, Any] = Field(default_factory=dict)
    normalized_usage: NormalizedUsage | None = None
    provider_response_raw: dict[str, Any] = Field(default_factory=dict)
    provider_request_id: str | None = None
    finish_reason: str | None = None
    response_char_count: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    is_refusal: bool = False
    is_truncated: bool = False
    status: RunStatus
    attempt_count: int = Field(default=1, ge=1)
    error_type: str | None = None
    error_message: str | None = None


class QualityFlag(BaseModel):
    code: str
    message: str
    severity: QualityDecision
