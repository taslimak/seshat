"""Canonical dataset records, ``schema_version = "1.0.0"``.

Four tables, joined by stable identifiers (``sample_id`` / ``request_id``):

* ``RequestRecord`` — identity & provenance (A), prompt & conversation features (B),
  execution configuration (C) and context/multimodal features (D). Everything here
  is known **before** the request executes.
* ``PreflightRecord`` — preflight input-token measurements (E), one per method.
* ``ObservationRecord`` — observed usage labels (F), execution results (G) and
  cost/billing metadata (H) for one billable invocation. **Post-inference only.**
* ``PredictionRecord`` — Seshat's own predictions; never an observation.

Unavailable values are ``None``; zero means "observed zero". Field definitions are
documented in ``docs/DATA_SCHEMA.md``.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from seshat.data.schemas.enums import (
    REAL_USAGE_SOURCES,
    SCHEMA_VERSION,
    UNLABELED_SOURCES,
    CollectionStatus,
    ConsentStatus,
    CostMethod,
    LabelQuality,
    ObservationKind,
    PreflightMethod,
    ResponseStatus,
    SourceType,
)

SAMPLE_NAMESPACE = uuid.UUID("6f1d3a52-7c0e-4b8e-9a51-5e5ea7000001")


def stable_uuid(*parts: object) -> uuid.UUID:
    """Deterministic UUIDv5 from parts (idempotent sample registration)."""
    return uuid.uuid5(SAMPLE_NAMESPACE, "|".join(str(part) for part in parts))


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION


class ImageMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mime_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)


class AttachmentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    mime_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    page_count: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)


class RequestRecord(_Record):
    """One planned or executed request: request-time information only."""

    # --- A. Identity and provenance -------------------------------------------
    sample_id: uuid.UUID
    request_id: str = Field(min_length=1)
    run_id: uuid.UUID
    conversation_id: str | None = None
    parent_request_id: str | None = None
    timestamp_utc: AwareDatetime
    provider: str = Field(min_length=1)
    api_surface: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    sdk_version: str | None = None
    source_type: SourceType
    dataset_origin: str = Field(min_length=1)
    data_license: str | None = None
    consent_status: ConsentStatus
    collection_status: CollectionStatus
    experiment_id: str | None = None
    design_cell_id: str | None = None
    variation_id: str | None = None
    repetition_index: int = Field(default=0, ge=0)
    generation_seed: int | None = None

    # --- B. Prompt and conversation features ----------------------------------
    prompt_id: str = Field(min_length=1)
    prompt_template_id: str | None = None
    prompt_family_id: str | None = None
    prompt_category: str = Field(min_length=1)
    prompt_subcategory: str | None = None
    prompt_language: str | None = None
    prompt_char_count: int = Field(ge=0)
    prompt_word_count: int = Field(ge=0)
    prompt_line_count: int = Field(ge=0)
    prompt_byte_count: int = Field(ge=0)
    system_prompt_char_count: int = Field(ge=0)
    message_count: int = Field(ge=1)
    conversation_turn_count: int = Field(ge=1)
    conversation_char_count: int = Field(ge=0)
    current_user_message_char_count: int = Field(ge=0)
    role_sequence: list[str]
    contains_code: bool
    contains_json: bool
    contains_markdown: bool
    contains_mathematics: bool
    requested_output_format: str
    explicit_output_length_hint: str | None = None
    prompt_hash: str = Field(min_length=16)
    conversation_hash: str | None = None
    full_prompt_storage_ref: str | None = None

    # --- C. Request execution configuration -----------------------------------
    max_output_tokens: int | None = Field(default=None, ge=1)
    requested_effort: str | None = None
    resolved_effort: str | None = None
    thinking_mode: str | None = None
    thinking_budget_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop_sequence_count: int = Field(ge=0)
    response_format: str | None = None
    verbosity: str | None = None
    tool_count: int = Field(ge=0)
    tool_choice: str | None = None
    tool_schema_bytes: int = Field(ge=0)
    tool_schema_hash: str | None = None
    enabled_tool_types: list[str]
    parallel_tool_calls: bool | None = None
    streaming_enabled: bool
    caching_enabled: bool
    requested_cache_ttl: str | None = None
    service_tier: str | None = None
    region: str | None = None
    routing_mode: str | None = None
    provider_settings: dict[str, Any] = Field(default_factory=dict)

    # --- D. Context and multimodal features -----------------------------------
    context_window_limit: int | None = Field(default=None, ge=1)
    context_message_count: int = Field(ge=0)
    context_character_count: int = Field(ge=0)
    context_preflight_tokens: int | None = Field(default=None, ge=0)
    image_count: int = Field(ge=0)
    image_total_bytes: int | None = Field(default=None, ge=0)
    image_metadata: list[ImageMetadata] = Field(default_factory=list)
    document_count: int = Field(ge=0)
    document_page_count: int | None = Field(default=None, ge=0)
    audio_duration_seconds: float | None = Field(default=None, ge=0)
    video_duration_seconds: float | None = Field(default=None, ge=0)
    modality_types: list[str]
    attachment_metadata: list[AttachmentMetadata] = Field(default_factory=list)

    # --- Retention ---------------------------------------------------------------
    retain_until: AwareDatetime | None = None


class PreflightRecord(_Record):
    """A preflight input-token measurement. Never authoritative usage."""

    preflight_id: str = Field(min_length=1)
    sample_id: uuid.UUID
    request_id: str = Field(min_length=1)
    provider: str
    requested_model: str
    preflight_input_tokens: int | None = Field(default=None, ge=0)
    preflight_method: PreflightMethod
    preflight_provider: str | None = None
    tokenizer_name: str | None = None
    tokenizer_version: str | None = None
    preflight_timestamp_utc: AwareDatetime | None = None
    preflight_error: str | None = None
    preflight_latency_ms: float | None = Field(default=None, ge=0)
    preflight_includes: list[str] = Field(default_factory=list)
    preflight_complete: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> PreflightRecord:
        if self.preflight_method == PreflightMethod.UNAVAILABLE:
            if self.preflight_input_tokens is not None:
                raise ValueError("unavailable preflight cannot carry a token count")
        elif self.preflight_input_tokens is None and self.preflight_error is None:
            raise ValueError("a preflight without a count must record preflight_error")
        return self


class ObservationRecord(_Record):
    """Observed usage (labels), execution result and cost for one billable call."""

    observation_id: str = Field(min_length=1)
    sample_id: uuid.UUID
    request_id: str = Field(min_length=1)
    run_id: uuid.UUID
    parent_request_id: str | None = None
    observation_kind: ObservationKind = ObservationKind.INVOCATION
    invocation_index: int = Field(default=0, ge=0)
    provider: str
    requested_model: str
    resolved_model: str | None = None
    model_version: str | None = None
    source_type: SourceType
    dataset_origin: str
    consent_status: ConsentStatus

    # --- F. Observed usage (target labels) -------------------------------------
    actual_input_tokens: int | None = Field(default=None, ge=0)
    actual_output_tokens: int | None = Field(default=None, ge=0)
    actual_reasoning_tokens: int | None = Field(default=None, ge=0)
    actual_total_tokens: int | None = Field(default=None, ge=0)
    actual_cached_input_tokens: int | None = Field(default=None, ge=0)
    actual_cache_creation_tokens: int | None = Field(default=None, ge=0)
    actual_uncached_input_tokens: int | None = Field(default=None, ge=0)
    actual_visible_output_tokens: int | None = Field(default=None, ge=0)
    actual_tool_input_tokens: int | None = Field(default=None, ge=0)
    actual_tool_output_tokens: int | None = Field(default=None, ge=0)
    input_usage_semantics: str
    output_usage_semantics: str
    reasoning_included_in_output: bool | None = None
    provider_total_tokens: int | None = Field(default=None, ge=0)
    normalized_total_tokens: int | None = Field(default=None, ge=0)
    usage_iteration_count: int = Field(default=1, ge=0)
    label_quality: LabelQuality
    usage_observation_source: str
    raw_usage: dict[str, Any] = Field(default_factory=dict)

    # --- G. Execution results ----------------------------------------------------
    response_id: str | None = None
    response_status: ResponseStatus
    stop_reason: str | None = None
    is_truncated: bool | None = None
    first_token_latency_ms: float | None = Field(default=None, ge=0)
    total_latency_ms: float | None = Field(default=None, ge=0)
    client_tool_call_count: int | None = Field(default=None, ge=0)
    server_tool_call_count: int | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    rate_limit_encountered: bool = False
    error_type: str | None = None
    error_code: str | None = None
    completed_at_utc: AwareDatetime | None = None

    # --- H. Cost and billing metadata -----------------------------------------
    pricing_version: str | None = None
    pricing_effective_date: date | None = None
    pricing_source: str | None = None
    input_rate_usd_per_million: Decimal | None = None
    output_rate_usd_per_million: Decimal | None = None
    cache_read_rate_usd_per_million: Decimal | None = None
    cache_write_rate_usd_per_million: Decimal | None = None
    calculated_cost_usd: Decimal | None = None
    provider_reported_cost_usd: Decimal | None = None
    currency: str = "USD"
    cost_method: CostMethod = CostMethod.UNAVAILABLE
    billing_dimensions: dict[str, Any] = Field(default_factory=dict)

    collected_at_utc: AwareDatetime
    retain_until: AwareDatetime | None = None

    @model_validator(mode="after")
    def _provenance_rules(self) -> ObservationRecord:
        if self.source_type in UNLABELED_SOURCES:
            raise ValueError(
                f"source_type={self.source_type.value} never has observed usage; "
                "execute the prompt in a controlled experiment to obtain labels"
            )
        if self.source_type == SourceType.TEST_FIXTURE and (
            self.label_quality not in {LabelQuality.SYNTHETIC, LabelQuality.UNAVAILABLE}
        ):
            raise ValueError("test-fixture usage must be labelled synthetic")
        if self.source_type in REAL_USAGE_SOURCES and self.label_quality == LabelQuality.SYNTHETIC:
            raise ValueError("real usage sources cannot carry synthetic labels")
        if (
            self.cost_method == CostMethod.PROVIDER_REPORTED
            and self.provider_reported_cost_usd is None
        ):
            raise ValueError("cost_method=provider_reported requires provider_reported_cost_usd")
        if self.cost_method == CostMethod.PRICE_TABLE_ESTIMATE and self.calculated_cost_usd is None:
            raise ValueError("cost_method=price_table_estimate requires calculated_cost_usd")
        return self

    @property
    def is_real_label(self) -> bool:
        return self.source_type in REAL_USAGE_SOURCES and self.label_quality in {
            LabelQuality.VERIFIED,
            LabelQuality.PARTIAL,
        }


class PredictionRecord(_Record):
    """A Seshat prediction for a request. Never an observation or a label."""

    prediction_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    created_at_utc: AwareDatetime
    predictor_version: str = Field(min_length=1)
    target: Literal["output_tokens"] = "output_tokens"
    predicted_output_tokens: int = Field(ge=0)
    lower_bound: int = Field(ge=0)
    upper_bound: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    interval_calibrated: bool = False
    # Forecast-quality lineage (optional; absent on records written before 2026-10-11).
    prediction_mode: str | None = None
    evaluation_role: str | None = None
    base_predicted_output_tokens: int | None = Field(default=None, ge=0)
    confidence_level: float | None = Field(default=None, gt=0.0, lt=1.0)
    lower_margin_tokens: int | None = Field(default=None, ge=0)
    upper_margin_tokens: int | None = Field(default=None, ge=0)
    confidence_margin_tokens: int | None = Field(default=None, ge=0)
    lower_margin_pct: float | None = None
    upper_margin_pct: float | None = None
    confidence_margin_pct: float | None = None
    predicted_reasoning_tokens: int | None = Field(default=None, ge=0)
    predicted_total_tokens: int | None = Field(default=None, ge=0)
    predicted_cost_usd: Decimal | None = Field(default=None, ge=0)
    uncertainty_model_version: str | None = None
    calibration_version: str | None = None
    dataset_version: str | None = None
    feature_set_version: str | None = None
    code_revision: str | None = None
    expected_absolute_error_tokens: float | None = Field(default=None, ge=0)
    preflight_input_tokens: int | None = Field(default=None, ge=0)
    preflight_method: str | None = None
    pricing_version: str | None = None
    artifact_sha256: str | None = None

    @model_validator(mode="after")
    def _bounds(self) -> PredictionRecord:
        if not self.lower_bound <= self.predicted_output_tokens <= self.upper_bound:
            raise ValueError("lower_bound <= predicted_output_tokens <= upper_bound must hold")
        return self
