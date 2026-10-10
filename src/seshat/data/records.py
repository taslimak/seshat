"""Versioned interchange records: request features, predictions and observed usage.

These three record types are the contract between Seshat's data sources (controlled
experiments, production logs, imports, synthetic generators), its store and its
predictors:

* ``RequestFeatureRecord`` — what is knowable *before* a request is sent.
* ``PredictionRecord`` — what Seshat predicted. Always ``record_type="prediction"``;
  it can never be read back as an observation.
* ``ObservedUsageRecord`` — what the provider actually reported afterwards.

Each record carries ``record_type`` and ``schema_version`` so stored files stay
interpretable as schemas evolve. Request-feature and observed-usage records also
carry a required ``Provenance`` label.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from seshat.data.models import EffortLevel, NormalizedUsage, RunStatus
from seshat.data.privacy import PromptHashScheme, RetentionPolicy, hash_prompt, redact_text
from seshat.data.provenance import Provenance
from seshat.features import PromptFeatures, extract_prompt_features

SCHEMA_VERSION: Literal["1.0"] = "1.0"
FEATURE_SET_VERSION: Literal["prompt_features/1"] = "prompt_features/1"

MetadataValue = str | int | float | bool | None


class RecordType(StrEnum):
    REQUEST_FEATURES = "request_features"
    PREDICTION = "prediction"
    OBSERVED_USAGE = "observed_usage"


def stable_id(prefix: str, value: str, length: int = 20) -> str:
    """Deterministic identifier: ``<prefix>_<first length hex chars of sha256(value)>``."""
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length]}"


def record_hash(record: BaseModel) -> str:
    """Content hash used to tell idempotent re-ingestion apart from conflicting IDs."""
    payload = record.model_dump(mode="json", exclude={"retain_until"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PromptFeatureValues(BaseModel):
    """Request-time prompt features (mirrors ``seshat.features.PromptFeatures``)."""

    model_config = ConfigDict(extra="forbid")

    char_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    line_count: int = Field(ge=0)
    sentence_count: int = Field(ge=0)
    digit_count: int = Field(ge=0)
    punctuation_count: int = Field(ge=0)
    code_fence_count: int = Field(ge=0)
    avg_word_length: float = Field(ge=0)
    log_char_count: float = Field(ge=0)

    @classmethod
    def from_prompt(cls, prompt: str) -> PromptFeatureValues:
        return cls.model_validate(extract_prompt_features(prompt).__dict__)

    def to_features(self) -> PromptFeatures:
        return PromptFeatures(**self.model_dump())


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = SCHEMA_VERSION


class RequestFeatureRecord(_Record):
    """Everything known about a request before it is executed. No outcomes."""

    record_type: Literal["request_features"] = "request_features"
    request_id: str = Field(min_length=1)
    created_at: AwareDatetime
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    effort: EffortLevel = "auto"
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    category: str | None = None
    message_count: int = Field(default=1, ge=1)
    tools_enabled: bool = False
    prompt_hash: str = Field(min_length=16)
    prompt_hash_scheme: PromptHashScheme
    prompt_family_id: str | None = None
    prompt_text: str | None = None
    feature_set_version: Literal["prompt_features/1"] = FEATURE_SET_VERSION
    features: PromptFeatureValues
    provenance: Provenance
    retain_until: AwareDatetime | None = None

    @classmethod
    def from_prompt(
        cls,
        *,
        request_id: str,
        prompt: str,
        provider: str,
        model: str,
        provenance: Provenance,
        created_at: datetime | None = None,
        policy: RetentionPolicy | None = None,
        hash_key: str | None = None,
        **fields: Any,
    ) -> RequestFeatureRecord:
        """Build a record from prompt text, keeping the text only if policy allows."""
        policy = policy or RetentionPolicy()
        created = created_at or datetime.now(UTC)
        digest, scheme = hash_prompt(prompt, hash_key)
        return cls(
            request_id=request_id,
            created_at=created,
            provider=provider,
            model=model,
            prompt_hash=digest,
            prompt_hash_scheme=scheme,
            prompt_text=prompt if policy.store_prompt_text else None,
            features=PromptFeatureValues.from_prompt(prompt),
            provenance=provenance,
            retain_until=policy.retain_until(provenance, created),
            **fields,
        )


class PredictionRecord(_Record):
    """A Seshat prediction. Never an observation, never a training label."""

    record_type: Literal["prediction"] = "prediction"
    prediction_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    created_at: AwareDatetime
    predictor_version: str = Field(min_length=1)
    target: Literal["output_tokens"] = "output_tokens"
    predicted_output_tokens: int = Field(ge=0)
    lower_bound: int = Field(ge=0)
    upper_bound: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    interval_calibrated: bool = False

    @model_validator(mode="after")
    def bounds_bracket_prediction(self) -> PredictionRecord:
        if not self.lower_bound <= self.predicted_output_tokens <= self.upper_bound:
            raise ValueError("lower_bound <= predicted_output_tokens <= upper_bound must hold")
        return self


class ObservedUsageRecord(_Record):
    """Usage a provider actually reported for one executed request."""

    record_type: Literal["observed_usage"] = "observed_usage"
    observation_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    prediction_id: str | None = None
    observed_at: AwareDatetime
    provider: str = Field(min_length=1)
    model_requested: str = Field(min_length=1)
    model_resolved: str | None = None
    provider_request_id: str | None = None
    status: RunStatus
    usage: NormalizedUsage | None = None
    finish_reason: str | None = None
    is_truncated: bool = False
    is_refusal: bool = False
    latency_ms: float | None = Field(default=None, ge=0)
    response_char_count: int | None = Field(default=None, ge=0)
    attempt_count: int = Field(default=1, ge=1)
    error_type: str | None = None
    error_message: str | None = None
    provenance: Provenance
    retain_until: AwareDatetime | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("error_message")
    @classmethod
    def _redact_error(cls, value: str | None) -> str | None:
        return redact_text(value)

    @field_validator("metadata")
    @classmethod
    def _redact_metadata(cls, value: dict[str, MetadataValue]) -> dict[str, MetadataValue]:
        return {
            key: redact_text(item) if isinstance(item, str) else item
            for key, item in value.items()
        }


AnyRecord = RequestFeatureRecord | PredictionRecord | ObservedUsageRecord

RECORD_MODELS: dict[RecordType, type[AnyRecord]] = {
    RecordType.REQUEST_FEATURES: RequestFeatureRecord,
    RecordType.PREDICTION: PredictionRecord,
    RecordType.OBSERVED_USAGE: ObservedUsageRecord,
}

ID_COLUMNS: dict[RecordType, str] = {
    RecordType.REQUEST_FEATURES: "request_id",
    RecordType.PREDICTION: "prediction_id",
    RecordType.OBSERVED_USAGE: "observation_id",
}

_PROVENANCE_COLUMNS = {
    "data_origin": "data_origin",
    "source": "data_source",
    "generator": "data_generator",
    "experiment_id": "experiment_id",
    "pipeline_version": "pipeline_version",
}


def _flatten_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    return {column: provenance.get(key) for key, column in _PROVENANCE_COLUMNS.items()}


def to_row(record: AnyRecord) -> dict[str, Any]:
    """Flatten a record into one tabular row for Parquet/CSV storage."""
    data = record.model_dump(mode="python")
    row: dict[str, Any] = {}
    if isinstance(record, RequestFeatureRecord):
        features = data.pop("features")
        provenance = data.pop("provenance")
        row.update(data)
        row.update(features)
        row.update(_flatten_provenance(provenance))
    elif isinstance(record, ObservedUsageRecord):
        usage = data.pop("usage") or {}
        provenance = data.pop("provenance")
        metadata = data.pop("metadata")
        row.update(data)
        for column in NormalizedUsage.model_fields:
            row[column] = usage.get(column)
        row.update(_flatten_provenance(provenance))
        row["metadata_json"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    else:
        row.update(data)
    for key, value in list(row.items()):
        if isinstance(value, StrEnum):
            row[key] = value.value
    row["record_hash"] = record_hash(record)
    return row
