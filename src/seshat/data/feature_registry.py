"""Column roles for the canonical dataset and the hard leakage guard.

Every column that can appear in a training dataset has exactly one availability
phase. Only ``REQUEST_RAW``, ``REQUEST_DERIVED`` and ``PREFLIGHT`` columns are known
before inference, and only those (minus a few high-cardinality identifiers) are
training features. ``assert_training_features`` raises if anything else enters ``X``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class AvailabilityPhase(StrEnum):
    REQUEST_RAW = "REQUEST_RAW"  # configuration chosen by the caller
    REQUEST_DERIVED = "REQUEST_DERIVED"  # deterministic function of the request content
    PREFLIGHT = "PREFLIGHT"  # measured before dispatch (token counting)
    TARGET = "TARGET"  # observed labels
    POST_INFERENCE = "POST_INFERENCE"  # known only after the response
    IDENTIFIER = "IDENTIFIER"
    METADATA = "METADATA"  # provenance / bookkeeping; never a feature


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    phase: AvailabilityPhase
    allowed_for_training: bool


_RAW = [
    "provider",
    "requested_model",
    "api_surface",
    "prompt_category",
    "prompt_subcategory",
    "prompt_language",
    "requested_effort",
    "resolved_effort",
    "thinking_mode",
    "thinking_budget_tokens",
    "max_output_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequence_count",
    "response_format",
    "verbosity",
    "tool_choice",
    "parallel_tool_calls",
    "streaming_enabled",
    "caching_enabled",
    "requested_cache_ttl",
    "service_tier",
    "region",
    "routing_mode",
    "context_window_limit",
]
_DERIVED = [
    "prompt_char_count",
    "prompt_word_count",
    "prompt_line_count",
    "prompt_byte_count",
    "system_prompt_char_count",
    "message_count",
    "conversation_turn_count",
    "conversation_char_count",
    "current_user_message_char_count",
    "contains_code",
    "contains_json",
    "contains_markdown",
    "contains_mathematics",
    "requested_output_format",
    "explicit_output_length_hint",
    "tool_count",
    "tool_schema_bytes",
    "context_message_count",
    "context_character_count",
    "image_count",
    "image_total_bytes",
    "document_count",
    "document_page_count",
    "audio_duration_seconds",
    "video_duration_seconds",
    "role_sequence_str",
    "enabled_tool_types_str",
    "modality_types_str",
]
_PREFLIGHT = ["preflight_input_tokens", "preflight_method", "preflight_complete"]
_TARGETS = [
    "actual_input_tokens",
    "actual_output_tokens",
    "actual_reasoning_tokens",
    "actual_total_tokens",
    "actual_cached_input_tokens",
    "actual_cache_creation_tokens",
    "actual_uncached_input_tokens",
    "actual_visible_output_tokens",
    "actual_tool_input_tokens",
    "actual_tool_output_tokens",
    "normalized_total_tokens",
    "calculated_cost_usd",
    "total_latency_ms",
]
_POST = [
    "resolved_model",
    "model_version",
    "provider_total_tokens",
    "response_status",
    "stop_reason",
    "is_truncated",
    "target_is_censored",
    "first_token_latency_ms",
    "client_tool_call_count",
    "server_tool_call_count",
    "retry_count",
    "rate_limit_encountered",
    "error_type",
    "error_code",
    "completed_at_utc",
    "collected_at_utc",
    "response_id",
    "usage_iteration_count",
    "reasoning_included_in_output",
    "provider_reported_cost_usd",
]
_IDS = [
    "sample_id",
    "request_id",
    "run_id",
    "observation_id",
    "prompt_id",
    "prompt_family_id",
    "prompt_template_id",
    "prompt_hash",
    "conversation_hash",
    "conversation_id",
    "parent_request_id",
    "experiment_id",
    "design_cell_id",
    "variation_id",
    "split_group_id",
    "tool_schema_hash",
    "seed",
    "generation_seed",
    "repetition_index",
]
_META = [
    "source_type",
    "data_origin",
    "label_origin",
    "label_quality",
    "dataset_origin",
    "data_license",
    "consent_status",
    "collection_status",
    "timestamp_utc",
    "sdk_version",
    "decision",
    "split",
]

_REGISTRY: dict[str, FeatureDefinition] = {}
for _names, _phase, _allowed in (
    (_RAW, AvailabilityPhase.REQUEST_RAW, True),
    (_DERIVED, AvailabilityPhase.REQUEST_DERIVED, True),
    (_PREFLIGHT, AvailabilityPhase.PREFLIGHT, True),
    (_TARGETS, AvailabilityPhase.TARGET, False),
    (_POST, AvailabilityPhase.POST_INFERENCE, False),
    (_IDS, AvailabilityPhase.IDENTIFIER, False),
    (_META, AvailabilityPhase.METADATA, False),
):
    for _name in _names:
        if _name in _REGISTRY:
            raise AssertionError(f"duplicate registry entry {_name}")
        _REGISTRY[_name] = FeatureDefinition(_name, _phase, _allowed)

TRAINING_FEATURES = tuple(name for name, d in _REGISTRY.items() if d.allowed_for_training)
TARGET_COLUMNS = tuple(_TARGETS)


def feature_registry() -> dict[str, FeatureDefinition]:
    return dict(_REGISTRY)


def phase_of(column: str) -> AvailabilityPhase | None:
    definition = _REGISTRY.get(column)
    return definition.phase if definition else None


def assert_training_features(columns: Iterable[str]) -> None:
    columns = list(columns)
    unknown = [column for column in columns if column not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unregistered training features: {sorted(unknown)}")
    forbidden = [column for column in columns if not _REGISTRY[column].allowed_for_training]
    if forbidden:
        raise ValueError(f"Leakage guard blocked non-request-time features: {sorted(forbidden)}")
