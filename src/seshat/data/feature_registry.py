"""Feature-role registry and hard leakage guard."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class AvailabilityPhase(StrEnum):
    REQUEST_RAW = "REQUEST_RAW"
    REQUEST_DERIVED = "REQUEST_DERIVED"
    TARGET = "TARGET"
    POST_INFERENCE = "POST_INFERENCE"
    EVALUATION = "EVALUATION"
    IDENTIFIER = "IDENTIFIER"


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    phase: AvailabilityPhase
    allowed_for_training: bool


_REGISTRY = {
    item.name: item
    for item in [
        FeatureDefinition("provider", AvailabilityPhase.REQUEST_RAW, True),
        FeatureDefinition("model_requested", AvailabilityPhase.REQUEST_RAW, True),
        FeatureDefinition("effort", AvailabilityPhase.REQUEST_RAW, True),
        FeatureDefinition("max_output_tokens", AvailabilityPhase.REQUEST_RAW, True),
        FeatureDefinition("temperature", AvailabilityPhase.REQUEST_RAW, True),
        FeatureDefinition("category", AvailabilityPhase.REQUEST_RAW, True),
        FeatureDefinition("char_count", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("word_count", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("line_count", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("sentence_count", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("digit_count", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("punctuation_count", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("code_fence_count", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("avg_word_length", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("log_char_count", AvailabilityPhase.REQUEST_DERIVED, True),
        FeatureDefinition("output_tokens", AvailabilityPhase.TARGET, False),
        FeatureDefinition("input_tokens", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("reasoning_tokens", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("cached_input_tokens", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("total_tokens", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("finish_reason", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("response_char_count", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("latency_ms", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("is_refusal", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("is_truncated", AvailabilityPhase.POST_INFERENCE, False),
        FeatureDefinition("observation_id", AvailabilityPhase.IDENTIFIER, False),
        FeatureDefinition("run_id", AvailabilityPhase.IDENTIFIER, False),
        FeatureDefinition("prompt_id", AvailabilityPhase.IDENTIFIER, False),
        FeatureDefinition("prompt_family_id", AvailabilityPhase.IDENTIFIER, False),
    ]
}

TRAINING_FEATURES = tuple(
    definition.name for definition in _REGISTRY.values() if definition.allowed_for_training
)


def feature_registry() -> dict[str, FeatureDefinition]:
    return dict(_REGISTRY)


def assert_training_features(columns: Iterable[str]) -> None:
    unknown = [column for column in columns if column not in _REGISTRY]
    if unknown:
        raise ValueError(f"Unregistered training features: {sorted(unknown)}")
    forbidden = [column for column in columns if not _REGISTRY[column].allowed_for_training]
    if forbidden:
        raise ValueError(f"Leakage guard blocked non-request-time features: {sorted(forbidden)}")
