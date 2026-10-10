from datetime import UTC, datetime

import pytest
from helpers import REAL_PRODUCTION, observation_payload
from pydantic import ValidationError

from seshat.data.privacy import RetentionPolicy
from seshat.data.provenance import DataOrigin, DataSource, Provenance
from seshat.data.records import (
    ObservedUsageRecord,
    PredictionRecord,
    RequestFeatureRecord,
    record_hash,
    stable_id,
    to_row,
)
from seshat.features import extract_prompt_features

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _request(policy: RetentionPolicy | None = None) -> RequestFeatureRecord:
    return RequestFeatureRecord.from_prompt(
        request_id="req_1",
        prompt="Write a haiku about tokens.",
        provider="openai",
        model="model-a",
        provenance=REAL_PRODUCTION,
        created_at=NOW,
        policy=policy,
        hash_key="",
    )


def test_request_record_keeps_hash_and_features_but_not_text_by_default() -> None:
    record = _request()
    assert record.prompt_text is None
    assert len(record.prompt_hash) == 64
    assert record.features.to_features() == extract_prompt_features("Write a haiku about tokens.")
    assert record.retain_until is not None and record.retain_until > NOW


def test_request_record_keeps_text_only_when_policy_allows() -> None:
    assert _request(RetentionPolicy(store_prompt_text=True)).prompt_text is not None


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        ObservedUsageRecord.model_validate(observation_payload(observed_at="2026-09-01T00:00:00"))


def test_unknown_fields_and_schema_versions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ObservedUsageRecord.model_validate(observation_payload(unexpected="x"))
    with pytest.raises(ValidationError):
        ObservedUsageRecord.model_validate(observation_payload(schema_version="2.0"))


def test_prediction_can_never_be_parsed_as_an_observation() -> None:
    prediction = PredictionRecord(
        prediction_id="pred_1",
        request_id="req_1",
        created_at=NOW,
        predictor_version="baseline-0.1",
        predicted_output_tokens=100,
        lower_bound=50,
        upper_bound=150,
        confidence=0.35,
    )
    with pytest.raises(ValidationError):
        ObservedUsageRecord.model_validate(prediction.model_dump(mode="json"))


def test_prediction_bounds_must_bracket_the_point_estimate() -> None:
    with pytest.raises(ValidationError, match="lower_bound"):
        PredictionRecord(
            prediction_id="pred_1",
            request_id="req_1",
            created_at=NOW,
            predictor_version="v",
            predicted_output_tokens=10,
            lower_bound=20,
            upper_bound=30,
            confidence=0.5,
        )


def test_free_text_is_redacted_on_observations() -> None:
    record = ObservedUsageRecord.model_validate(
        observation_payload(
            status="failed",
            error_message="quota exceeded for ops@example.com",
            metadata={"user": "ops@example.com", "tier": 2},
        )
    )
    assert record.error_message is not None and "@" not in record.error_message
    assert record.metadata == {"user": "[REDACTED_EMAIL]", "tier": 2}


def test_rows_are_flat_and_hash_ignores_retention() -> None:
    record = ObservedUsageRecord.model_validate(observation_payload())
    row = to_row(record)
    assert row["output_tokens"] == 20
    assert row["data_origin"] == "real"
    assert row["data_source"] == "production_log"
    assert "usage" not in row and "provenance" not in row
    later = record.model_copy(update={"retain_until": NOW})
    assert record_hash(later) == record_hash(record)


def test_stable_id_is_deterministic() -> None:
    assert stable_id("obs", "run_1") == stable_id("obs", "run_1")
    assert stable_id("obs", "run_1") != stable_id("obs", "run_2")


def test_synthetic_provenance_round_trips() -> None:
    provenance = Provenance(data_origin=DataOrigin.SYNTHETIC, source=DataSource.SIMULATION)
    record = ObservedUsageRecord.model_validate(
        observation_payload(provenance=provenance.model_dump(mode="json"))
    )
    assert record.provenance.data_origin == DataOrigin.SYNTHETIC
