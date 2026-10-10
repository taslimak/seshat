import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pytest
from pydantic import ValidationError

from seshat.data.features import extract_request_features
from seshat.data.schemas import (
    SCHEMA_VERSION,
    CollectionStatus,
    ConsentStatus,
    CostMethod,
    LabelQuality,
    LLMRequest,
    ObservationRecord,
    PredictionRecord,
    PreflightMethod,
    PreflightRecord,
    RequestRecord,
    ResponseStatus,
    SourceType,
)
from seshat.data.schemas.arrow import arrow_schema, from_table, to_table

NOW = datetime(2026, 10, 1, 12, tzinfo=UTC)
SID = uuid.uuid4()
RUN = uuid.uuid4()


def _request(**overrides: object) -> LLMRequest:
    payload: dict[str, object] = {
        "provider": "mock-openai",
        "model": "mock-gpt",
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Explain photosynthesis in two sentences."}],
        "params": {"max_output_tokens": 256, "effort": "low"},
    }
    payload.update(overrides)
    return LLMRequest.model_validate(payload)


def request_record(**overrides: object) -> RequestRecord:
    features = extract_request_features(_request()).model_dump()
    values: dict[str, object] = {
        "sample_id": SID,
        "request_id": "req_1",
        "run_id": RUN,
        "timestamp_utc": NOW,
        "provider": "mock-openai",
        "api_surface": "mock.responses",
        "requested_model": "mock-gpt",
        "source_type": SourceType.TEST_FIXTURE,
        "dataset_origin": "unit-test",
        "consent_status": ConsentStatus.NOT_REQUIRED,
        "collection_status": CollectionStatus.COMPLETED,
        "prompt_id": "p1",
        "prompt_category": "question_answering",
        **features,
    }
    values.update(overrides)
    return RequestRecord.model_validate(values)


def observation_record(**overrides: object) -> ObservationRecord:
    values: dict[str, object] = {
        "observation_id": "obs_1",
        "sample_id": SID,
        "request_id": "req_1",
        "run_id": RUN,
        "provider": "mock-openai",
        "requested_model": "mock-gpt",
        "source_type": SourceType.TEST_FIXTURE,
        "dataset_origin": "unit-test",
        "consent_status": ConsentStatus.NOT_REQUIRED,
        "actual_input_tokens": 20,
        "actual_output_tokens": 50,
        "actual_total_tokens": 70,
        "normalized_total_tokens": 70,
        "input_usage_semantics": "x",
        "output_usage_semantics": "y",
        "label_quality": LabelQuality.SYNTHETIC,
        "usage_observation_source": "fixture",
        "raw_usage": {"input_tokens": 20},
        "response_status": ResponseStatus.COMPLETED,
        "calculated_cost_usd": Decimal("0.000123"),
        "cost_method": CostMethod.PRICE_TABLE_ESTIMATE,
        "collected_at_utc": NOW,
    }
    values.update(overrides)
    return ObservationRecord.model_validate(values)


def test_schema_version_is_fixed() -> None:
    assert SCHEMA_VERSION == "1.0.0"
    assert request_record().schema_version == "1.0.0"
    with pytest.raises(ValidationError):
        request_record(schema_version="2.0.0")


def test_llm_request_structure_rules() -> None:
    with pytest.raises(ValidationError, match="system"):
        _request(messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}])
    with pytest.raises(ValidationError, match="last message"):
        _request(messages=[{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    with pytest.raises(ValidationError, match="unique"):
        _request(tools=[{"name": "t"}, {"name": "t"}])
    with pytest.raises(ValidationError):
        _request(params={"max_output_tokens": 0})


def test_records_reject_unknown_fields_and_naive_timestamps() -> None:
    with pytest.raises(ValidationError):
        request_record(unexpected=1)
    with pytest.raises(ValidationError):
        request_record(timestamp_utc=datetime(2026, 1, 1))


def test_observation_provenance_rules() -> None:
    with pytest.raises(ValidationError, match="never has observed usage"):
        observation_record(source_type=SourceType.SYNTHETIC_UNEXECUTED)
    with pytest.raises(ValidationError, match="synthetic"):
        observation_record(label_quality=LabelQuality.VERIFIED)  # fixture claiming a real label
    with pytest.raises(ValidationError, match="real usage sources"):
        observation_record(
            source_type=SourceType.CONTROLLED_EXPERIMENT
        )  # real source, synthetic label
    real = observation_record(
        source_type=SourceType.CONTROLLED_EXPERIMENT, label_quality=LabelQuality.VERIFIED
    )
    assert real.is_real_label
    assert not observation_record().is_real_label


def test_observation_cost_method_consistency_and_nonnegative_tokens() -> None:
    with pytest.raises(ValidationError, match="price_table_estimate"):
        observation_record(calculated_cost_usd=None)
    with pytest.raises(ValidationError, match="provider_reported"):
        observation_record(cost_method=CostMethod.PROVIDER_REPORTED)
    with pytest.raises(ValidationError):
        observation_record(actual_output_tokens=-1)
    unavailable = observation_record(
        actual_input_tokens=None, actual_output_tokens=None, label_quality=LabelQuality.UNAVAILABLE
    )
    assert unavailable.actual_output_tokens is None  # unavailable stays null, never 0


def test_preflight_rules() -> None:
    base = {
        "preflight_id": "pre_1",
        "sample_id": SID,
        "request_id": "req_1",
        "provider": "p",
        "requested_model": "m",
        "preflight_timestamp_utc": NOW,
    }
    with pytest.raises(ValidationError, match="unavailable"):
        PreflightRecord.model_validate(
            {**base, "preflight_method": PreflightMethod.UNAVAILABLE, "preflight_input_tokens": 3}
        )
    with pytest.raises(ValidationError, match="preflight_error"):
        PreflightRecord.model_validate(
            {**base, "preflight_method": PreflightMethod.PROVIDER_NATIVE}
        )
    ok = PreflightRecord.model_validate(
        {**base, "preflight_method": PreflightMethod.LOCAL_TOKENIZER, "preflight_input_tokens": 12}
    )
    assert ok.preflight_input_tokens == 12


def test_prediction_bounds() -> None:
    with pytest.raises(ValidationError):
        PredictionRecord(
            prediction_id="p",
            request_id="r",
            created_at_utc=NOW,
            predictor_version="v",
            predicted_output_tokens=5,
            lower_bound=6,
            upper_bound=9,
            confidence=0.3,
        )


@pytest.mark.parametrize(
    "record", [request_record(), observation_record()], ids=["request", "observation"]
)
def test_arrow_round_trip(record: RequestRecord | ObservationRecord) -> None:
    model = type(record)
    table = to_table([record], model)
    assert table.schema.equals(arrow_schema(model), check_metadata=False)
    assert from_table(table, model) == [record]


def test_arrow_types() -> None:
    schema = arrow_schema(ObservationRecord)
    assert schema.field("calculated_cost_usd").type == pa.decimal128(24, 12)
    assert schema.field("collected_at_utc").type == pa.timestamp("us", tz="UTC")
    assert schema.field("raw_usage_json").type == pa.string()
    request_schema = arrow_schema(RequestRecord)
    assert request_schema.field("role_sequence").type == pa.list_(pa.string())
    assert pa.types.is_struct(request_schema.field("image_metadata").type.value_type)


def test_schema_docs_cover_every_field_and_are_current() -> None:
    from seshat.data.schemas.docs import render, undocumented_fields
    from tests.data.conftest import REPO_ROOT

    assert undocumented_fields() == []
    assert (REPO_ROOT / "docs" / "DATA_SCHEMA.md").read_text(encoding="utf-8") == render(), (
        "regenerate: python -m seshat.data.schemas.docs > docs/DATA_SCHEMA.md"
    )
