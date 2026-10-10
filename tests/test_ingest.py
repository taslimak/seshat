import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from helpers import REAL_PRODUCTION, observation_payload, request_payload

from seshat.data.ingest import ingest_jsonl, ingest_records
from seshat.data.privacy import RetentionPolicy
from seshat.data.provenance import DataOrigin, DataSource, Provenance
from seshat.data.records import RecordType
from seshat.data.store import LOG_FILE, DatasetStore


def _ingest(store: DatasetStore, record_type: RecordType, *payloads: Any, **kwargs: Any) -> Any:
    return ingest_records(
        store,
        record_type,
        ((f"test:{index}", payload) for index, payload in enumerate(payloads)),
        source_name="test",
        **kwargs,
    )


def _quarantine(report: Any) -> list[dict[str, Any]]:
    text = Path(report.quarantine_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_valid_observation_is_stored_with_quality_and_provenance(store: DatasetStore) -> None:
    report = _ingest(store, RecordType.OBSERVED_USAGE, observation_payload())
    assert (report.stored, report.quarantined) == (1, 0)
    row = store.read(RecordType.OBSERVED_USAGE).iloc[0]
    assert row["quality_decision"] == "PASS"
    assert row["data_origin"] == "real"
    assert row["retain_until"] > row["observed_at"]


def test_missing_provenance_is_quarantined_never_assumed_real(store: DatasetStore) -> None:
    payload = observation_payload()
    del payload["provenance"]
    report = _ingest(store, RecordType.OBSERVED_USAGE, payload)
    assert report.stored == 0
    assert report.quarantine_reasons == {"missing_provenance": 1}


def test_default_provenance_is_applied_only_when_explicit(store: DatasetStore) -> None:
    payload = observation_payload()
    del payload["provenance"]
    synthetic = Provenance(data_origin=DataOrigin.SYNTHETIC, source=DataSource.IMPORT)
    report = _ingest(store, RecordType.OBSERVED_USAGE, payload, default_provenance=synthetic)
    assert report.stored == 1
    assert store.read(RecordType.OBSERVED_USAGE).iloc[0]["data_origin"] == "synthetic"


def test_mock_source_labelled_real_is_a_schema_error(store: DatasetStore) -> None:
    payload = observation_payload(provenance={"data_origin": "real", "source": "mock_adapter"})
    report = _ingest(store, RecordType.OBSERVED_USAGE, payload)
    assert report.quarantine_reasons == {"schema_error": 1}


def test_reingesting_identical_records_is_idempotent(store: DatasetStore) -> None:
    _ingest(store, RecordType.OBSERVED_USAGE, observation_payload())
    report = _ingest(store, RecordType.OBSERVED_USAGE, observation_payload())
    assert (report.stored, report.duplicates, report.quarantined) == (0, 1, 0)
    assert len(store.read(RecordType.OBSERVED_USAGE)) == 1


def test_same_id_with_different_content_is_quarantined(store: DatasetStore) -> None:
    _ingest(store, RecordType.OBSERVED_USAGE, observation_payload())
    changed = observation_payload(usage={"input_tokens": 10, "output_tokens": 99})
    report = _ingest(store, RecordType.OBSERVED_USAGE, changed)
    assert report.quarantine_reasons == {"conflicting_duplicate_id": 1}


def test_reused_provider_request_id_is_quarantined(store: DatasetStore) -> None:
    first = observation_payload()
    second = observation_payload(observation_id="obs_2", request_id="req_2")
    report = _ingest(store, RecordType.OBSERVED_USAGE, first, second)
    assert report.stored == 1
    assert report.quarantine_reasons == {"duplicate_provider_request_id": 1}


def test_quality_failures_are_quarantined_with_flags(store: DatasetStore) -> None:
    zero = observation_payload(
        observation_id="obs_zero", provider_request_id="r0", usage={"output_tokens": 0}
    )
    # Impossible totals are preserved and quarantined rather than lost at parse time.
    inconsistent = observation_payload(
        observation_id="obs_total",
        provider_request_id="r1",
        usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 120},
    )
    failed = observation_payload(
        observation_id="obs_failed", provider_request_id="r2", status="failed", usage=None
    )
    truncated = observation_payload(
        observation_id="obs_trunc", provider_request_id="r3", is_truncated=True
    )
    report = _ingest(store, RecordType.OBSERVED_USAGE, zero, inconsistent, failed, truncated)
    assert report.stored == 1
    assert report.quality_decisions == {"WARNING": 1}
    assert report.quarantine_reasons == {"quality_quarantine": 3}
    codes = {
        entry["record_id"]: {flag["code"] for flag in entry["flags"]}
        for entry in _quarantine(report)
    }
    assert "zero_output_tokens" in codes["obs_zero"]
    assert "inconsistent_total_tokens" in codes["obs_total"]
    assert "failed_run" in codes["obs_failed"]


def test_prompt_text_becomes_hash_and_features_then_is_dropped(store: DatasetStore) -> None:
    report = _ingest(store, RecordType.REQUEST_FEATURES, request_payload())
    assert (report.stored, report.prompt_text_dropped) == (1, 1)
    row = store.read(RecordType.REQUEST_FEATURES).iloc[0]
    assert row["prompt_text"] is None
    assert len(row["prompt_hash"]) == 64
    assert row["word_count"] == 6
    stored_bytes = b"".join(path.read_bytes() for path in store.parts(RecordType.REQUEST_FEATURES))
    assert b"jane@example.com" not in stored_bytes


def test_quarantine_files_never_contain_prompt_text(store: DatasetStore) -> None:
    bad = request_payload(effort="extreme")
    report = _ingest(store, RecordType.REQUEST_FEATURES, bad)
    assert report.quarantine_reasons == {"schema_error": 1}
    raw = Path(report.quarantine_path).read_text(encoding="utf-8")
    assert "Summarize the attached report" not in raw
    assert "jane@example.com" not in raw
    assert _quarantine(report)[0]["errors"][0]["loc"] == "effort"


def test_prompt_text_kept_when_policy_allows(store: DatasetStore) -> None:
    _ingest(
        store,
        RecordType.REQUEST_FEATURES,
        request_payload(),
        policy=RetentionPolicy(store_prompt_text=True),
    )
    assert store.read(RecordType.REQUEST_FEATURES).iloc[0]["prompt_text"].startswith("Summarize")


def test_jsonl_ingestion_handles_bad_lines_and_logs_batches(
    store: DatasetStore, tmp_path: Path
) -> None:
    source = tmp_path / "usage.jsonl"
    source.write_text(
        json.dumps(observation_payload()) + "\n" + "{not json\n" + "[1, 2]\n", encoding="utf-8"
    )
    report = ingest_jsonl(store, RecordType.OBSERVED_USAGE, source)
    assert report.read == 3 and report.stored == 1
    assert report.quarantine_reasons == {"invalid_json": 1, "invalid_record": 1}
    log = [json.loads(line) for line in (store.root / LOG_FILE).read_text().splitlines()]
    assert log[-1]["event"] == "ingest" and log[-1]["stored"] == 1


def test_wrong_record_type_is_quarantined(store: DatasetStore) -> None:
    report = _ingest(store, RecordType.PREDICTION, {"record_type": "observed_usage"})
    assert report.quarantine_reasons == {"wrong_record_type": 1}


def test_purge_removes_expired_records_and_their_predictions(store: DatasetStore) -> None:
    _ingest(store, RecordType.REQUEST_FEATURES, request_payload())
    _ingest(store, RecordType.OBSERVED_USAGE, observation_payload())
    synthetic = Provenance(data_origin=DataOrigin.SYNTHETIC, source=DataSource.SIMULATION)
    _ingest(
        store,
        RecordType.OBSERVED_USAGE,
        observation_payload(
            observation_id="obs_syn",
            request_id="req_syn",
            provider_request_id="r_syn",
            provenance=synthetic.model_dump(mode="json"),
        ),
    )
    _ingest(
        store,
        RecordType.PREDICTION,
        {
            "prediction_id": "pred_1",
            "request_id": "req_1",
            "created_at": "2026-09-01T00:00:00+00:00",
            "predictor_version": "v",
            "predicted_output_tokens": 10,
            "lower_bound": 5,
            "upper_bound": 15,
            "confidence": 0.3,
        },
    )
    assert store.purge_expired(datetime(2026, 9, 2, tzinfo=UTC)) == {
        "request_features": 0,
        "observed_usage": 0,
        "prediction": 0,
    }
    removed = store.purge_expired(datetime(2027, 1, 1, tzinfo=UTC))
    assert removed == {"request_features": 1, "observed_usage": 1, "prediction": 1}
    remaining = store.read(RecordType.OBSERVED_USAGE)
    assert remaining["observation_id"].tolist() == ["obs_syn"]
    assert store.read(RecordType.REQUEST_FEATURES).empty


def test_real_production_policy_days_are_configurable(store: DatasetStore) -> None:
    _ingest(
        store,
        RecordType.OBSERVED_USAGE,
        observation_payload(),
        policy=RetentionPolicy(production_log_days=1),
    )
    removed = store.purge_expired(datetime(2026, 9, 3, tzinfo=UTC))
    assert removed["observed_usage"] == 1


def test_batches_are_immutable(store: DatasetStore) -> None:
    import pytest

    rows = [{"observation_id": "obs_1", "record_hash": "h"}]
    store.write_batch(RecordType.OBSERVED_USAGE, rows, "b1")
    with pytest.raises(FileExistsError):
        store.write_batch(RecordType.OBSERVED_USAGE, rows, "b1")
    assert REAL_PRODUCTION.data_origin == DataOrigin.REAL
