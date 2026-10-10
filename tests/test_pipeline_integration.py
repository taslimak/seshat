"""Integration of the record store with the experiment pipeline and predictor."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from seshat.cli import main
from seshat.data.build_features import build_feature_frame, store_usage_frame
from seshat.data.compare import compare_predictions, prediction_records, summarize_comparison
from seshat.data.ingest import ingest_experiment, ingest_records
from seshat.data.io import append_jsonl, load_jsonl
from seshat.data.models import PromptRecord, RawObservation
from seshat.data.normalize import normalize_raw_observations
from seshat.data.records import RecordType
from seshat.data.release import freeze_release
from seshat.data.split import split_to_parquet
from seshat.data.store import DatasetStore
from seshat.data.synthetic import generate_synthetic_records
from seshat.data.validate import validate_dataframe
from seshat.experiments.manifest import build_manifest_from_config
from seshat.experiments.runner import run_manifest
from seshat.predictor import BaselinePredictor
from seshat.providers.base import ProviderAdapter
from seshat.providers.mock import MockAdapter
from seshat.providers.registry import PROVIDERS, ProviderCallsNotAllowedError

REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest(provider: str = "mock", rows: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": f"run_{index}",
                "condition_id": f"cond_{index}",
                "experiment_id": "exp",
                "prompt_id": "p_0001",
                "prompt_family_id": "pf_0001",
                "category": "question_answering",
                "provider": provider,
                "model": "m",
                "effort": "auto",
                "repetition": 1,
                "max_output_tokens": 512,
                "temperature": None,
            }
            for index in range(rows)
        ]
    )


def _synthetic_store(store: DatasetStore, prompts: list[PromptRecord], n: int = 40) -> None:
    requests, observations = generate_synthetic_records(prompts, n=n, seed=7)
    for record_type, records in (
        (RecordType.REQUEST_FEATURES, requests),
        (RecordType.OBSERVED_USAGE, observations),
    ):
        ingest_records(
            store,
            record_type,
            ((str(index), record.model_dump(mode="json")) for index, record in enumerate(records)),
            source_name="synthetic",
        )


# --- provider safety gate -------------------------------------------------------


def test_real_providers_are_blocked_before_any_call(
    prompts: list[PromptRecord], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SESHAT_ALLOW_PROVIDER_CALLS", raising=False)
    raw = tmp_path / "raw.jsonl"
    with pytest.raises(ProviderCallsNotAllowedError, match="openai"):
        run_manifest(_manifest("openai"), prompts, raw)
    assert not raw.exists()


def test_external_provider_runs_when_explicitly_allowed(
    prompts: list[PromptRecord], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeExternal(MockAdapter):
        makes_external_calls = True

    monkeypatch.setitem(PROVIDERS, "fake", FakeExternal)
    monkeypatch.setenv("SESHAT_ALLOW_PROVIDER_CALLS", "1")
    assert run_manifest(_manifest("fake"), prompts, tmp_path / "raw.jsonl") == 1


# --- retries (B-05) -------------------------------------------------------------


def test_successful_retry_supersedes_failure(
    prompts: list[PromptRecord], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(ProviderAdapter):
        name = "mock"
        makes_external_calls = False

        def execute(self, **_: Any) -> Any:
            raise ValueError("bad request")

    raw = tmp_path / "raw.jsonl"
    monkeypatch.setitem(PROVIDERS, "mock", Broken)
    run_manifest(_manifest(), prompts, raw, sleep=lambda _: None)
    monkeypatch.setitem(PROVIDERS, "mock", MockAdapter)
    run_manifest(_manifest(), prompts, raw, retry_failed=True, sleep=lambda _: None)
    assert len(load_jsonl(raw, RawObservation)) == 2  # raw stays append-only
    validated = validate_dataframe(normalize_raw_observations(raw))
    assert validated[["status", "quality_decision", "raw_record_count"]].to_dict("records") == [
        {"status": "success", "quality_decision": "PASS", "raw_record_count": 2}
    ]


# --- provenance through the experiment path ------------------------------------


def test_mock_raw_observations_are_labelled_synthetic(
    prompts: list[PromptRecord], tmp_path: Path
) -> None:
    raw = tmp_path / "raw.jsonl"
    run_manifest(_manifest(rows=2), prompts, raw)
    observations = load_jsonl(raw, RawObservation)
    assert {item.provenance.data_origin for item in observations if item.provenance} == {
        "synthetic"
    }
    assert set(normalize_raw_observations(raw)["data_origin"]) == {"synthetic"}


def test_legacy_raw_without_provenance_is_inferred_from_adapter(
    prompts: list[PromptRecord], tmp_path: Path
) -> None:
    raw = tmp_path / "raw.jsonl"
    run_manifest(_manifest(), prompts, raw)
    legacy = tmp_path / "legacy.jsonl"
    for observation in load_jsonl(raw, RawObservation):
        payload = observation.model_dump(mode="json")
        del payload["provenance"]
        append_jsonl(legacy, payload)
    assert normalize_raw_observations(legacy)["data_origin"].tolist() == ["synthetic"]


# --- label-origin policy -------------------------------------------------------


def _usage_rows() -> pd.DataFrame:
    base = {
        "provider": "p",
        "model_requested": "m",
        "effort": "low",
        "category": "c",
        "prompt_family_id": "pf",
        "word_count": 3,
        "quality_decision": "PASS",
    }
    return pd.DataFrame(
        [
            {**base, "observation_id": "o1", "output_tokens": 10, "data_origin": "real"},
            {**base, "observation_id": "o2", "output_tokens": 99, "data_origin": "synthetic"},
        ]
    )


def test_feature_builder_uses_real_labels_by_default_and_never_mixes() -> None:
    real = build_feature_frame(_usage_rows())
    assert real["observation_id"].tolist() == ["o1"]
    assert set(real["label_origin"]) == {"real"}
    synthetic = build_feature_frame(_usage_rows(), label_origin="synthetic")
    assert synthetic["observation_id"].tolist() == ["o2"]


def test_feature_builder_refuses_synthetic_only_data_without_opt_in() -> None:
    synthetic_only = _usage_rows().iloc[[1]]
    with pytest.raises(ValueError, match="--label-origin synthetic"):
        build_feature_frame(synthetic_only)


def test_feature_builder_requires_provenance_column() -> None:
    with pytest.raises(ValueError, match="data_origin"):
        build_feature_frame(_usage_rows().drop(columns=["data_origin"]))


# --- store -> features -> split -> release ------------------------------------


def test_store_feeds_features_split_and_release(
    store: DatasetStore, prompts: list[PromptRecord], tmp_path: Path
) -> None:
    _synthetic_store(store, prompts)
    usage = store_usage_frame(store)
    assert len(usage) == 40 and usage["prompt_family_id"].notna().all()
    features = build_feature_frame(usage, label_origin="synthetic")
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    splits_path = tmp_path / "splits.parquet"
    split_to_parquet(features_path, splits_path)
    manifest = freeze_release(features_path, splits_path, tmp_path / "rel", version="0.0.1")
    assert manifest["label_origin"] == "synthetic"
    assert "Pipeline testing only" in str(manifest["intended_use"])
    release_dir = tmp_path / "rel" / "dataset_v0.0.1_synthetic"
    schema = json.loads((release_dir / "schema.json").read_text())
    assert schema["columns"]["output_tokens"]["role"] == "TARGET"
    assert schema["columns"]["word_count"]["role"] == "REQUEST_DERIVED"
    with pytest.raises(FileExistsError, match="immutable"):
        freeze_release(features_path, splits_path, tmp_path / "rel", version="0.0.1")


def test_requests_without_family_use_prompt_hash(store: DatasetStore) -> None:
    from helpers import observation_payload, request_payload

    for index, prompt in enumerate(["same prompt", "same prompt", "other prompt"]):
        ingest_records(
            store,
            RecordType.REQUEST_FEATURES,
            [("x", request_payload(request_id=f"req_{index}", prompt_text=prompt))],
            source_name="t",
        )
        ingest_records(
            store,
            RecordType.OBSERVED_USAGE,
            [
                (
                    "x",
                    observation_payload(
                        observation_id=f"obs_{index}",
                        request_id=f"req_{index}",
                        provider_request_id=f"r_{index}",
                    ),
                )
            ],
            source_name="t",
        )
    usage = store_usage_frame(store).sort_values("request_id")
    families = usage["prompt_family_id"].tolist()
    assert families[0] == families[1] != families[2]
    assert set(usage["category"]) == {"uncategorized"}


def test_release_rejects_mixed_label_origins(tmp_path: Path) -> None:
    frame = build_feature_frame(_usage_rows())
    frame.loc[len(frame)] = frame.iloc[0]
    frame.loc[1, "label_origin"] = "synthetic"
    features_path = tmp_path / "f.parquet"
    frame.to_parquet(features_path)
    splits = pd.DataFrame([{"prompt_family_id": "pf", "category": "c", "split": "train"}])
    splits.to_parquet(tmp_path / "s.parquet")
    with pytest.raises(ValueError, match="exactly one label_origin"):
        freeze_release(features_path, tmp_path / "s.parquet", tmp_path / "rel", version="x")


# --- experiment ingestion, prediction and comparison --------------------------


def test_experiment_ingestion_and_predicted_vs_observed(
    store: DatasetStore, prompts: list[PromptRecord], tmp_path: Path
) -> None:
    raw = tmp_path / "raw.jsonl"
    run_manifest(_manifest(rows=3), prompts, raw)
    request_report, usage_report = ingest_experiment(store, raw, prompts)
    assert (request_report.stored, usage_report.stored) == (3, 3)
    stored = store.read(RecordType.REQUEST_FEATURES)
    assert stored["prompt_text"].isna().all()
    assert stored["retain_until"].isna().all()  # controlled experiments do not expire

    predictor = BaselinePredictor()
    first = ingest_records(
        store, RecordType.PREDICTION, prediction_records(store, predictor), source_name="p"
    )
    again = ingest_records(
        store, RecordType.PREDICTION, prediction_records(store, predictor), source_name="p"
    )
    assert (first.stored, again.read, again.stored, again.quarantined) == (3, 0, 0, 0)
    expected = predictor.predict(prompts[0].text, "m", "auto").predicted_output_tokens
    assert set(store.read(RecordType.PREDICTION)["predicted_output_tokens"]) == {expected}

    comparison = compare_predictions(store)
    assert len(comparison) == 3
    [summary] = summarize_comparison(comparison)
    assert summary["data_origin"] == "synthetic" and summary["pairs"] == 3
    assert "plumbing only" in summary["note"]


def test_manifest_config_can_live_anywhere_with_root(tmp_path: Path) -> None:
    config = tmp_path / "pilot.json"
    config.write_text(
        (REPO_ROOT / "configs" / "experiments" / "pilot_001.json").read_text(), encoding="utf-8"
    )
    frame = build_manifest_from_config(config, tmp_path / "m.parquet", root=REPO_ROOT)
    assert len(frame) == 144


# --- CLI ----------------------------------------------------------------------


def test_cli_reports_errors_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["store-summary", "--store", str(tmp_path / "empty")])
    assert code == 0
    code = main(["features", "--from-store", str(tmp_path / "empty"), "--output", "x.parquet"])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_ingest_requires_origin_and_source_together(tmp_path: Path) -> None:
    source = tmp_path / "x.jsonl"
    source.write_text("", encoding="utf-8")
    argv = ["ingest", "--store", str(tmp_path / "s"), "--record-type", "observed_usage"]
    assert main([*argv, "--input", str(source), "--origin", "real"]) == 2
