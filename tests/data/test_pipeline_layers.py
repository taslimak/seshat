"""Storage, validation, training preparation and CSV export on a mocked collection run."""

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from seshat.data.collectors.collector import Collector
from seshat.data.collectors.normalize import normalize_experiment
from seshat.data.exports import COLUMNS, export_csv
from seshat.data.generators import generate_prompts
from seshat.data.io import read_parquet
from seshat.data.pricing import PriceTable
from seshat.data.privacy import RetentionPolicy
from seshat.data.storage import StorageLayout
from seshat.data.training import leakage_groups, prepare_training_dataset
from seshat.data.validators import validate_layer, validate_storage
from seshat.experiments.config import CollectionConfig
from seshat.experiments.design import build_plan
from seshat.providers.catalog import load_catalog
from tests.data.test_schemas import observation_record, request_record


@pytest.fixture
def collected(small_config: Callable[..., CollectionConfig]) -> Path:
    config = small_config()
    plan = build_plan(config, generate_prompts(34, config.seed), load_catalog())
    Collector(config, plan, mode="offline").run()
    normalize_experiment(
        config.storage.root,
        config.experiment_id,
        catalog=load_catalog(),
        prices=PriceTable.load(),
        policy=RetentionPolicy(),
    )
    return Path(config.storage.root)


def test_partitioned_parquet_layout_and_round_trip(collected: Path) -> None:
    layout = StorageLayout(collected)
    parts = layout.parts("observations")
    assert parts and all("date=" in str(p) and "provider=" in str(p) for p in parts)
    from seshat.data.schemas import ObservationRecord

    records = layout.read_records("observations", ObservationRecord)
    assert len(records) == len(layout.read_frame("observations"))


def test_validation_report_on_fixture_run(collected: Path) -> None:
    report = validate_storage(collected)
    totals = report["totals"]
    assert totals["observations"] == totals["eligible_labeled"] > 0
    assert totals["eligible_real_labels"] == 0  # mocks never yield real labels
    assert report["by_source_type"] == {"test_fixture": totals["observations"]}
    assert set(report["token_distributions"]) >= {"actual_input_tokens", "actual_output_tokens"}
    assert (
        (collected / "normalized" / "_validation" / "quality_report.md")
        .read_text()
        .startswith("# Seshat dataset quality report")
    )


def _frames(**obs_overrides: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req = pd.DataFrame([{**request_record().model_dump(), "sample_id": request_record().sample_id}])
    obs = pd.DataFrame([observation_record(**obs_overrides).model_dump()])
    obs["billing_dimensions_json"] = json.dumps(obs_overrides.get("billing_dimensions", {}))
    return req, pd.DataFrame(), obs


@pytest.mark.parametrize(
    ("overrides", "code", "decision"),
    [
        ({"normalized_total_tokens": 99}, "total_accounting", "QUARANTINE"),
        ({"actual_cached_input_tokens": 50}, "cache_accounting", "QUARANTINE"),
        (
            {"actual_reasoning_tokens": 60, "reasoning_included_in_output": True},
            "reasoning_accounting",
            "QUARANTINE",
        ),
        (
            {"billing_dimensions": {"accounting_issues": ["total_tokens_mismatch"]}},
            "provider_accounting",
            "QUARANTINE",
        ),
        ({"is_truncated": True}, "censored_output", "WARNING"),
        ({"resolved_model": "other-model"}, "unexpected_model_routing", "WARNING"),
        ({"provider": "mock-gemini"}, "identity_mismatch", "REJECT"),
    ],
)
def test_validation_checks(overrides: dict[str, object], code: str, decision: str) -> None:
    req, pre, obs = _frames(**overrides)
    result = validate_layer(req, pre, obs).iloc[0]
    assert code in result["flag_codes"]
    assert result["decision"] == decision
    assert result["eligible_for_training"] == (decision in {"PASS", "WARNING"})


def test_failed_and_orphan_observations_are_excluded() -> None:
    req, pre, obs = _frames(
        response_status="failed",
        actual_input_tokens=None,
        actual_output_tokens=None,
        actual_total_tokens=None,
        normalized_total_tokens=None,
        label_quality="unavailable",
        calculated_cost_usd=None,
        cost_method="unavailable",
    )
    result = validate_layer(req, pre, obs).iloc[0]
    assert result["decision"] == "QUARANTINE" and not result["eligible_for_training"]
    orphan = validate_layer(pd.DataFrame(columns=req.columns), pre, _frames()[2]).iloc[0]
    assert "orphan_observation" in orphan["flag_codes"]


def test_preflight_discrepancy_warning() -> None:
    req, _, obs = _frames()
    pre = pd.DataFrame(
        [
            {
                "sample_id": obs["sample_id"].iloc[0],
                "preflight_input_tokens": 10,
                "preflight_method": "provider_native",
                "preflight_complete": True,
            }
        ]
    )
    result = validate_layer(req, pre, obs, preflight_tolerance=0.1).iloc[0]
    assert "preflight_discrepancy" in result["flag_codes"]


def test_training_defaults_to_real_labels_and_never_mixes(collected: Path) -> None:
    validate_storage(collected)
    with pytest.raises(ValueError, match="--label-origin synthetic"):
        prepare_training_dataset(collected, "0.1.0")
    manifest = prepare_training_dataset(collected, "0.1.0", label_origin="synthetic")
    assert (
        manifest["label_origin"] == "synthetic"
        and "Pipeline testing only" in manifest["intended_use"]
    )
    out = collected / "training" / "dataset_v0.1.0_synthetic"
    frames = {
        s: read_parquet(out / f"{s}.parquet")
        for s in ("train", "validation", "calibration", "test")
    }
    assert all(len(f) for f in frames.values())
    groups = {s: set(f["split_group_id"]) for s, f in frames.items()}
    for a in groups:
        for b in groups:
            if a < b:
                assert not groups[a] & groups[b], (a, b)
    families = {s: set(f["prompt_family_id"]) for s, f in frames.items()}
    assert not families["train"] & families["test"]
    schema = json.loads((out / "schema.json").read_text())
    assert not set(schema["features"]) & set(schema["targets"])
    assert (
        "actual_output_tokens" not in schema["features"]
        and "preflight_input_tokens" in schema["features"]
    )
    assert json.loads((out / "split_stats.json").read_text())["train"]["rows"] == len(
        frames["train"]
    )
    with pytest.raises(FileExistsError):
        prepare_training_dataset(collected, "0.1.0", label_origin="synthetic")


def test_temporal_holdout_takes_newest_groups(collected: Path) -> None:
    validate_storage(collected)
    prepare_training_dataset(
        collected, "t", label_origin="synthetic", temporal_holdout_fraction=0.25
    )
    out = collected / "training" / "dataset_vt_synthetic"
    test = read_parquet(out / "test.parquet")
    train = read_parquet(out / "train.parquet")
    test_first = test.groupby("split_group_id")["timestamp_utc"].min()
    train_first = train.groupby("split_group_id")["timestamp_utc"].min()
    assert test_first.min() >= train_first.min()
    assert len(test) >= 0.2 * (len(test) + len(train))


def test_leakage_groups_union_shared_keys() -> None:
    frame = pd.DataFrame(
        [
            {
                "prompt_family_id": "f1",
                "prompt_hash": "h1",
                "conversation_hash": None,
                "conversation_id": None,
            },
            {
                "prompt_family_id": "f2",
                "prompt_hash": "h1",
                "conversation_hash": None,
                "conversation_id": None,
            },
            {
                "prompt_family_id": "f3",
                "prompt_hash": "h3",
                "conversation_hash": "c1",
                "conversation_id": None,
            },
            {
                "prompt_family_id": "f4",
                "prompt_hash": "h4",
                "conversation_hash": "c1",
                "conversation_id": None,
            },
            {
                "prompt_family_id": "f5",
                "prompt_hash": "h5",
                "conversation_hash": None,
                "conversation_id": None,
            },
        ]
    )
    groups = leakage_groups(
        frame, ("prompt_family_id", "prompt_hash", "conversation_hash", "conversation_id")
    )
    assert groups[0] == groups[1] and groups[2] == groups[3]
    assert len({groups[0], groups[2], groups[4]}) == 3


def test_csv_export_is_readable_and_documented(collected: Path) -> None:
    validate_storage(collected)
    path = export_csv(collected)
    frame = pd.read_csv(path)
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == len(StorageLayout(collected).read_frame("observations"))
    for column in ("actual_input_tokens", "actual_output_tokens", "prompt_char_count"):
        assert pd.api.types.is_integer_dtype(frame[column]) or frame[column].isna().any()
    assert not any(c.endswith("_json") for c in frame.columns)
    dictionary = path.with_name("data_dictionary.md").read_text()
    assert all(f"| {name} |" in dictionary for name in COLUMNS)


def test_purge_expired_removes_only_expired_rows(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    now = datetime(2026, 10, 1, tzinfo=UTC)
    keep = request_record()
    expire = request_record(
        sample_id=__import__("uuid").uuid4(), request_id="req_2", retain_until=now
    )
    layout.replace_source("requests", "t", [keep, expire])
    layout.replace_source(
        "observations",
        "t",
        [
            observation_record(
                sample_id=expire.sample_id,
                request_id="req_2",
                observation_id="obs_2",
                retain_until=now,
            )
        ],
    )
    removed = layout.purge_expired(now + timedelta(seconds=1))
    assert removed["requests"] == 1 and removed["observations"] == 1
    assert layout.read_frame("requests")["request_id"].tolist() == ["req_1"]
    assert layout.read_frame("observations").empty


def test_committed_offline_example_is_safe_fixture_data() -> None:
    """The committed sample export must hold only mock (test_fixture) rows and no text."""
    from tests.data.conftest import REPO_ROOT

    folder = REPO_ROOT / "data" / "fixtures" / "offline_example"
    frame = pd.read_csv(folder / "sample_export.csv")
    assert list(frame.columns) == list(COLUMNS)
    assert set(frame["source_type"]) == {"test_fixture"}
    assert set(frame["label_quality"]) <= {"synthetic"}
    assert all(provider.startswith("mock-") for provider in frame["provider"])
    report = json.loads((folder / "quality_report.json").read_text())
    assert report["totals"]["eligible_real_labels"] == 0
