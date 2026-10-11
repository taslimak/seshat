"""End-to-end: train -> register -> outcomes -> snapshots -> drift -> serve (offline)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seshat.modeling.dataset import family_bucket, load_modeling_frame
from seshat.modeling.features import model_features, request_from_prompt
from seshat.modeling.pipeline import train_and_evaluate
from seshat.predictor import BaselinePredictor, CalibratedPredictor
from seshat.quality.outcomes import read_outcomes
from seshat.quality.refresh import refresh_quality, register_model
from seshat.quality.snapshots import drift_report
from tests.real_shape import mock_collection, relabel_as_real


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict]:
    base = tmp_path_factory.mktemp("fc")
    data = mock_collection(base / "run", count=220, seed=11)
    relabel_as_real(data)
    model_dir = base / "model"
    evaluation = train_and_evaluate(data, model_dir, n_folds=4)
    return data, model_dir, evaluation


def test_protected_test_is_disjoint_and_hash_stable(trained: tuple[Path, Path, dict]) -> None:
    data, _, evaluation = trained
    frame = load_modeling_frame(data)
    dev, test = frame[~frame["protected_test"]], frame[frame["protected_test"]]
    assert not set(dev["split_group_id"]) & set(test["split_group_id"])
    assert not set(dev["prompt_family_id"]) & set(test["prompt_family_id"])
    assert family_bucket("fam_x") == family_bucket("fam_x")
    assert evaluation["counts"]["test_rows"] == len(test)


def test_evaluation_record_is_complete(trained: tuple[Path, Path, dict]) -> None:
    _, model_dir, evaluation = trained
    for key in (
        "predictor_version",
        "uncertainty_model_version",
        "calibration_version",
        "dataset_version",
        "feature_set_version",
        "hyperparameters",
        "protected_test",
        "promotion",
        "code_revision",
        "created_at",
    ):
        assert key in evaluation
    assert evaluation["promotion"]["decision"] in {"PASS", "FAIL"}
    assert "baseline-0.1" in evaluation["protected_test"]
    artifact = json.loads((model_dir / "forecaster.json").read_text())
    assert artifact["artifact_format"] == "seshat-forecaster/1"


def test_register_refresh_is_idempotent_and_origin_separated(
    trained: tuple[Path, Path, dict],
) -> None:
    data, model_dir, evaluation = trained
    register_model(data, model_dir)
    first = refresh_quality(data)
    register_model(data, model_dir)
    second = refresh_quality(data)
    assert first["outcomes"] == second["outcomes"]
    # Nothing new since the registered evaluation: no retraining on the data-volume rule.
    assert second["retraining"]["new_since_last_evaluation"] == 0
    assert not any("new real observations" in r for r in second["retraining"]["reasons"])
    outcomes = read_outcomes(data)
    assert set(outcomes["label_origin"]) == {"real"}  # relabelled test data only
    version = evaluation["predictor_version"]
    mine = outcomes[outcomes["predictor_version"] == version]
    assert mine["prediction_id"].is_unique
    assert mine["valid_out_of_sample"].all()
    assert set(mine["evaluation_role"]) == {"protected_test", "cross_validation"}
    assert (Path(data) / "quality" / "model_quality_snapshots.parquet").exists()
    ledger = (Path(data) / "quality" / "model_evaluations.jsonl").read_text().splitlines()
    assert len(ledger) == 1


def test_synthetic_outcomes_are_tagged_and_never_real(tmp_path: Path) -> None:
    from seshat.data.cli import main as cli_main

    data = mock_collection(tmp_path / "fixture", count=40, seed=2)
    assert cli_main(["validate", "--root", str(data)]) == 0
    summary = refresh_quality(data)
    assert set(summary["outcomes_by_origin"]) == {"synthetic"}


def test_calibrated_predictor_serves_margins(trained: tuple[Path, Path, dict]) -> None:
    _, model_dir, evaluation = trained
    predictor = CalibratedPredictor(str(model_dir / "forecaster.json"))
    result = predictor.predict(
        "Write a Python function `merge_intervals` with type hints. Include unit tests.",
        "mock-gpt",
        "high",
        max_output_tokens=512,
    )
    assert result.interval_calibrated and result.confidence_level == pytest.approx(0.9)
    assert result.lower_bound <= result.predicted_output_tokens <= result.upper_bound <= 512
    assert result.confidence_margin_tokens == max(
        result.lower_margin_tokens or 0, result.upper_margin_tokens or 0
    )
    assert result.predictor_version == evaluation["predictor_version"]
    assert result.calibration_version == evaluation["calibration_version"]


def test_serving_features_match_training_features() -> None:
    request = request_from_prompt(
        "Translate the following text into Dutch, preserving tone.\n\nHello.",
        provider="openai",
        model="gpt-6.1-sol",
        effort="low",
    )
    features = model_features(request)
    assert features == model_features(request.model_copy(deep=True))
    assert features["task_group"] == "translate" and features["effort"] == "low"


def test_baseline_reports_uncalibrated_margins() -> None:
    result = BaselinePredictor().predict("Hello there", "m", "auto")
    assert result.interval_calibrated is False and result.confidence_level is None
    assert result.confidence_margin_tokens is not None


def _outcomes(n: int, shift: float, start: datetime, experiment: str) -> pd.DataFrame:
    rng = np.random.default_rng(int(shift * 100) + n)
    err = rng.normal(loc=shift, scale=0.2, size=n)
    return pd.DataFrame(
        {
            "predictor_version": "v",
            "label_origin": "real",
            "valid_out_of_sample": True,
            "evaluation_role": None,
            "experiment_id": experiment,
            "timestamp_actual_observed": [start + timedelta(minutes=i) for i in range(n)],
            "realized_log_error": err,
            "realized_absolute_percentage_error": 100 * np.abs(np.expm1(err)),
            "realized_interval_hit": np.abs(err) < 0.33,
            "confidence_level": 0.9,
        }
    )


def test_drift_requires_evidence_and_detects_real_shifts() -> None:
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    stable = pd.concat(
        [_outcomes(80, 0.0, t0, "a"), _outcomes(60, 0.0, t0 + timedelta(days=1), "b")]
    )
    assert drift_report(stable, "v")["status"] == "ok"
    shifted = pd.concat(
        [_outcomes(80, 0.0, t0, "a"), _outcomes(60, 0.6, t0 + timedelta(days=1), "b")]
    )
    report = drift_report(shifted, "v")
    assert report["status"] == "drift" and "error distribution worsened" in report["flags"]
    small = pd.concat(
        [_outcomes(80, 0.0, t0, "a"), _outcomes(10, 0.6, t0 + timedelta(days=1), "b")]
    )
    assert drift_report(small, "v")["status"] == "insufficient_data"
