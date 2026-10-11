"""Safety gates for real-label model training."""

import json

import pandas as pd
import pytest

from seshat.train_predictor import train_from_dataset


def test_rejects_synthetic_training_before_reading_splits(tmp_path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"label_origin": "synthetic", "source_types": ["test_fixture"]})
    )
    with pytest.raises(ValueError, match="Refusing synthetic"):
        train_from_dataset(source, tmp_path / "model")


def test_rejects_insufficient_verified_real_observations(tmp_path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "label_origin": "real",
                "source_types": ["controlled_experiment"],
                "dataset_version": "0.2.0",
            }
        )
    )
    row = {
        "label_origin": "real",
        "source_type": "controlled_experiment",
        "label_quality": "verified",
        "target_is_censored": False,
        "actual_output_tokens": 100,
        "prompt_word_count": 15,
        "requested_model": "test-model",
        "requested_effort": None,
        "split_group_id": "a",
    }
    for name in ("train", "validation", "calibration", "test"):
        pd.DataFrame([row]).to_parquet(source / f"{name}.parquet")
    with pytest.raises(ValueError, match="Insufficient verified"):
        train_from_dataset(source, tmp_path / "model")


def test_trained_predictor_end_to_end_on_isolated_fixture(tmp_path) -> None:
    """Synthetic *test fixtures* exercise the trainer; not production training data."""
    from seshat.predictor import TrainedPredictor

    source = tmp_path / "mocked_real_shape"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "label_origin": "real",
                "source_types": ["controlled_experiment"],
                "dataset_version": "fixture",
            }
        )
    )
    for name, n in (("train", 120), ("validation", 15), ("calibration", 25), ("test", 25)):
        rows = [
            {
                "label_origin": "real",
                "source_type": "controlled_experiment",
                "label_quality": "verified",
                "target_is_censored": False,
                "actual_output_tokens": 30 + i % 50,
                "prompt_word_count": 20 + i % 50,
                "requested_model": "fixture-model",
                "requested_effort": "low",
                "split_group_id": f"{name}-{i}",
            }
            for i in range(n)
        ]
        pd.DataFrame(rows).to_parquet(source / f"{name}.parquet")
    target = tmp_path / "trained"
    from seshat.train_predictor import train_from_dataset

    metrics = train_from_dataset(source, target)
    assert metrics["train_rows"] == 120
    assert 0 <= metrics["test_interval_coverage"] <= 1
    assert (target / "metrics.json").exists()
    predictor = TrainedPredictor(str(target / "predictor.joblib"))
    prediction = predictor.predict("Please summarize this text.", "fixture-model", "low")
    assert prediction.lower_bound <= prediction.predicted_output_tokens <= prediction.upper_bound
    assert prediction.predictor_version == "ridge-fixture"
