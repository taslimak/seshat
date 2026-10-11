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
