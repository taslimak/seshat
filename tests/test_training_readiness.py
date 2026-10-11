"""Fail-safe training readiness reporting (no paid API requests)."""

from scripts.report_training_readiness import readiness


def test_empty_storage_is_not_trainable(tmp_path) -> None:
    report = readiness(tmp_path)
    assert report["status"] == "no_data"
    assert report["trainer_minimums"] == {"train": 100, "calibration": 20, "test": 20}
