"""Fail-safe training readiness reporting (no paid API requests)."""

from scripts.report_training_readiness import readiness


def test_empty_storage_is_not_trainable(tmp_path) -> None:
    report = readiness(tmp_path)
    assert report["status"] == "insufficient_data"
    assert report["eligible_real_uncensored"] == 0
    assert report["minimum_total_usable"] == 140
