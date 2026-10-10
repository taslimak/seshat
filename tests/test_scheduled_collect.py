"""Offline tests for the scheduled collector."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from scripts.scheduled_collect import MAX_REQUESTS_PER_RUN, collect


def test_mock_collection_is_bounded_and_exports(tmp_path: Path) -> None:
    result = collect(
        mode="mock", output_dir=tmp_path,
        now=datetime(2026, 10, 10, 9, 17, tzinfo=ZoneInfo("America/New_York")),
    )
    assert result["manifest_requests"] <= MAX_REQUESTS_PER_RUN
    assert result["successful_observations"] == result["manifest_requests"]
    assert (tmp_path / "observations.jsonl").exists()
    assert (tmp_path / "records.csv").exists()
    assert len(pd.read_parquet(tmp_path / "validated.parquet")) <= MAX_REQUESTS_PER_RUN


def test_paid_requires_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESHAT_ENABLE_PAID_COLLECTION", raising=False)
    with pytest.raises(RuntimeError, match="Paid collection disabled"):
        collect(mode="paid", output_dir=tmp_path)


def test_paid_requires_budget_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SESHAT_ENABLE_PAID_COLLECTION", "true")
    monkeypatch.delenv("SESHAT_PROVIDER_BUDGET_CONFIRMED", raising=False)
    with pytest.raises(RuntimeError, match="Confirm an external provider billing cap"):
        collect(mode="paid", output_dir=tmp_path)
