"""Readiness split simulation, evaluation metrics and censoring diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seshat.data.evaluation import (
    censored_diagnostics,
    evaluate_baseline,
    evaluate_frame,
    group_bootstrap,
    interval_metrics,
    point_metrics,
)
from seshat.data.readiness import implied_minimum_usable, readiness_report
from seshat.data.split import DEFAULT_RATIOS
from seshat.data.training import REAL_LABEL_RATIOS
from tests.real_shape import mock_collection, relabel_as_real


def test_implied_minimums_explain_the_ratio_mismatch() -> None:
    # 20 calibration rows at 8% => 250 usable rows under the historical defaults.
    assert implied_minimum_usable(DEFAULT_RATIOS) == 250
    assert implied_minimum_usable(REAL_LABEL_RATIOS) == 167


def test_point_and_interval_metrics() -> None:
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 180.0, 300.0])
    point = point_metrics(actual, predicted)
    assert point["mae"] == pytest.approx(10.0)
    assert point["bias"] == pytest.approx(-10 / 3)
    assert point["wape"] == pytest.approx(30 / 600)
    assert point["rmse"] == pytest.approx(np.sqrt((100 + 400) / 3))
    interval = interval_metrics(actual, actual - 5, actual + [5, -1, 5])
    assert interval["coverage"] == pytest.approx(2 / 3)
    assert point_metrics(np.array([]), np.array([]))["mae"] is None


def test_censored_rows_are_lower_bounds() -> None:
    cap = np.array([512.0, 512.0])
    diag = censored_diagnostics(cap, np.array([300.0, 900.0]), upper=np.array([500.0, 1200.0]))
    assert diag["known_underprediction_rate"] == 0.5
    assert diag["known_interval_miss_rate"] == 0.5


def test_group_bootstrap_resamples_groups() -> None:
    frame = pd.DataFrame({"g": ["a", "a", "b", "c"], "x": [1.0, 1.0, 5.0, 9.0]})
    result = group_bootstrap(frame, lambda f: float(f["x"].mean()), group_column="g", n_boot=300)
    assert result["groups"] == 3
    assert result["lower"] <= result["estimate"] <= result["upper"]


def test_censored_rows_excluded_from_point_metrics() -> None:
    frame = pd.DataFrame(
        {
            "actual": [100.0, 512.0],
            "predicted": [100.0, 100.0],
            "lower": [50.0, 50.0],
            "upper": [150.0, 150.0],
            "censored": [False, True],
            "cap": [512.0, 512.0],
            "group": ["a", "b"],
        }
    )
    report = evaluate_frame(frame, n_boot=50)
    assert report["point"]["n"] == 1 and report["point"]["mae"] == 0.0
    assert report["censored"]["known_underprediction_rate"] == 1.0


@pytest.fixture(scope="module")
def real_shaped(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = mock_collection(tmp_path_factory.mktemp("rs"), count=150)
    relabel_as_real(data)
    return data


def test_readiness_simulates_exact_split(real_shaped: Path) -> None:
    report = readiness_report(real_shaped)
    totals = report["totals"]
    assert totals["eligible_real_rows"] == 150
    assert totals["usable_rows"] + totals["censored_rows"] <= 150
    for sim in report["split_simulation"].values():
        usable = sum(info["usable"] for info in sim["splits"].values())
        assert usable == totals["usable_rows"]
        if not sim["meets_trainer_minimums"]:
            assert sim["estimated_additional_usable_rows"] > 0
    assert report["status"] in {"insufficient_data", "ready_for_training"}
    assert any("effort" in gap or "model" in gap for gap in report["coverage_gaps"])


def test_baseline_evaluation_runs_on_real_shaped_rows(real_shaped: Path) -> None:
    report = evaluate_baseline(real_shaped, n_boot=100)
    assert report["rows"] == 150
    assert report["uncensored_rows"] + report["censored_rows"] == 150
    assert set(report["slices"]) >= {"prompt_category", "length_bucket", "requested_effort"}
    assert report["bootstrap_95"]["mae"]["groups"] > 1


def test_spend_by_month_counts_real_observations(real_shaped: Path) -> None:
    from decimal import Decimal

    from seshat.data.consolidate import month_to_date_spend, spend_by_month

    months = spend_by_month(real_shaped)["months"]
    assert len(months) == 1
    month = next(iter(months))
    assert month_to_date_spend(real_shaped, month) > Decimal(0)
    assert month_to_date_spend(real_shaped, "1999-01") == Decimal(0)
