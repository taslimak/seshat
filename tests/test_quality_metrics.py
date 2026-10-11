"""Per-prediction and aggregate forecast-error metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from seshat.quality.metrics import add_row_metrics, aggregate_metrics


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "predicted_output_tokens": [1200.0, 800.0, 5.0, 300.0, 700.0],
            "actual_output_tokens": [1000.0, 1000.0, 2.0, 512.0, 512.0],
            "target_is_censored": [False, False, False, True, True],
            "max_output_tokens": [4096, 4096, 4096, 512, 512],
            "prediction_interval_lower": [1020.0, 700.0, 1.0, 200.0, 400.0],
            "prediction_interval_upper": [1380.0, 900.0, 10.0, 400.0, 900.0],
            "confidence_level": [0.9] * 5,
            "confidence_margin_tokens": [180.0, 100.0, 5.0, 100.0, 300.0],
            "confidence_margin_pct": [15.0, 12.5, 100.0, 33.3, 42.9],
            "predicted_cost_usd": [0.012, 0.008, 0.0001, 0.003, 0.005],
            "calculated_actual_cost_usd": [0.010, 0.010, 0.0001, 0.005, 0.005],
        }
    )


def test_sign_convention_and_percentage_forms() -> None:
    out = add_row_metrics(_rows())
    first = out.iloc[0]
    assert first["signed_error_tokens"] == 200  # prediction - actual: over-prediction
    assert first["signed_percentage_error"] == pytest.approx(20.0)
    assert first["symmetric_absolute_percentage_error"] == pytest.approx(100 * 2 * 200 / 2200)
    assert first["multiplicative_error"] == pytest.approx(1201 / 1001)
    assert first["error_direction"] == "overprediction"
    assert out.iloc[1]["error_direction"] == "underprediction"
    # Tiny denominators: percentage undefined, sMAPE still bounded.
    tiny = out.iloc[2]
    assert math.isnan(tiny["absolute_percentage_error"]) and tiny["pct_denominator_too_small"]
    assert 0 <= tiny["symmetric_absolute_percentage_error"] <= 200


def test_censored_rows_are_lower_bounds_not_exact_labels() -> None:
    out = add_row_metrics(_rows())
    censored = out[out["target_is_censored"]]
    assert censored["absolute_error_tokens"].isna().all()
    assert censored["absolute_percentage_error"].isna().all()
    assert bool(censored.iloc[0]["prediction_below_known_lower_bound"]) is True
    assert censored.iloc[0]["lower_bound_error_tokens"] == 212
    # Interval: censored y=512 > U=400 is a known miss; with U=900 it is unknown.
    assert list(censored["interval_position"]) == ["above_interval", "unknown_censored"]
    # Realized usage is exact: min(prediction, cap) vs the cap actually produced.
    assert list(censored["realized_absolute_error_tokens"]) == [212.0, 0.0]


def test_interval_metrics() -> None:
    out = add_row_metrics(_rows())
    assert list(out["interval_position"][:3]) == [
        "below_interval",
        "above_interval",
        "inside_interval",
    ]
    assert out.iloc[0]["interval_miss_tokens"] == 20
    assert out.iloc[0]["interval_width_tokens"] == 360


def test_aggregates() -> None:
    m = aggregate_metrics(add_row_metrics(_rows()))
    assert m["n_observations"] == 5 and m["n_uncensored"] == 3 and m["n_censored"] == 2
    assert m["mae_tokens"] == pytest.approx((200 + 200 + 3) / 3)
    assert m["wape_pct"] == pytest.approx(100 * 403 / 2002)
    assert m["underprediction_count"] == 1 and m["overprediction_count"] == 2
    assert m["censored_known_underprediction_count"] == 1
    # Known rows: below, above, inside, censored known-above -> 1 / 4; one censored unknown.
    assert m["interval_coverage_pct"] == pytest.approx(25.0)
    assert m["interval_coverage_pct_bounds"] == [20.0, 40.0]
    assert m["below_interval_rate_pct"] == pytest.approx(25.0)
    assert m["above_interval_rate_pct"] == pytest.approx(50.0)
    assert m["nominal_coverage_pct"] == pytest.approx(90.0)
    assert m["catastrophic_gt100pct_count"] == 0
    assert m["realized_n"] == 5
    assert m["cost_mae_usd"] == pytest.approx(np.mean([0.002, 0.002, 0.0, 0.002, 0.0]))
