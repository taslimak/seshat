"""Statistical correctness of the forecasting stack on simulated data with known truth."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from seshat.modeling.calibration import BiasCorrector, ConformalCalibrator, km_quantile
from seshat.modeling.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SIGNAL_FEATURES
from seshat.modeling.models import CensoredLogNormal, UncensoredRidge


def _frame(n: int, seed: int, cap: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Latent log1p(y) = 5 + 1.0 * x, sd 0.5; right-censored at ``cap``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = 5.0 + 1.0 * x + rng.normal(scale=0.5, size=n)
    latent = np.expm1(z)
    frame = pd.DataFrame({c: 0.0 for c in NUMERIC_FEATURES + SIGNAL_FEATURES}, index=range(n))
    frame["log_prompt_words"] = x
    for c in CATEGORICAL_FEATURES:
        frame[c] = "a"
    frame["actual_output_tokens"] = np.minimum(latent, cap)
    frame["censored"] = latent > cap
    frame["cap"] = cap
    return frame, latent


def test_censored_lognormal_recovers_truth_where_uncensored_ridge_is_biased() -> None:
    frame, _ = _frame(3000, 1, cap=300.0)  # ~40% censored
    assert 0.25 < frame["censored"].mean() < 0.6
    model = CensoredLogNormal(penalty=0.01, scale_penalty=0.01).fit(frame)
    probe = frame.iloc[:1].copy()
    probe["log_prompt_words"] = 0.0
    mu, sigma = model.predict_log(probe)
    assert mu[0] == pytest.approx(5.0, abs=0.08)
    assert sigma[0] == pytest.approx(0.5, abs=0.06)
    biased, _ = UncensoredRidge(penalty=0.01).fit(frame).predict_log(probe)
    assert biased[0] < 4.85  # dropping censored rows under-predicts


def test_km_quantile_matches_empirical_without_censoring_and_handles_censoring() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=5000)
    none = np.zeros(values.size, dtype=bool)
    assert km_quantile(values, none, 0.9) == pytest.approx(np.quantile(values, 0.9), abs=0.05)
    # Censor the top 30% at a fixed point: the naive quantile of observed values is too low.
    cut = np.quantile(values, 0.7)
    observed = np.minimum(values, cut)
    censored = values > cut
    assert math.isinf(km_quantile(observed, censored, 0.9))  # unreachable: honest
    assert km_quantile(observed, censored, 0.5) == pytest.approx(np.median(values), abs=0.05)


def test_conformal_intervals_reach_nominal_coverage_out_of_sample() -> None:
    rng = np.random.default_rng(3)
    scores = rng.standard_t(df=4, size=400)  # heavy tails
    cal = ConformalCalibrator(confidence_level=0.9, segment_column=None).fit(
        scores, np.zeros(400, dtype=bool)
    )
    fresh = rng.standard_t(df=4, size=20000)
    coverage = float(np.mean((fresh >= cal.q_lo) & (fresh <= cal.q_hi)))
    assert 0.88 <= coverage <= 0.94


def test_segments_too_small_for_their_own_tail_use_the_global_quantile() -> None:
    rng = np.random.default_rng(4)
    scores = rng.normal(size=200)
    segments = {"task_group": np.array(["big"] * 190 + ["tiny"] * 10)}
    cal = ConformalCalibrator(confidence_level=0.9, segment_column="task_group").fit(
        scores, np.zeros(200, dtype=bool), segments
    )
    assert "tiny" not in cal.segment_q and "big" in cal.segment_q


def test_bias_correction_shrinks_small_segments() -> None:
    residual = np.concatenate([np.full(200, 0.0), np.full(4, 1.0)])
    segments = {
        "provider_model": np.array(["m"] * 204),
        "effort": np.array(["low"] * 204),
        "task_group": np.array(["common"] * 200 + ["rare"] * 4),
    }
    bias = BiasCorrector(shrinkage=15.0).fit(residual, np.zeros(204, dtype=bool), segments)
    delta = bias.delta(segments)
    assert delta[0] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < delta[-1] < 0.3  # 4 / (4 + 15) of the segment's +1.0 median
