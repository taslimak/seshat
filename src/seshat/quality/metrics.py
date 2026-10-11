"""Canonical forecast-error definitions (per prediction and aggregated).

Sign convention (project-wide): ``error = prediction - actual``.
Positive = over-prediction, negative = under-prediction.

Targets
-------
* **Latent output tokens** -- what the model would generate without a cap. A
  truncated response only bounds it from below (``y >= cap``), so output-token
  error metrics use *uncensored* rows only; censored rows get lower-bound
  diagnostics instead (``prediction_below_known_lower_bound`` etc.).
* **Realized usage** -- ``min(latent, cap)``: what is actually generated and
  billed. Censored rows are *exact* for realized usage, so realized-output,
  cost and total-token errors use every row, with the forecast
  ``min(prediction, cap)`` (the cap is known before dispatch). Because these are
  observed for all rows they are free of the selection bias that affects
  latent metrics (uncensored rows over-represent short outputs), and they are
  Seshat's primary model-selection and promotion target.

Percentages use the actual value as denominator and are *undefined* (NaN) when
the actual is below :data:`MIN_PCT_DENOMINATOR_TOKENS` (resp. USD), so tiny
outputs cannot dominate aggregates; ``smape`` (bounded, 0-200%) and the log
ratio are reported alongside. All raw predicted and actual values are kept, so
every metric can be recomputed under a future definition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METRICS_VERSION = "quality_metrics/1.0.0"
MIN_PCT_DENOMINATOR_TOKENS = 10.0
MIN_PCT_DENOMINATOR_USD = 1e-6
NEAR_EXACT_TOKENS = 2.0
NEAR_EXACT_FRACTION = 0.02
CATASTROPHIC_PCT_THRESHOLDS = (25.0, 50.0, 100.0)
CATASTROPHIC_TOKEN_THRESHOLDS = (250.0, 500.0, 1000.0)
ERROR_PERCENTILES = (50, 75, 90, 95, 99)


def _pct(numerator: pd.Series, denominator: pd.Series, minimum: float) -> pd.Series:
    valid = denominator >= minimum
    return (100.0 * numerator / denominator.where(valid)).astype(float)


def _smape(p: pd.Series, y: pd.Series) -> pd.Series:
    denom = p.abs() + y.abs()
    return (100.0 * 2.0 * (p - y).abs() / denom.where(denom > 0)).fillna(0.0)


def add_row_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add every per-prediction error column.

    Required columns: ``predicted_output_tokens``, ``actual_output_tokens``,
    ``target_is_censored``, ``max_output_tokens``. Optional (metrics are NaN
    when absent): ``prediction_interval_lower/upper``, ``predicted_cost_usd``,
    ``calculated_actual_cost_usd``, ``predicted_total_tokens``,
    ``actual_total_tokens``, ``predicted_reasoning_tokens``,
    ``actual_reasoning_tokens``, ``preflight_input_tokens``, ``actual_input_tokens``.
    """
    out = frame.copy()
    p = out["predicted_output_tokens"].astype(float)
    y = out["actual_output_tokens"].astype(float)
    censored = out["target_is_censored"].fillna(False).astype(bool)
    exact = ~censored & y.notna()
    err = (p - y).where(exact)
    out["signed_error_tokens"] = err
    out["absolute_error_tokens"] = err.abs()
    out["squared_error_tokens2"] = err**2
    out["signed_percentage_error"] = _pct(err, y.where(exact), MIN_PCT_DENOMINATOR_TOKENS)
    out["absolute_percentage_error"] = out["signed_percentage_error"].abs()
    out["symmetric_absolute_percentage_error"] = _smape(p, y).where(exact)
    out["log_error"] = pd.Series(np.log((p + 1.0) / (y + 1.0)), index=out.index).where(exact)
    out["multiplicative_error"] = np.exp(out["log_error"])
    out["pct_denominator_too_small"] = exact & (y < MIN_PCT_DENOMINATOR_TOKENS)
    tolerance = np.maximum(NEAR_EXACT_TOKENS, NEAR_EXACT_FRACTION * y)
    direction = np.where(
        err.abs() <= tolerance,
        "near_exact",
        np.where(err > 0, "overprediction", "underprediction"),
    )
    out["error_direction"] = pd.Series(direction, index=out.index).where(exact)

    # Censored: the observed value is a lower bound on the latent output.
    out["observed_lower_bound_tokens"] = y.where(censored)
    below = censored & (p < y)
    out["prediction_below_known_lower_bound"] = below.where(censored)
    out["lower_bound_error_tokens"] = (y - p).clip(lower=0).where(censored)
    out["lower_bound_error_percentage"] = _pct(
        out["lower_bound_error_tokens"], y.where(censored), MIN_PCT_DENOMINATOR_TOKENS
    )

    # Realized usage: exact for every row (truncated responses produced exactly the cap).
    cap = pd.to_numeric(out["max_output_tokens"], errors="coerce").astype(float).fillna(np.inf)
    rp = p.where(p <= cap, cap)
    out["realized_predicted_output_tokens"] = rp
    r_err = rp - y
    out["realized_signed_error_tokens"] = r_err
    out["realized_absolute_error_tokens"] = r_err.abs()
    out["realized_squared_error_tokens2"] = r_err**2
    out["realized_signed_percentage_error"] = _pct(r_err, y, MIN_PCT_DENOMINATOR_TOKENS)
    out["realized_absolute_percentage_error"] = out["realized_signed_percentage_error"].abs()
    out["realized_symmetric_percentage_error"] = _smape(rp, y)
    out["realized_log_error"] = pd.Series(np.log((rp + 1.0) / (y + 1.0)), index=out.index)

    # Prediction interval.
    if {"prediction_interval_lower", "prediction_interval_upper"} <= set(out.columns):
        lo = out["prediction_interval_lower"].astype(float)
        hi = out["prediction_interval_upper"].astype(float)
        position = np.where(
            y < lo, "below_interval", np.where(y > hi, "above_interval", "inside_interval")
        )
        # Censored: only "above" can be known (latent >= y > hi); otherwise unknown.
        position = np.where(
            censored, np.where(y > hi, "above_interval", "unknown_censored"), position
        )
        out["interval_position"] = position
        out["interval_hit"] = pd.Series(position == "inside_interval", index=out.index).where(
            position != "unknown_censored"
        )
        miss = np.where(y < lo, lo - y, np.where(y > hi, y - hi, 0.0))
        out["interval_miss_tokens"] = pd.Series(miss, index=out.index).where(
            position != "unknown_censored"
        )
        out["interval_miss_percentage"] = _pct(
            out["interval_miss_tokens"], y.where(exact), MIN_PCT_DENOMINATOR_TOKENS
        )
        out["interval_width_tokens"] = hi - lo
        out["interval_width_percentage"] = _pct(hi - lo, p, MIN_PCT_DENOMINATOR_TOKENS)
        r_lo, r_hi = np.minimum(lo, cap), np.minimum(hi, cap)
        out["realized_interval_hit"] = (y >= r_lo) & (y <= r_hi)
        out["realized_interval_position"] = np.where(
            y < r_lo, "below_interval", np.where(y > r_hi, "above_interval", "inside_interval")
        )

    # Realized usage (exact for every row): cost and total tokens.
    for kind, pred_col, act_col, minimum in (
        ("cost", "predicted_cost_usd", "calculated_actual_cost_usd", MIN_PCT_DENOMINATOR_USD),
        ("total", "predicted_total_tokens", "actual_total_tokens", MIN_PCT_DENOMINATOR_TOKENS),
        (
            "reasoning",
            "predicted_reasoning_tokens",
            "actual_reasoning_tokens",
            MIN_PCT_DENOMINATOR_TOKENS,
        ),
        (
            "preflight_input",
            "preflight_input_tokens",
            "actual_input_tokens",
            MIN_PCT_DENOMINATOR_TOKENS,
        ),
    ):
        if pred_col not in out.columns or act_col not in out.columns:
            continue
        pv = pd.to_numeric(out[pred_col], errors="coerce").astype(float)
        av = pd.to_numeric(out[act_col], errors="coerce").astype(float)
        e = pv - av
        unit = "usd" if kind == "cost" else "tokens"
        prefix = kind if kind != "preflight_input" else "preflight_input"
        out[f"{prefix}_signed_error_{unit}"] = e
        out[f"{prefix}_absolute_error_{unit}"] = e.abs()
        if kind == "cost":
            out["squared_cost_error_usd2"] = e**2
        out[f"{prefix}_signed_percentage_error"] = _pct(e, av, minimum)
        out[f"{prefix}_absolute_percentage_error"] = _pct(e.abs(), av, minimum)
        out[f"{prefix}_symmetric_percentage_error"] = _smape(pv, av).where(pv.notna() & av.notna())
    return out


def _q(values: pd.Series, q: float) -> float | None:
    values = values.dropna()
    return float(values.quantile(q)) if len(values) else None


def _mean(values: pd.Series) -> float | None:
    values = values.dropna()
    return float(values.mean()) if len(values) else None


def aggregate_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Aggregate quality of rows already passed through :func:`add_row_metrics`."""
    censored = rows["target_is_censored"].fillna(False).astype(bool)
    exact = rows[~censored & rows["absolute_error_tokens"].notna()]
    n = len(exact)
    y = exact["actual_output_tokens"].astype(float)
    p = exact["predicted_output_tokens"].astype(float)
    abs_err = exact["absolute_error_tokens"]
    total_actual = float(y.sum()) if n else 0.0
    m: dict[str, Any] = {
        "metrics_version": METRICS_VERSION,
        "n_observations": int(len(rows)),
        "n_uncensored": n,
        "n_censored": int(censored.sum()),
        "censoring_rate_pct": 100.0 * float(censored.mean()) if len(rows) else None,
        "mae_tokens": _mean(abs_err),
        "median_absolute_error_tokens": _q(abs_err, 0.5),
        "rmse_tokens": float(np.sqrt(exact["squared_error_tokens2"].mean())) if n else None,
        "wape_pct": 100.0 * float(abs_err.sum()) / total_actual if total_actual > 0 else None,
        "median_absolute_percentage_error": _q(exact["absolute_percentage_error"], 0.5),
        "smape_pct": _mean(exact["symmetric_absolute_percentage_error"]),
        "nrmse_pct": 100.0 * float(np.sqrt(exact["squared_error_tokens2"].mean())) / float(y.mean())
        if n and y.mean() > 0
        else None,
        "mean_signed_error_tokens": _mean(exact["signed_error_tokens"]),
        "median_signed_error_tokens": _q(exact["signed_error_tokens"], 0.5),
        "mean_signed_percentage_error": _mean(exact["signed_percentage_error"]),
        "aggregate_percentage_bias": 100.0 * float((p - y).sum()) / total_actual
        if total_actual > 0
        else None,
        "geometric_mean_ratio": float(np.exp(exact["log_error"].mean())) if n else None,
    }
    for pct in ERROR_PERCENTILES:
        m[f"p{pct}_absolute_error_tokens"] = _q(abs_err, pct / 100)
        m[f"p{pct}_absolute_percentage_error"] = _q(exact["absolute_percentage_error"], pct / 100)
    directions = exact["error_direction"]
    for label in ("underprediction", "overprediction", "near_exact"):
        count = int((directions == label).sum())
        m[f"{label}_count"] = count
        m[f"{label}_rate_pct"] = 100.0 * count / n if n else None
    m["mean_underprediction_tokens"] = _mean(
        -exact.loc[directions == "underprediction", "signed_error_tokens"]
    )
    m["mean_overprediction_tokens"] = _mean(
        exact.loc[directions == "overprediction", "signed_error_tokens"]
    )
    ape = exact["absolute_percentage_error"]
    for threshold in CATASTROPHIC_PCT_THRESHOLDS:
        count = int((ape > threshold).sum())
        valid = int(ape.notna().sum())
        m[f"catastrophic_gt{int(threshold)}pct_count"] = count
        m[f"catastrophic_gt{int(threshold)}pct_rate_pct"] = 100.0 * count / valid if valid else None
    for threshold in CATASTROPHIC_TOKEN_THRESHOLDS:
        count = int((abs_err > threshold).sum())
        m[f"abs_error_gt{int(threshold)}tok_count"] = count
        m[f"abs_error_gt{int(threshold)}tok_rate_pct"] = 100.0 * count / n if n else None

    if "realized_absolute_error_tokens" in rows.columns:
        r = rows[rows["realized_absolute_error_tokens"].notna()]
        ry = r["actual_output_tokens"].astype(float)
        rtot = float(ry.sum())
        m["realized_n"] = int(len(r))
        m["realized_mae_tokens"] = _mean(r["realized_absolute_error_tokens"])
        m["realized_median_absolute_error_tokens"] = _q(r["realized_absolute_error_tokens"], 0.5)
        m["realized_rmse_tokens"] = (
            float(np.sqrt(r["realized_squared_error_tokens2"].mean())) if len(r) else None
        )
        m["realized_wape_pct"] = (
            100.0 * float(r["realized_absolute_error_tokens"].sum()) / rtot if rtot > 0 else None
        )
        m["realized_nrmse_pct"] = (
            100.0 * m["realized_rmse_tokens"] / float(ry.mean())
            if len(r) and ry.mean() > 0
            else None
        )
        m["realized_median_absolute_percentage_error"] = _q(
            r["realized_absolute_percentage_error"], 0.5
        )
        m["realized_smape_pct"] = _mean(r["realized_symmetric_percentage_error"])
        m["realized_mean_signed_error_tokens"] = _mean(r["realized_signed_error_tokens"])
        m["realized_aggregate_percentage_bias"] = (
            100.0 * float(r["realized_signed_error_tokens"].sum()) / rtot if rtot > 0 else None
        )
        for pct in ERROR_PERCENTILES:
            m[f"realized_p{pct}_absolute_error_tokens"] = _q(
                r["realized_absolute_error_tokens"], pct / 100
            )
            m[f"realized_p{pct}_absolute_percentage_error"] = _q(
                r["realized_absolute_percentage_error"], pct / 100
            )
        signed = r["realized_signed_error_tokens"]
        tol = np.maximum(NEAR_EXACT_TOKENS, NEAR_EXACT_FRACTION * ry)
        under = int(((signed < 0) & (signed.abs() > tol)).sum())
        over = int(((signed > 0) & (signed.abs() > tol)).sum())
        m["realized_underprediction_rate_pct"] = 100.0 * under / len(r) if len(r) else None
        m["realized_overprediction_rate_pct"] = 100.0 * over / len(r) if len(r) else None
        rape = r["realized_absolute_percentage_error"]
        for threshold in CATASTROPHIC_PCT_THRESHOLDS:
            valid = int(rape.notna().sum())
            count = int((rape > threshold).sum())
            m[f"realized_catastrophic_gt{int(threshold)}pct_count"] = count
            m[f"realized_catastrophic_gt{int(threshold)}pct_rate_pct"] = (
                100.0 * count / valid if valid else None
            )
        if "realized_interval_hit" in r.columns:
            m["realized_interval_coverage_pct"] = 100.0 * float(r["realized_interval_hit"].mean())
            pos = r["realized_interval_position"]
            m["realized_below_interval_rate_pct"] = 100.0 * float((pos == "below_interval").mean())
            m["realized_above_interval_rate_pct"] = 100.0 * float((pos == "above_interval").mean())

    cen = rows[censored]
    m["censored_known_underprediction_count"] = int(
        cen["prediction_below_known_lower_bound"].astype("boolean").fillna(False).sum()
    )
    m["censored_known_underprediction_rate_pct"] = (
        100.0 * m["censored_known_underprediction_count"] / len(cen) if len(cen) else None
    )
    m["censored_mean_lower_bound_error_tokens"] = (
        _mean(cen["lower_bound_error_tokens"]) if len(cen) else None
    )

    if "interval_position" in rows.columns:
        positions = rows["interval_position"]
        known = positions != "unknown_censored"
        hits = int((positions == "inside_interval").sum())
        below = int((positions == "below_interval").sum())
        above = int((positions == "above_interval").sum())
        unknown = int((~known).sum())
        k = int(known.sum())
        m["interval_hits"] = hits
        m["interval_misses"] = below + above
        m["interval_unknown_censored"] = unknown
        m["interval_coverage_pct"] = 100.0 * hits / k if k else None
        # Bounds if every unknown censored row were a hit / a miss.
        m["interval_coverage_pct_bounds"] = (
            [100.0 * hits / len(rows), 100.0 * (hits + unknown) / len(rows)] if len(rows) else None
        )
        m["below_interval_rate_pct"] = 100.0 * below / k if k else None
        m["above_interval_rate_pct"] = 100.0 * above / k if k else None
        m["mean_interval_width_tokens"] = _mean(rows["interval_width_tokens"])
        m["median_interval_width_tokens"] = _q(rows["interval_width_tokens"], 0.5)
        m["mean_interval_width_pct"] = _mean(rows["interval_width_percentage"])
        m["median_interval_width_pct"] = _q(rows["interval_width_percentage"], 0.5)
        if "confidence_margin_tokens" in rows.columns:
            m["mean_confidence_margin_tokens"] = _mean(rows["confidence_margin_tokens"])
            m["median_confidence_margin_tokens"] = _q(rows["confidence_margin_tokens"], 0.5)
            m["mean_confidence_margin_pct"] = _mean(rows["confidence_margin_pct"])
            m["median_confidence_margin_pct"] = _q(rows["confidence_margin_pct"], 0.5)
        if "confidence_level" in rows.columns and rows["confidence_level"].notna().any():
            nominal = 100.0 * float(rows["confidence_level"].dropna().iloc[0])
            m["nominal_coverage_pct"] = nominal
            m["coverage_error_pp"] = (
                m["interval_coverage_pct"] - nominal
                if m["interval_coverage_pct"] is not None
                else None
            )
            # Winkler/interval score (lower is better): width + 2/alpha * miss, on known rows.
            alpha = 1 - nominal / 100
            known_rows = rows[known]
            score = (
                known_rows["interval_width_tokens"]
                + (2 / alpha) * known_rows["interval_miss_tokens"]
            )
            m["mean_interval_score_tokens"] = _mean(score)

    if "cost_absolute_error_usd" in rows.columns:
        cost = rows[rows["cost_absolute_error_usd"].notna()]
        actual_cost = pd.to_numeric(cost["calculated_actual_cost_usd"], errors="coerce").astype(
            float
        )
        m["n_cost"] = int(len(cost))
        m["cost_mae_usd"] = _mean(cost["cost_absolute_error_usd"])
        m["cost_rmse_usd"] = (
            float(np.sqrt(cost["squared_cost_error_usd2"].mean())) if len(cost) else None
        )
        m["cost_mean_signed_error_usd"] = _mean(cost["cost_signed_error_usd"])
        m["cost_wape_pct"] = (
            100.0 * float(cost["cost_absolute_error_usd"].sum()) / float(actual_cost.sum())
            if len(cost) and actual_cost.sum() > 0
            else None
        )
        m["cost_median_ape_pct"] = _q(cost["cost_absolute_percentage_error"], 0.5)
        m["cost_signed_bias_pct"] = (
            100.0 * float(cost["cost_signed_error_usd"].sum()) / float(actual_cost.sum())
            if len(cost) and actual_cost.sum() > 0
            else None
        )
    for kind in ("total", "reasoning", "preflight_input"):
        col = f"{kind}_absolute_error_tokens"
        if col in rows.columns:
            present = rows[rows[col].notna()]
            m[f"{kind}_n"] = int(len(present))
            m[f"{kind}_mae_tokens"] = _mean(present[col])
            m[f"{kind}_wape_pct"] = _wape(present, kind)
            m[f"{kind}_mean_signed_error_tokens"] = _mean(present[f"{kind}_signed_error_tokens"])
    return m


def _wape(rows: pd.DataFrame, kind: str) -> float | None:
    actual_col = {
        "total": "actual_total_tokens",
        "reasoning": "actual_reasoning_tokens",
        "preflight_input": "actual_input_tokens",
    }[kind]
    actual = pd.to_numeric(rows[actual_col], errors="coerce").astype(float)
    if not len(rows) or actual.sum() <= 0:
        return None
    return 100.0 * float(rows[f"{kind}_absolute_error_tokens"].sum()) / float(actual.sum())
