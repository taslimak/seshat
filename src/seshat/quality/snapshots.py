"""Model-quality snapshots, rolling windows, drift detection and run reports.

Snapshots (``<root>/quality/model_quality_snapshots.parquet``) hold one row per
(predictor version, label origin, window, slice) with ``n_observations``,
``n_uncensored``, a ``reliable`` flag and every aggregate from
:func:`seshat.quality.metrics.aggregate_metrics` (JSON) plus headline columns.
Slices with fewer than :data:`MIN_RELIABLE_N` rows are kept but flagged, with
``fallback_slice`` naming the parent slice (model+category+effort -> model+category
-> model -> global) whose statistics should be shown instead.

Windows: lifetime, the last 50 / 100 / 500 outcomes (by observation time) and the
trailing 7 / 30 days before ``as_of``. Count windows answer "how good is it on the
latest N requests"; time windows answer "how good was it this week".

Only out-of-sample outcomes (``valid_out_of_sample``) enter snapshots, and real and
synthetic origins are never pooled.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from seshat.quality.metrics import METRICS_VERSION, aggregate_metrics

MIN_RELIABLE_N = 20
COUNT_WINDOWS = (50, 100, 500)
TIME_WINDOWS_DAYS = (7, 30)
SLICES: tuple[tuple[str, ...], ...] = (
    (),
    ("provider_model",),
    ("provider_model", "effort"),
    ("provider_model", "prompt_category"),
    ("provider_model", "prompt_category", "effort"),
    ("provider_model", "task_group"),
    ("prompt_category",),
    ("prompt_subcategory",),
    ("task_group",),
    ("effort",),
    ("prompt_length_band",),
    ("predicted_output_band",),
    ("actual_output_band",),
    ("requested_output_format",),
    ("tool_use",),
    ("context_band",),
    ("experiment_id",),
)
HEADLINE = (
    "realized_mae_tokens",
    "realized_wape_pct",
    "realized_aggregate_percentage_bias",
    "realized_interval_coverage_pct",
    "mae_tokens",
    "wape_pct",
    "mean_signed_error_tokens",
    "interval_coverage_pct",
    "median_confidence_margin_pct",
    "censoring_rate_pct",
)
DRIFT_MIN_N = 30
DRIFT_ALPHA = 0.01


def _windows(frame: pd.DataFrame, as_of: datetime) -> list[tuple[str, pd.DataFrame]]:
    ordered = frame.sort_values("timestamp_actual_observed")
    out = [("lifetime", ordered)]
    for n in COUNT_WINDOWS:
        if len(ordered) > n:
            out.append((f"last_{n}", ordered.tail(n)))
    observed = pd.to_datetime(ordered["timestamp_actual_observed"], utc=True)
    for days in TIME_WINDOWS_DAYS:
        part = ordered[observed > pd.Timestamp(as_of - timedelta(days=days))]
        if len(part):
            out.append((f"trailing_{days}d", part))
    return out


def _parent(dims: tuple[str, ...]) -> tuple[str, ...] | None:
    return dims[:-1] if dims else None


def quality_snapshots(outcomes: pd.DataFrame, *, as_of: datetime | None = None) -> pd.DataFrame:
    as_of = as_of or datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    valid = outcomes[outcomes["valid_out_of_sample"].astype(bool)] if len(outcomes) else outcomes
    if len(valid):
        valid = valid.assign(calibration_version=valid["calibration_version"].fillna("none"))
    for (version, _cal, origin), group in valid.groupby(
        ["predictor_version", "calibration_version", "label_origin"]
    ):
        meta = group.iloc[0]
        for window, part in _windows(group, as_of):
            for dims in SLICES:
                keyed = part.groupby([part[d].astype(str) for d in dims]) if dims else [((), part)]
                for key, sub in keyed:
                    key_t = key if isinstance(key, tuple) else (key,)
                    metrics = aggregate_metrics(sub)
                    n_unc = metrics["n_uncensored"]
                    reliable = metrics["n_observations"] >= MIN_RELIABLE_N
                    parent = _parent(dims)
                    rows.append(
                        {
                            "as_of": as_of.isoformat(),
                            "metrics_version": METRICS_VERSION,
                            "predictor_version": str(version),
                            "uncertainty_model_version": meta.get("uncertainty_model_version"),
                            "calibration_version": meta.get("calibration_version"),
                            "label_origin": str(origin),
                            "window": window,
                            "slice_dims": "+".join(dims) or "global",
                            "slice_key": "|".join(
                                f"{d}={v}" for d, v in zip(dims, key_t, strict=True)
                            )
                            or "all",
                            "n_observations": metrics["n_observations"],
                            "n_uncensored": n_unc,
                            "reliable": reliable,
                            "fallback_slice": None
                            if reliable or parent is None
                            else ("+".join(parent) or "global"),
                            "first_observed": str(sub["timestamp_actual_observed"].min()),
                            "last_observed": str(sub["timestamp_actual_observed"].max()),
                            **{h: metrics.get(h) for h in HEADLINE},
                            "metrics_json": json.dumps(metrics, default=_json),
                        }
                    )
    return pd.DataFrame(rows)


def drift_report(
    outcomes: pd.DataFrame, predictor_version: str, *, label_origin: str = "real"
) -> dict[str, Any]:
    """Compare the most recent outcomes with the model's calibration baseline.

    Reference = out-of-fold / protected-test backtest outcomes (what the model was
    calibrated and accepted on). Recent = the last ``DRIFT_MIN_N``.. outcomes that are
    not part of that reference (new preflight forecasts, or the newest experiment
    when only backtests exist). Flags require ``n >= DRIFT_MIN_N`` per side and
    ``p < DRIFT_ALPHA``; otherwise the status is ``insufficient_data``.
    """
    from scipy import stats

    rows = outcomes[
        (outcomes["predictor_version"] == predictor_version)
        & (outcomes["label_origin"] == label_origin)
        & outcomes["valid_out_of_sample"].astype(bool)
    ].sort_values("timestamp_actual_observed")
    reference = rows[rows["evaluation_role"].isin(["cross_validation", "protected_test"])]
    recent = rows[~rows.index.isin(reference.index)]
    basis = "new forecasts after training"
    if (recent.empty or len(reference) < DRIFT_MIN_N) and rows["experiment_id"].nunique() > 1:
        latest = rows["experiment_id"].iloc[-1]
        recent = rows[rows["experiment_id"] == latest]
        reference = rows[rows["experiment_id"] != latest]
        basis = f"newest experiment ({latest}) vs earlier experiments (backtests only)"
    report: dict[str, Any] = {
        "predictor_version": predictor_version,
        "label_origin": label_origin,
        "basis": basis,
        "n_reference": int(len(reference)),
        "n_recent": int(len(recent)),
        "min_n": DRIFT_MIN_N,
        "alpha": DRIFT_ALPHA,
        "tests": {},
    }
    if len(reference) < DRIFT_MIN_N or len(recent) < DRIFT_MIN_N:
        report["status"] = "insufficient_data"
        report["reason"] = f"need >= {DRIFT_MIN_N} outcomes on each side"
        return report
    flags = []
    ref_err = reference["realized_log_error"].abs().dropna()
    rec_err = recent["realized_log_error"].abs().dropna()
    u = stats.mannwhitneyu(rec_err, ref_err, alternative="greater")
    report["tests"]["abs_log_error_increase"] = {
        "reference_median": float(ref_err.median()),
        "recent_median": float(rec_err.median()),
        "p_value": float(u.pvalue),
    }
    if u.pvalue < DRIFT_ALPHA:
        flags.append("error distribution worsened")
    t = stats.ttest_ind(
        recent["realized_log_error"].dropna(),
        reference["realized_log_error"].dropna(),
        equal_var=False,
    )
    report["tests"]["bias_shift"] = {
        "reference_mean_log_error": float(reference["realized_log_error"].mean()),
        "recent_mean_log_error": float(recent["realized_log_error"].mean()),
        "p_value": float(t.pvalue),
    }
    if t.pvalue < DRIFT_ALPHA:
        flags.append("bias shifted")
    if "realized_interval_hit" in recent.columns and recent["confidence_level"].notna().any():
        nominal = float(recent["confidence_level"].dropna().iloc[0])
        hits = int(recent["realized_interval_hit"].astype(bool).sum())
        b = stats.binomtest(hits, len(recent), nominal, alternative="less")
        report["tests"]["coverage_below_nominal"] = {
            "nominal": nominal,
            "recent_coverage": hits / len(recent),
            "p_value": float(b.pvalue),
        }
        if b.pvalue < DRIFT_ALPHA:
            flags.append("interval coverage below nominal")
    cat_ref = int((reference["realized_absolute_percentage_error"] > 100).sum())
    cat_rec = int((recent["realized_absolute_percentage_error"] > 100).sum())
    table = [[cat_rec, len(recent) - cat_rec], [cat_ref, len(reference) - cat_ref]]
    fisher = stats.fisher_exact(table, alternative="greater")
    report["tests"]["catastrophic_rate_increase"] = {
        "reference_rate": cat_ref / len(reference),
        "recent_rate": cat_rec / len(recent),
        "p_value": float(fisher.pvalue),
    }
    if fisher.pvalue < DRIFT_ALPHA:
        flags.append("catastrophic misses increased")
    report["status"] = "drift" if flags else "ok"
    report["flags"] = flags
    report["recommendation"] = (
        "Recalibrate and evaluate a challenger on new data; do not promote without gates."
        if flags
        else "No retraining needed on drift grounds."
    )
    return report


def segment_quality(outcomes: pd.DataFrame, predictor_version: str) -> pd.DataFrame:
    """Per inference-time segment error statistics (for adaptive collection)."""
    rows = outcomes[
        (outcomes["predictor_version"] == predictor_version)
        & (outcomes["label_origin"] == "real")
        & outcomes["valid_out_of_sample"].astype(bool)
    ]
    records = []
    for category, part in rows.groupby("prompt_category"):
        m = aggregate_metrics(part)
        records.append(
            {
                "prompt_category": category,
                "n": m["n_observations"],
                "realized_wape_pct": m.get("realized_wape_pct"),
                "realized_bias_pct": m.get("realized_aggregate_percentage_bias"),
                "realized_coverage_pct": m.get("realized_interval_coverage_pct"),
                "median_margin_pct": m.get("median_confidence_margin_pct"),
                "catastrophic_rate_pct": m.get("realized_catastrophic_gt100pct_rate_pct"),
                "mean_cost_usd": float(part["calculated_actual_cost_usd"].mean()),
            }
        )
    return pd.DataFrame(records)


def render_run_report(rows: pd.DataFrame, title: str) -> str:
    m = aggregate_metrics(rows)
    meta = rows.iloc[0]

    def f(key: str, digits: int = 1, suffix: str = "") -> str:
        value = m.get(key)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        return f"{value:,.{digits}f}{suffix}"

    lines = [
        f"# {title}",
        "",
        f"- Predictor `{meta['predictor_version']}` · uncertainty "
        f"`{meta.get('uncertainty_model_version') or 'n/a'}` · calibration "
        f"`{meta.get('calibration_version') or 'n/a'}` · label origin `{meta['label_origin']}`",
        f"- Observations: {m['n_observations']} (uncensored {m['n_uncensored']}, censored "
        f"{m['n_censored']} / {f('censoring_rate_pct', 1, '%')})",
        "",
        "## Realized usage (exact for every row; primary)",
        "",
        f"- MAE {f('realized_mae_tokens')} tokens · WAPE {f('realized_wape_pct', 1, '%')}",
        f"- Median AE {f('realized_median_absolute_error_tokens')} tokens · median APE "
        f"{f('realized_median_absolute_percentage_error', 1, '%')}",
        f"- RMSE {f('realized_rmse_tokens')} tokens · nRMSE {f('realized_nrmse_pct', 1, '%')}",
        f"- Bias {f('realized_mean_signed_error_tokens')} tokens · "
        f"{f('realized_aggregate_percentage_bias', 1, '%')}",
        f"- Under-prediction {f('realized_underprediction_rate_pct', 1, '%')} · over-prediction "
        f"{f('realized_overprediction_rate_pct', 1, '%')}",
        f"- p90 absolute error {f('realized_p90_absolute_error_tokens')} tokens / "
        f"{f('realized_p90_absolute_percentage_error', 1, '%')}",
        f"- Catastrophic >25% {f('realized_catastrophic_gt25pct_count', 0)} / "
        f"{f('realized_catastrophic_gt25pct_rate_pct', 1, '%')} · >50% "
        f"{f('realized_catastrophic_gt50pct_count', 0)} / "
        f"{f('realized_catastrophic_gt50pct_rate_pct', 1, '%')} · >100% "
        f"{f('realized_catastrophic_gt100pct_count', 0)} / "
        f"{f('realized_catastrophic_gt100pct_rate_pct', 1, '%')}",
        f"- Nominal coverage {f('nominal_coverage_pct', 0, '%')} · actual (realized) "
        f"{f('realized_interval_coverage_pct', 1, '%')}",
        "",
        "## Latent output length (uncensored rows only; censored rows are lower bounds)",
        "",
        f"- MAE {f('mae_tokens')} tokens · WAPE {f('wape_pct', 1, '%')} · median AE "
        f"{f('median_absolute_error_tokens')} · RMSE {f('rmse_tokens')} / nRMSE "
        f"{f('nrmse_pct', 1, '%')}",
        f"- Bias {f('mean_signed_error_tokens')} tokens / {f('aggregate_percentage_bias', 1, '%')}",
        f"- p90 absolute error {f('p90_absolute_error_tokens')} tokens / "
        f"{f('p90_absolute_percentage_error', 1, '%')}",
        f"- Interval coverage {f('interval_coverage_pct', 1, '%')} (bounds incl. censored "
        f"{m.get('interval_coverage_pct_bounds')}) · below {f('below_interval_rate_pct', 1, '%')}"
        f" · above {f('above_interval_rate_pct', 1, '%')}",
        f"- Censored rows predicted below the known lower bound: "
        f"{f('censored_known_underprediction_count', 0)} / "
        f"{f('censored_known_underprediction_rate_pct', 1, '%')}",
        "",
        "## Confidence margin",
        "",
        f"- Average ±{f('mean_confidence_margin_tokens')} tokens / "
        f"±{f('mean_confidence_margin_pct', 1, '%')}"
        f" · median ±{f('median_confidence_margin_tokens')} tokens / "
        f"±{f('median_confidence_margin_pct', 1, '%')}",
        f"- Interval width median {f('median_interval_width_tokens')} tokens / "
        f"{f('median_interval_width_pct', 1, '%')} · interval score "
        f"{f('mean_interval_score_tokens')}",
        "",
        "## Cost (price-table estimates, not invoices)",
        "",
        f"- Cost MAE ${f('cost_mae_usd', 5)} · WAPE {f('cost_wape_pct', 1, '%')} · signed bias "
        f"{f('cost_signed_bias_pct', 1, '%')}",
        "",
    ]
    return "\n".join(lines)


def _json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
