"""Prediction-error metrics, censoring-aware diagnostics and group bootstrap.

Conventions
-----------
* ``error = predicted - actual`` (positive = over-prediction).
* Point metrics use **uncensored** rows only. A truncated response's observed
  output equals the cap, which is only a lower bound on the true length, so it is
  never treated as an exact label.
* Censored rows are scored separately: if ``predicted < cap`` the prediction is
  *known* to be too low (the true value exceeds the cap); likewise an interval
  whose upper bound is below the cap is *known* to miss.
* Uncertainty uses a **group bootstrap** (resampling leakage groups / prompt
  families, not rows), because rows of one family are correlated.
* Evaluating the heuristic baseline on all real rows is legitimate out-of-sample
  evaluation *of the baseline* (it was not fit to these data), but it is still
  subject to selection bias: the uncensored subset over-represents short outputs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

LENGTH_BUCKETS = [0, 50, 150, 400, 1000, 10**9]
LENGTH_LABELS = ["<=50w", "51-150w", "151-400w", "401-1000w", ">1000w"]


def point_metrics(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float | int | None]:
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    n = int(act.size)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "median_ae": None, "wape": None, "bias": None}
    error = pred - act
    abs_error = np.abs(error)
    total = float(np.sum(np.abs(act)))
    positive = act > 0
    return {
        "n": n,
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_ae": float(np.median(abs_error)),
        "wape": float(abs_error.sum() / total) if total > 0 else None,
        "bias": float(error.mean()),
        "median_ape": float(np.median(abs_error[positive] / act[positive]))
        if np.any(positive)
        else None,
    }


def interval_metrics(
    actual: ArrayLike, lower: ArrayLike, upper: ArrayLike
) -> dict[str, float | int | None]:
    act = np.asarray(actual, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if act.size == 0:
        return {"n": 0, "coverage": None, "mean_width": None, "median_width": None}
    inside = (act >= low) & (act <= high)
    width = high - low
    return {
        "n": int(act.size),
        "coverage": float(inside.mean()),
        "below_lower": float((act < low).mean()),
        "above_upper": float((act > high).mean()),
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
    }


def censored_diagnostics(
    cap: ArrayLike, predicted: ArrayLike, upper: ArrayLike | None = None
) -> dict[str, float | int | None]:
    caps = np.asarray(cap, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if caps.size == 0:
        return {"n": 0, "known_underprediction_rate": None, "known_interval_miss_rate": None}
    out: dict[str, float | int | None] = {
        "n": int(caps.size),
        "known_underprediction_rate": float((pred < caps).mean()),
        "mean_known_shortfall_lower_bound": float(np.maximum(caps - pred, 0).mean()),
    }
    if upper is not None:
        out["known_interval_miss_rate"] = float((np.asarray(upper, dtype=float) < caps).mean())
    return out


def group_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float | None],
    *,
    group_column: str,
    n_boot: int = 2000,
    seed: int = 7,
    level: float = 0.95,
) -> dict[str, float | int | None]:
    """Percentile CI of ``statistic`` resampling whole groups with replacement."""
    point = statistic(frame)
    groups = frame[group_column].astype(str).to_numpy()
    unique = np.unique(groups)
    if point is None or unique.size < 2:
        return {"estimate": point, "lower": None, "upper": None, "groups": int(unique.size)}
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index_by_group[g] for g in chosen])
        value = statistic(frame.iloc[rows])
        if value is not None:
            values.append(value)
    alpha = (1 - level) / 2
    return {
        "estimate": point,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "groups": int(unique.size),
        "level": level,
        "n_boot": n_boot,
    }


def _metric(name: str) -> Callable[[pd.DataFrame], float | None]:
    def compute(frame: pd.DataFrame) -> float | None:
        unc = frame[~frame["censored"]]
        if name == "coverage":
            values = interval_metrics(unc["actual"], unc["lower"], unc["upper"])["coverage"]
        else:
            values = point_metrics(unc["actual"], unc["predicted"])[name]
        return None if values is None else float(values)

    return compute


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    slices: tuple[str, ...] = ("prompt_category", "length_bucket", "requested_effort", "model"),
    group_column: str = "group",
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Evaluate predictions in ``frame``.

    Required columns: ``actual``, ``predicted``, ``lower``, ``upper``, ``censored``,
    ``cap`` and ``group_column``. Slices that are missing are skipped.
    """
    unc = frame[~frame["censored"]]
    cen = frame[frame["censored"]]
    result: dict[str, Any] = {
        "rows": int(len(frame)),
        "uncensored_rows": int(len(unc)),
        "censored_rows": int(len(cen)),
        "censor_rate": float(frame["censored"].mean()) if len(frame) else None,
        "point": point_metrics(unc["actual"], unc["predicted"]),
        "interval": interval_metrics(unc["actual"], unc["lower"], unc["upper"]),
        "censored": censored_diagnostics(cen["cap"], cen["predicted"], cen["upper"]),
        "bootstrap_95": {
            name: group_bootstrap(frame, _metric(name), group_column=group_column, n_boot=n_boot)
            for name in ("mae", "bias", "coverage")
        }
        if len(unc)
        else {},
        "slices": {},
    }
    for column in slices:
        if column not in frame.columns:
            continue
        table: dict[str, Any] = {}
        for level, part in frame.groupby(frame[column].fillna("unset").astype(str), sort=True):
            p_unc = part[~part["censored"]]
            p_cen = part[part["censored"]]
            table[str(level)] = {
                "rows": int(len(part)),
                "censor_rate": float(part["censored"].mean()),
                "point": point_metrics(p_unc["actual"], p_unc["predicted"]),
                "interval": interval_metrics(p_unc["actual"], p_unc["lower"], p_unc["upper"]),
                "censored": censored_diagnostics(p_cen["cap"], p_cen["predicted"], p_cen["upper"]),
                "cost_usd": float(part["cost_usd"].sum()) if "cost_usd" in part else None,
                "cost_per_usable_usd": float(part["cost_usd"].sum() / len(p_unc))
                if "cost_usd" in part and len(p_unc)
                else None,
            }
        result["slices"][column] = table
    return result


def real_observation_frame(root: str | Path) -> pd.DataFrame:
    """Validated real controlled-experiment observations with request-time context."""
    from seshat.data.io import read_parquet
    from seshat.data.storage import StorageLayout

    layout = StorageLayout(root)
    observations = layout.read_frame("observations")
    decisions_path = layout.validation / "decisions.parquet"
    if observations.empty or not decisions_path.exists():
        raise ValueError("Need validated observations (run consolidate/validate first)")
    decisions = read_parquet(decisions_path)[
        ["observation_id", "eligible_for_training", "data_origin"]
    ]
    obs = observations.merge(decisions, on="observation_id", how="inner")
    obs = obs[
        obs["eligible_for_training"]
        & (obs["data_origin"] == "real")
        & (obs["source_type"] == "controlled_experiment")
        & obs["actual_output_tokens"].notna()
    ]
    requests = layout.read_frame("requests").drop_duplicates("sample_id")
    cols = [
        "sample_id",
        "prompt_word_count",
        "prompt_category",
        "prompt_family_id",
        "requested_effort",
        "max_output_tokens",
        "design_cell_id",
    ]
    frame = obs.merge(requests[cols], on="sample_id", how="inner", validate="many_to_one")
    frame["censored"] = frame["is_truncated"].fillna(False).astype(bool) | (
        frame["label_quality"] == "censored"
    )
    frame["length_bucket"] = pd.cut(
        frame["prompt_word_count"].astype(float), LENGTH_BUCKETS, labels=LENGTH_LABELS
    ).astype(str)
    frame["model"] = frame["provider"] + "/" + frame["requested_model"]
    frame["cost_usd"] = pd.to_numeric(frame["calculated_cost_usd"], errors="coerce").fillna(0.0)
    return pd.DataFrame(frame.reset_index(drop=True))


def evaluate_baseline(root: str | Path, *, n_boot: int = 2000) -> dict[str, Any]:
    """Score the heuristic baseline on every real observation (diagnostic)."""
    from seshat.predictor import BaselinePredictor

    frame = real_observation_frame(root)
    if frame.empty:
        raise ValueError("No real controlled-experiment observations to evaluate")
    predictor = BaselinePredictor()
    preds = [
        predictor.predict_from_word_count(int(w), m, e)
        for w, m, e in zip(
            frame["prompt_word_count"],
            frame["requested_model"],
            frame["requested_effort"].where(frame["requested_effort"].notna(), None),
            strict=True,
        )
    ]
    scored = pd.DataFrame(
        {
            "actual": frame["actual_output_tokens"].astype(float),
            "predicted": [p.predicted_output_tokens for p in preds],
            "lower": [p.lower_bound for p in preds],
            "upper": [p.upper_bound for p in preds],
            "censored": frame["censored"],
            "cap": frame["max_output_tokens"].astype(float),
            "group": frame["prompt_family_id"].fillna(frame["sample_id"].astype(str)),
            "prompt_category": frame["prompt_category"],
            "length_bucket": frame["length_bucket"],
            "requested_effort": frame["requested_effort"],
            "model": frame["model"],
            "cost_usd": frame["cost_usd"],
        }
    )
    report = evaluate_frame(scored, n_boot=n_boot)
    report["predictor_version"] = predictor.version
    report["target"] = "actual_output_tokens (provider-reported; includes reasoning when the "
    report["target"] += "provider counts reasoning inside output tokens)"
    report["reasoning_included_in_output"] = sorted(
        str(v) for v in frame["reasoning_included_in_output"].dropna().unique()
    )
    report["nominal_interval_meaning"] = (
        "Baseline interval is +/-65% of the estimate (min 24 tokens) and is NOT calibrated; "
        "its displayed confidence 0.35 has no empirical meaning."
    )
    report["caveat"] = (
        "Out-of-sample for the (unfitted) heuristic, but point metrics cover uncensored rows "
        "only, which over-represent short outputs (selection bias)."
    )
    return report


def render_evaluation_markdown(report: dict[str, Any], title: str) -> str:
    def fmt(value: Any, digits: int = 1) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    point, interval, cens = report["point"], report["interval"], report["censored"]
    lines = [
        f"# {title}",
        "",
        f"Rows: {report['rows']} (uncensored {report['uncensored_rows']}, censored "
        f"{report['censored_rows']}, censor rate {fmt(report['censor_rate'], 3)})",
        "",
        "| Metric (uncensored rows) | Value | 95% group-bootstrap CI |",
        "|---|---:|---|",
    ]
    boot = report.get("bootstrap_95", {})
    for key, label in (
        ("mae", "MAE (tokens)"),
        ("rmse", "RMSE"),
        ("median_ae", "Median AE"),
        ("wape", "WAPE"),
        ("bias", "Mean signed error (pred - actual)"),
        ("median_ape", "Median APE"),
    ):
        ci = boot.get(key)
        ci_text = f"[{fmt(ci['lower'])}, {fmt(ci['upper'])}]" if ci and ci["lower"] else ""
        lines.append(
            f"| {label} | {fmt(point.get(key), 3 if key.endswith('ape') else 1)} | {ci_text} |"
        )
    cov = boot.get("coverage")
    lines += [
        f"| Interval coverage | {fmt(interval.get('coverage'), 3)} | "
        + (f"[{fmt(cov['lower'], 3)}, {fmt(cov['upper'], 3)}]" if cov and cov["lower"] else "")
        + " |",
        f"| Mean interval width | {fmt(interval.get('mean_width'))} | |",
        "",
        f"Censored rows: prediction below the cap (known under-prediction) in "
        f"{fmt(cens.get('known_underprediction_rate'), 3)}; interval upper bound below the cap "
        f"(known miss) in {fmt(cens.get('known_interval_miss_rate'), 3)}.",
        "",
    ]
    for column, table in report["slices"].items():
        lines += [
            f"## By {column}",
            "",
            "| Level | Rows | Censor rate | Uncensored n | MAE | Bias | Coverage "
            "| Known under-pred. (censored) | Cost / usable obs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for level, info in table.items():
            per_usable = info["cost_per_usable_usd"]
            cost_text = f"${per_usable:.4f}" if per_usable else "-"
            lines.append(
                f"| {level} | {info['rows']} | {fmt(info['censor_rate'], 2)} | "
                f"{info['point']['n']} | {fmt(info['point']['mae'])} | "
                f"{fmt(info['point']['bias'])} | {fmt(info['interval']['coverage'], 2)} | "
                f"{fmt(info['censored'].get('known_underprediction_rate'), 2)} | "
                f"{cost_text} |"
            )
        lines.append("")
    for key in ("nominal_interval_meaning", "caveat"):
        if key in report:
            lines.append(f"_{report[key]}_")
            lines.append("")
    return "\n".join(lines)
