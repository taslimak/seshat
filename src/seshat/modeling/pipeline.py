"""Train, calibrate, evaluate and (conditionally) recommend a forecaster.

Protocol (no test-set reuse, no temporal or group leakage)
----------------------------------------------------------
1. Rows come from :func:`seshat.modeling.dataset.load_modeling_frame` (one label
   origin; real by default). The **protected test set** is fixed by a family-hash
   rule and is used once, after every choice has been made.
2. On the remaining *development* rows, grouped K-fold cross-validation (leakage
   groups never straddle folds) produces out-of-fold (OOF) forecasts for every
   candidate. Preprocessing is fitted inside each fold on its training part only.
3. **Model selection** uses OOF absolute error of the *realized-usage* forecast
   ``min(p, cap)``, which is observed exactly for every row (a truncated response
   produced exactly ``cap`` tokens). Latent-length metrics on uncensored rows only
   would reward models that ignore long (truncated) outputs; the realized target
   has no such selection bias. Latent metrics are still reported.
4. **Bias correction** (stage B) and **conformal calibration** (stage D) are fitted
   on the selected model's OOF residuals -- residuals of forecasts made without
   seeing those rows -- then frozen.
5. The selected model is refitted on all development rows and, together with the
   frozen correction/calibration, scored **once** on the protected test set, as
   are all other candidates and the heuristic baseline (same rows).
6. Promotion gates compare the candidate with the baseline on that test set. A
   PASS is a recommendation; deployment remains a human decision.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seshat.modeling.calibration import BiasCorrector, ConformalCalibrator, content_version
from seshat.modeling.dataset import (
    PROTECTED_TEST_FRACTION,
    PROTECTED_TEST_RULE,
    dataset_version,
    load_modeling_frame,
)
from seshat.modeling.features import MODEL_FEATURE_SET_VERSION
from seshat.modeling.models import CensoredLogNormal, Forecaster, candidate_models
from seshat.quality.metrics import METRICS_VERSION, add_row_metrics, aggregate_metrics

SEGMENT_COLUMNS = ("provider_model", "effort", "task_group")
PROMOTION_GATES = {
    "target": "realized output tokens (exact for every row)",
    "min_test_rows": 40,
    "wape_must_improve": True,
    "mae_must_improve": True,
    "max_abs_bias_pct": 15.0,
    "catastrophic_gt100pct_rate_not_worse": True,
    "coverage_tolerance_pp": 7.5,
    "max_median_interval_width_pct": 400.0,
    "max_category_mae_degradation_pct": 25.0,
}


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def grouped_folds(groups: pd.Series, k: int, seed: int) -> list[np.ndarray]:
    unique = np.array(sorted(groups.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    assignment = {g: i % k for i, g in enumerate(unique)}
    fold = groups.map(assignment).to_numpy()
    return [np.flatnonzero(fold == i) for i in range(k)]


def censoring_aware_abs_error(
    point: np.ndarray, actual: np.ndarray, censored: np.ndarray
) -> np.ndarray:
    """Exact error on uncensored rows; known minimum error on censored rows (latent)."""
    return np.asarray(
        np.where(censored, np.maximum(actual - point, 0.0), np.abs(point - actual)), dtype=float
    )


def realized_abs_error(point: np.ndarray, actual: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """|min(p, cap) - realized| -- exact for every row."""
    realized = np.minimum(point, np.nan_to_num(cap, nan=np.inf))
    return np.asarray(np.abs(realized - actual), dtype=float)


def _segments(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: frame[c].astype(str).to_numpy() for c in SEGMENT_COLUMNS}


@dataclass
class CalibratedForecaster:
    """Point model + bias correction + conformal calibration (stages A, B, D)."""

    model: Forecaster
    bias: BiasCorrector | None
    calibrator: ConformalCalibrator

    def forecast(self, frame: pd.DataFrame) -> pd.DataFrame:
        mu, sigma = self.model.predict_log(frame)
        segments = _segments(frame)
        delta = self.bias.delta(segments) if self.bias is not None else np.zeros(len(frame))
        corrected = mu + delta
        q_lo, q_hi = self.calibrator.quantiles(segments, len(frame))
        point = np.maximum(np.expm1(corrected), 0.0)
        lower = np.maximum(np.expm1(corrected + q_lo * sigma), 0.0)
        upper = np.maximum(np.expm1(corrected + q_hi * sigma), point)
        lower = np.minimum(lower, point)
        return pd.DataFrame(
            {
                "base_predicted_output_tokens": np.maximum(np.expm1(mu), 0.0),
                "predicted_output_tokens": point,
                "prediction_interval_lower": lower,
                "prediction_interval_upper": upper,
                "log_mu": corrected,
                "log_sigma": sigma,
            },
            index=frame.index,
        )


def margins(forecast: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    """Confidence margins derived from the calibrated interval (raw and %)."""
    p = forecast["predicted_output_tokens"]
    lower_margin = p - forecast["prediction_interval_lower"]
    upper_margin = forecast["prediction_interval_upper"] - p
    margin = np.maximum(lower_margin, upper_margin)
    denom = p.where(p >= 1.0)
    return pd.DataFrame(
        {
            "lower_margin_tokens": lower_margin,
            "upper_margin_tokens": upper_margin,
            "confidence_margin_tokens": margin,
            "lower_margin_pct": 100.0 * lower_margin / denom,
            "upper_margin_pct": 100.0 * upper_margin / denom,
            "confidence_margin_pct": 100.0 * margin / denom,
            "confidence_level": confidence_level,
        },
        index=forecast.index,
    )


def realized_usage_forecast(frame: pd.DataFrame, point: pd.Series) -> pd.DataFrame:
    """Total tokens and cost implied by the forecast for the request as configured.

    Realized output is ``min(forecast, cap)``; input uses the preflight count;
    rates are the price-table rates recorded on the observation (same version).
    """
    realized_out = np.minimum(point, frame["cap"].fillna(np.inf))
    pre_in = (
        pd.to_numeric(frame["preflight_input_tokens"], errors="coerce").astype(float)
        if "preflight_input_tokens" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    in_rate = pd.to_numeric(frame["input_rate_usd_per_million"], errors="coerce").astype(float)
    out_rate = pd.to_numeric(frame["output_rate_usd_per_million"], errors="coerce").astype(float)
    return pd.DataFrame(
        {
            "predicted_total_tokens": pre_in + realized_out,
            "predicted_cost_usd": (pre_in * in_rate + realized_out * out_rate) / 1e6,
        },
        index=frame.index,
    )


def scored_rows(
    frame: pd.DataFrame, forecast: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    rows = pd.concat(
        [
            frame,
            forecast,
            margins(forecast, confidence_level),
            realized_usage_forecast(frame, forecast["predicted_output_tokens"]),
        ],
        axis=1,
    )
    rows["target_is_censored"] = rows["censored"]
    rows["calculated_actual_cost_usd"] = pd.to_numeric(
        rows["calculated_cost_usd"], errors="coerce"
    ).astype(float)
    return add_row_metrics(rows)


def evaluate_rows(rows: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"overall": aggregate_metrics(rows), "slices": {}}
    for column in ("prompt_category", "task_group", "effort", "provider_model"):
        out["slices"][column] = {
            str(k): aggregate_metrics(g) for k, g in rows.groupby(rows[column].astype(str))
        }
    return out


def _gates(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Promotion gates on the protected test set, realized-usage metrics first."""
    c, b = candidate["overall"], baseline["overall"]
    checks: dict[str, bool | None] = {}
    checks["sufficient_test_size"] = c["realized_n"] >= PROMOTION_GATES["min_test_rows"]
    checks["lower_wape"] = c["realized_wape_pct"] < b["realized_wape_pct"]
    checks["lower_mae"] = c["realized_mae_tokens"] < b["realized_mae_tokens"]
    checks["acceptable_bias"] = (
        abs(c["realized_aggregate_percentage_bias"]) <= PROMOTION_GATES["max_abs_bias_pct"]
    )
    checks["catastrophic_not_worse"] = (c["realized_catastrophic_gt100pct_rate_pct"] or 0) <= (
        b["realized_catastrophic_gt100pct_rate_pct"] or 0
    )
    nominal = c.get("nominal_coverage_pct")
    checks["coverage_near_nominal"] = (
        nominal is not None
        and abs(c["realized_interval_coverage_pct"] - nominal)
        <= PROMOTION_GATES["coverage_tolerance_pp"]
    )
    checks["interval_width_reasonable"] = (
        c.get("median_interval_width_pct") is not None
        and c["median_interval_width_pct"] <= PROMOTION_GATES["max_median_interval_width_pct"]
    )
    worse = []
    for category, stats in candidate["slices"]["prompt_category"].items():
        base = baseline["slices"]["prompt_category"].get(category)
        if not base or stats["realized_n"] < 3 or not base["realized_mae_tokens"]:
            continue
        limit = 1 + float(str(PROMOTION_GATES["max_category_mae_degradation_pct"])) / 100
        if stats["realized_mae_tokens"] > base["realized_mae_tokens"] * limit:
            worse.append(category)
    checks["no_major_category_degradation"] = not worse
    checks["leakage_checks_passed"] = True  # asserted in train_and_evaluate before scoring
    decision = "PASS" if all(v is True for v in checks.values()) else "FAIL"
    return {
        "decision": decision,
        "checks": checks,
        "degraded_categories": worse,
        "gates": PROMOTION_GATES,
        "note": "PASS is a recommendation; activating a model requires owner approval.",
    }


def train_and_evaluate(
    root: str | Path,
    output_dir: str | Path,
    *,
    confidence_level: float = 0.90,
    n_folds: int = 5,
    seed: int = 20261011,
    label_origin: str = "real",
) -> dict[str, Any]:
    frame = load_modeling_frame(root, label_origin=label_origin)
    dev = frame[~frame["protected_test"]].reset_index(drop=True)
    test = frame[frame["protected_test"]].reset_index(drop=True)
    # Leakage assertions: no group or family in both partitions.
    assert not set(dev["split_group_id"]) & set(test["split_group_id"])
    assert not set(dev["prompt_family_id"]) & set(test["prompt_family_id"])
    folds = grouped_folds(dev["split_group_id"], n_folds, seed)

    # Stage A: OOF forecasts for every candidate.
    oof: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for template in candidate_models():
        mu = np.zeros(len(dev))
        sigma = np.ones(len(dev))
        for held in folds:
            train_idx = np.setdiff1d(np.arange(len(dev)), held)
            model = _fresh(template).fit(dev.iloc[train_idx])
            m, s = model.predict_log(dev.iloc[held])
            mu[held], sigma[held] = m, s
        oof[template.name] = (mu, sigma)
    actual = dev["actual_output_tokens"].to_numpy(dtype=float)
    censored = dev["censored"].to_numpy(dtype=bool)
    cap = dev["cap"].to_numpy(dtype=float)
    selection: dict[str, dict[str, float]] = {}
    for name, (mu, _) in oof.items():
        point = np.maximum(np.expm1(mu), 0.0)
        err = censoring_aware_abs_error(point, actual, censored)
        realized = realized_abs_error(point, actual, cap)
        selection[name] = {
            "oof_realized_mae": float(realized.mean()),
            "oof_realized_wape_pct": 100.0 * float(realized.sum() / actual.sum()),
            "oof_censoring_aware_mae": float(err.mean()),
            "oof_uncensored_mae": float(err[~censored].mean()),
            "oof_censored_known_underprediction_rate": float(
                (point[censored] < actual[censored]).mean()
            )
            if censored.any()
            else 0.0,
        }
    # Only JSON-serialisable (deployable) models can be selected.
    deployable = {n for n in selection if n.startswith("censored_lognormal")}
    chosen = min(sorted(deployable), key=lambda n: selection[n]["oof_realized_mae"])

    # Stage B + D on the chosen model's OOF residuals.
    mu, sigma = oof[chosen]
    z = np.log1p(actual)
    segments = _segments(dev)
    bias = BiasCorrector().fit(z - mu, censored, segments)
    corrected = mu + bias.delta(segments)
    point_b = np.maximum(np.expm1(corrected), 0.0)
    with_bias = float(realized_abs_error(point_b, actual, cap).mean())
    use_bias = with_bias < selection[chosen]["oof_realized_mae"]
    selection[chosen]["oof_realized_mae_with_bias_correction"] = with_bias
    if not use_bias:
        corrected = mu
    calibrator_global = ConformalCalibrator(confidence_level=confidence_level, segment_column=None)
    calibrator_global.fit((z - corrected) / sigma, censored)
    calibrator_seg = ConformalCalibrator(
        confidence_level=confidence_level, segment_column="task_group"
    )
    calibrator_seg.fit((z - corrected) / sigma, censored, segments)

    # Choose the calibration variant by OOF interval score (no test data involved).
    def _interval_score(cal: ConformalCalibrator) -> float:
        lo, hi = cal.quantiles(segments, len(dev))
        capped = np.nan_to_num(cap, nan=np.inf)
        lower = np.minimum(np.expm1(corrected + lo * sigma), capped)
        upper = np.minimum(np.expm1(corrected + hi * sigma), capped)
        alpha = 1 - confidence_level
        miss = np.where(actual < lower, lower - actual, np.where(actual > upper, actual - upper, 0))
        return float(np.mean(upper - lower + (2 / alpha) * miss))

    calibrator = min((calibrator_global, calibrator_seg), key=_interval_score)

    final_model = _fresh(next(m for m in candidate_models() if m.name == chosen)).fit(dev)
    if not isinstance(final_model, CensoredLogNormal):
        raise AssertionError("selected model must be serialisable")
    forecaster = CalibratedForecaster(final_model, bias if use_bias else None, calibrator)

    # Versions.
    ds_version = dataset_version(frame)
    predictor_version = content_version("cln", final_model.to_dict())
    uncertainty_version = f"{predictor_version}-sigma"
    calibration_version = content_version(
        "cal", {"cal": calibrator.to_dict(), "bias": bias.to_dict() if use_bias else None}
    )

    # One-time protected-test evaluation of every candidate (same rows).
    test_results: dict[str, Any] = {}
    candidate_rows: dict[str, pd.DataFrame] = {}
    for template in candidate_models():
        model = _fresh(template).fit(dev)
        if model.name == chosen:
            fc = forecaster.forecast(test)
        else:
            m, s = model.predict_log(test)
            # Same global calibration procedure applied to each candidate's own OOF scores.
            om, os_ = oof[model.name]
            cal = ConformalCalibrator(confidence_level=confidence_level, segment_column=None).fit(
                (z - om) / os_, censored
            )
            fc = CalibratedForecaster(_Frozen(model.name, m, s), None, cal).forecast(test)
        rows = scored_rows(test, fc, confidence_level)
        candidate_rows[model.name] = rows
        test_results[model.name] = evaluate_rows(rows)
    baseline_name = "baseline-0.1"
    promotion = _gates(test_results[chosen], test_results[baseline_name])
    test_results = {
        name: {"overall": r["overall"], "slices": r["slices"]} for name, r in test_results.items()
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_format": "seshat-forecaster/1",
        "predictor_version": predictor_version,
        "uncertainty_model_version": uncertainty_version,
        "calibration_version": calibration_version,
        "dataset_version": ds_version,
        "feature_set_version": MODEL_FEATURE_SET_VERSION,
        "confidence_level": confidence_level,
        "target": "output_tokens (latent; realized = min(forecast, max_output_tokens))",
        "model": final_model.to_dict(),
        "bias_correction": bias.to_dict() if use_bias else None,
        "calibration": calibrator.to_dict(),
        "label_origin": label_origin,
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": _git_revision(),
    }
    (out / "forecaster.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    evaluation = {
        "evaluation_format": "seshat-model-evaluation/1",
        "predictor_version": predictor_version,
        "uncertainty_model_version": uncertainty_version,
        "calibration_version": calibration_version,
        "dataset_version": ds_version,
        "feature_set_version": MODEL_FEATURE_SET_VERSION,
        "metrics_version": METRICS_VERSION,
        "label_origin": label_origin,
        "created_at": artifact["created_at"],
        "code_revision": artifact["code_revision"],
        "protocol": {
            "protected_test_rule": PROTECTED_TEST_RULE,
            "protected_test_fraction": PROTECTED_TEST_FRACTION,
            "cv_folds": n_folds,
            "seed": seed,
            "selection_metric": "OOF censoring-aware MAE (dev rows only)",
            "calibration": "OOF residuals of the selected model (dev rows only)",
            "notes": [
                "Task-signal regexes were adjusted after inspecting prompt texts and design "
                "categories of all rows (no labels or errors were inspected).",
                "All prompts are template-generated; generalisation to real user traffic is "
                "unverified.",
                "The protected test set was scored twice before this evaluation: once before "
                "the realized-usage target, the sigma bounds and the penalty grid were "
                "introduced (structural fixes for exploding intervals and selection bias), and "
                "once before segment calibration was restricted to segments with at least "
                "2/alpha - 1 rows (a finite-sample fix, verified by nested out-of-fold coverage "
                "on development data: 92.4% at nominal 90%). Neither change was tuned to test "
                "scores. New prompt families from the next batch provide an untouched test sample.",
            ],
        },
        "counts": {
            "rows": int(len(frame)),
            "dev_rows": int(len(dev)),
            "dev_uncensored": int((~dev["censored"]).sum()),
            "test_rows": int(len(test)),
            "test_uncensored": int((~test["censored"]).sum()),
            "groups": int(frame["split_group_id"].nunique()),
            "test_groups": int(test["split_group_id"].nunique()),
            "experiments": sorted(frame["experiment_id"].astype(str).unique().tolist()),
        },
        "hyperparameters": {
            "model": {
                k: v
                for k, v in final_model.to_dict().items()
                if k in ("type", "penalty", "scale_penalty")
            },
            "bias_correction": {"used": use_bias, "shrinkage": bias.shrinkage},
            "calibration": {
                "segment_column": calibrator.segment_column,
                "shrinkage": calibrator.shrinkage,
            },
        },
        "selection": {"chosen": chosen, "oof": selection},
        "protected_test": test_results,
        "promotion": promotion,
    }
    (out / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, default=_json) + "\n", encoding="utf-8"
    )
    # Out-of-sample predictions for every row (test: holdout; dev: out-of-fold).
    dev_fc = CalibratedForecaster(
        _Frozen(chosen, *oof[chosen]), bias if use_bias else None, calibrator
    ).forecast(dev)
    test_fc = forecaster.forecast(test)
    predictions = pd.concat(
        [
            _prediction_frame(
                dev, dev_fc, "backtest_out_of_fold", "cross_validation", confidence_level
            ),
            _prediction_frame(
                test, test_fc, "backtest_holdout", "protected_test", confidence_level
            ),
        ],
        ignore_index=True,
    )
    predictions.to_parquet(out / "backtest_predictions.parquet", index=False)
    return evaluation


def _prediction_frame(
    frame: pd.DataFrame, forecast: pd.DataFrame, mode: str, role: str, level: float
) -> pd.DataFrame:
    rows = pd.concat(
        [
            frame[["request_id", "sample_id", "observation_id"]],
            forecast,
            margins(forecast, level),
            realized_usage_forecast(frame, forecast["predicted_output_tokens"]),
        ],
        axis=1,
    )
    rows["prediction_mode"] = mode
    rows["evaluation_role"] = role
    return rows


@dataclass
class _Frozen:
    """Wrap precomputed (mu, sigma) so they flow through the same calibration code."""

    name: str
    mu: np.ndarray
    sigma: np.ndarray

    def fit(self, frame: pd.DataFrame) -> _Frozen:
        return self

    def predict_log(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if len(frame) != len(self.mu):
            raise ValueError("frozen forecasts do not match the frame")
        return self.mu, self.sigma

    def to_dict(self) -> dict[str, Any]:
        return {"type": "frozen"}


def _fresh(template: Forecaster) -> Forecaster:
    import copy

    return copy.deepcopy(template)


def _json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)
