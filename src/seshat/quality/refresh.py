"""Every-run quality loop (stage E) and model registration.

``refresh_quality(root)`` is idempotent and restart-safe:

1. re-scores stored requests with the heuristic baseline (deterministic ids);
2. rebuilds ``prediction_outcomes`` from canonical tables (no duplicates on rerun);
3. writes rolling/sliced quality snapshots, per-predictor drift reports and a
   quality report per (experiment, predictor) for real outcomes;
4. writes a summary under ``quality/summary.json``.

``register_model(root, model_dir)`` stores a trained forecaster's artifact and
immutable evaluation record under ``models/<predictor_version>/``, appends the
evaluation to ``quality/model_evaluations.jsonl`` (once per version) and writes its
out-of-sample backtest forecasts into the canonical ``predictions`` table, so they
flow into outcomes like any other prediction. It never activates a model.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.io import write_json
from seshat.data.schemas.records import PredictionRecord
from seshat.data.storage import StorageLayout
from seshat.quality.outcomes import build_outcomes, write_outcomes
from seshat.quality.snapshots import drift_report, quality_snapshots, render_run_report

QUALITY = Path("quality")
MODELS = Path("models")
RETRAIN_MIN_NEW_REAL_OBSERVATIONS = 50


def _int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else max(int(round(float(value))), 0)


def _float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(str(value))


def _str(value: Any) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def register_model(root: str | Path, model_dir: str | Path) -> dict[str, Any]:
    source = Path(model_dir)
    artifact = json.loads((source / "forecaster.json").read_text(encoding="utf-8"))
    evaluation = json.loads((source / "evaluation.json").read_text(encoding="utf-8"))
    version = artifact["predictor_version"]
    target = Path(root) / MODELS / version
    target.mkdir(parents=True, exist_ok=True)
    for name in ("forecaster.json", "evaluation.json"):
        existing = target / name
        if existing.exists() and existing.read_bytes() != (source / name).read_bytes():
            # Same content-addressed version, different calibration or evaluation: keep both.
            name_new = f"{name[:-5]}.{artifact['calibration_version']}.json"
            shutil.copyfile(source / name, target / name_new)
        else:
            shutil.copyfile(source / name, existing)
    ledger = Path(root) / QUALITY / "model_evaluations.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    key = (version, artifact["calibration_version"], evaluation["dataset_version"])
    seen = set()
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                seen.add((e["predictor_version"], e["calibration_version"], e["dataset_version"]))
    if key not in seen:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evaluation, sort_keys=True) + "\n")
    backtest = pd.read_parquet(source / "backtest_predictions.parquet")
    created = datetime.fromisoformat(artifact["created_at"])
    records = []
    for row in backtest.itertuples(index=False):
        point = _int(row.predicted_output_tokens) or 0
        lower = min(_int(row.prediction_interval_lower) or 0, point)
        upper = max(_int(row.prediction_interval_upper) or point, point)
        records.append(
            PredictionRecord(
                prediction_id=f"pred_{row.request_id}_{version}_{artifact['calibration_version']}",
                request_id=str(row.request_id),
                created_at_utc=created,
                predictor_version=version,
                predicted_output_tokens=point,
                lower_bound=lower,
                upper_bound=upper,
                confidence=float(artifact["confidence_level"]),
                interval_calibrated=True,
                prediction_mode=_str(row.prediction_mode),
                evaluation_role=_str(row.evaluation_role),
                base_predicted_output_tokens=_int(row.base_predicted_output_tokens),
                confidence_level=float(artifact["confidence_level"]),
                lower_margin_tokens=point - lower,
                upper_margin_tokens=upper - point,
                confidence_margin_tokens=max(point - lower, upper - point),
                lower_margin_pct=_float(row.lower_margin_pct),
                upper_margin_pct=_float(row.upper_margin_pct),
                confidence_margin_pct=_float(row.confidence_margin_pct),
                predicted_total_tokens=_int(row.predicted_total_tokens),
                predicted_cost_usd=None
                if _float(row.predicted_cost_usd) is None
                else Decimal(str(round(float(str(row.predicted_cost_usd)), 12))),
                uncertainty_model_version=artifact["uncertainty_model_version"],
                calibration_version=artifact["calibration_version"],
                dataset_version=artifact["dataset_version"],
                feature_set_version=artifact["feature_set_version"],
                code_revision=artifact.get("code_revision"),
            )
        )
    StorageLayout(root).replace_source(
        "predictions", f"predictor-{version}-{artifact['calibration_version']}", records
    )
    return {
        "predictor_version": version,
        "calibration_version": artifact["calibration_version"],
        "predictions": len(records),
        "promotion": evaluation["promotion"]["decision"],
    }


def refresh_quality(root: str | Path, *, as_of: datetime | None = None) -> dict[str, Any]:
    from seshat.data.predictions import score_requests
    from seshat.predictor import BaselinePredictor

    as_of = as_of or datetime.now(UTC)
    base = Path(root)
    score_requests(str(base), BaselinePredictor())
    outcomes = build_outcomes(base)
    summary: dict[str, Any] = {
        "as_of": as_of.isoformat(),
        "outcomes": {},
        "outcomes_by_origin": {},
    }
    if outcomes.empty:
        write_json(base / QUALITY / "summary.json", summary)
        return summary
    summary["outcomes"] = write_outcomes(base, outcomes)
    counts = outcomes.groupby("label_origin").size().to_dict()
    summary["outcomes_by_origin"] = {str(k): int(v) for k, v in counts.items()}
    summary["censored_outcomes"] = int(outcomes["target_is_censored"].sum())
    snapshots = quality_snapshots(outcomes, as_of=as_of)
    path = base / QUALITY / "model_quality_snapshots.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(path, index=False)
    summary["snapshot_rows"] = int(len(snapshots))
    drift: dict[str, Any] = {}
    reports = base / QUALITY / "run_reports"
    reports.mkdir(parents=True, exist_ok=True)
    real = outcomes[(outcomes["label_origin"] == "real") & outcomes["valid_out_of_sample"]]
    for version in sorted(outcomes["predictor_version"].unique()):
        drift[version] = drift_report(outcomes, version)
        write_json(base / QUALITY / "drift" / f"{_safe(version)}.json", drift[version])
        for experiment, part in real[real["predictor_version"] == version].groupby("experiment_id"):
            (reports / f"{_safe(str(experiment))}__{_safe(version)}.md").write_text(
                render_run_report(part, f"Run quality: {experiment} · {version}"), encoding="utf-8"
            )
    summary["drift"] = {v: d["status"] for v, d in drift.items()}
    summary["retraining"] = retraining_due(base, outcomes, drift)
    write_json(base / QUALITY / "summary.json", summary)
    return summary


def retraining_due(
    root: Path, outcomes: pd.DataFrame, drift: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Train a challenger only on enough new evidence: new real data or detected drift."""
    real_observations = int(
        outcomes.loc[outcomes["label_origin"] == "real", "observation_id"].nunique()
    )
    ledger = root / QUALITY / "model_evaluations.jsonl"
    last_rows = 0
    last_version = None
    if ledger.exists():
        lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            last = json.loads(lines[-1])
            last_rows = int(last["counts"]["rows"])
            last_version = last["predictor_version"]
    new_rows = real_observations - last_rows
    # Drift justifies a challenger only when there is new data to learn from; otherwise
    # retraining would refit the same rows on every sync.
    drifted = bool(
        last_version and new_rows > 0 and drift.get(last_version, {}).get("status") == "drift"
    )
    reasons = []
    if last_version is None and real_observations >= RETRAIN_MIN_NEW_REAL_OBSERVATIONS:
        reasons.append("no forecaster evaluated yet")
    if last_version is not None and new_rows >= RETRAIN_MIN_NEW_REAL_OBSERVATIONS:
        reasons.append(f"{new_rows} new real observations since {last_version}")
    if drifted:
        reasons.append(f"drift detected for {last_version}")
    if last_version and not drifted and drift.get(last_version, {}).get("status") == "drift":
        reasons_blocked = f"drift detected for {last_version}, but no new data since evaluation"
    else:
        reasons_blocked = None
    return {
        "due": bool(reasons),
        "reasons": reasons,
        "deferred": reasons_blocked,
        "real_observations": real_observations,
        "new_since_last_evaluation": new_rows,
        "min_new_observations": RETRAIN_MIN_NEW_REAL_OBSERVATIONS,
        "note": "Retraining produces a challenger; promotion needs gates + owner approval.",
    }


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
