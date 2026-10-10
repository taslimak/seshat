"""Score stored requests with a predictor and compare predictions with observed usage."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd

from seshat.data.records import PredictionRecord, PromptFeatureValues, RecordType, stable_id
from seshat.data.store import DatasetStore
from seshat.features import PromptFeatures
from seshat.predictor import Prediction


class FeaturePredictor(Protocol):
    version: str

    def predict_features(self, features: PromptFeatures, model: str, effort: str) -> Prediction:
        ...


def prediction_records(
    store: DatasetStore,
    predictor: FeaturePredictor,
    *,
    now: datetime | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(location, PredictionRecord payload)`` for stored requests not yet scored.

    Prediction IDs are ``hash(request_id | predictor_version)``. Requests that already
    have a prediction from this predictor version are skipped, so re-running is a
    no-op rather than a conflicting re-prediction with a new timestamp.
    """
    created = now or datetime.now(UTC)
    requests = store.read(RecordType.REQUEST_FEATURES)
    if requests.empty:
        return
    already_scored = set(store.record_hashes(RecordType.PREDICTION))
    feature_names = list(PromptFeatureValues.model_fields)
    for row in requests.to_dict(orient="records"):
        prediction_id = stable_id("pred", f"{row['request_id']}|{predictor.version}")
        if prediction_id in already_scored:
            continue
        features = PromptFeatureValues.model_validate(
            {name: row[name] for name in feature_names}
        ).to_features()
        result = predictor.predict_features(features, str(row["model"]), str(row["effort"]))
        record = PredictionRecord(
            prediction_id=prediction_id,
            request_id=str(row["request_id"]),
            created_at=created,
            predictor_version=result.predictor_version,
            predicted_output_tokens=result.predicted_output_tokens,
            lower_bound=result.lower_bound,
            upper_bound=result.upper_bound,
            confidence=result.confidence,
            interval_calibrated=False,
        )
        yield f"request:{row['request_id']}", record.model_dump(mode="json")


def compare_predictions(store: DatasetStore) -> pd.DataFrame:
    """One row per (prediction, successful observation) sharing a ``request_id``.

    Truncated observations are right-censored: their true output length is unknown,
    so they are kept but excluded from error metrics in ``summarize_comparison``.
    """
    predictions = store.read(RecordType.PREDICTION)
    observations = store.read(RecordType.OBSERVED_USAGE)
    if predictions.empty or observations.empty:
        raise ValueError("Store needs both predictions and observed usage to compare")
    observed = observations[observations["status"] == "success"][
        [
            "observation_id",
            "request_id",
            "provider",
            "model_requested",
            "output_tokens",
            "is_truncated",
            "data_origin",
        ]
    ].rename(columns={"output_tokens": "observed_output_tokens"})
    frame = predictions[
        [
            "prediction_id",
            "request_id",
            "predictor_version",
            "predicted_output_tokens",
            "lower_bound",
            "upper_bound",
            "confidence",
            "interval_calibrated",
        ]
    ].merge(observed, on="request_id", how="inner")
    frame["error"] = frame["predicted_output_tokens"] - frame["observed_output_tokens"]
    frame["abs_error"] = frame["error"].abs()
    frame["within_interval"] = frame["observed_output_tokens"].between(
        frame["lower_bound"], frame["upper_bound"]
    )
    return frame.sort_values(["predictor_version", "data_origin", "request_id"]).reset_index(
        drop=True
    )


def summarize_comparison(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Metrics per predictor version and data origin (never pooled across origins)."""
    summary: list[dict[str, Any]] = []
    for (version, origin), group in frame.groupby(["predictor_version", "data_origin"]):
        scored = group[~group["is_truncated"].astype(bool)]
        observed = scored["observed_output_tokens"].astype(float)
        pct = (scored["abs_error"] / observed.where(observed > 0)).dropna()
        summary.append(
            {
                "predictor_version": version,
                "data_origin": origin,
                "pairs": int(len(group)),
                "censored_excluded": int(len(group) - len(scored)),
                "mae_tokens": float(scored["abs_error"].mean()) if len(scored) else None,
                "median_abs_pct_error": float(pct.median()) if len(pct) else None,
                "interval_coverage": (
                    float(scored["within_interval"].mean()) if len(scored) else None
                ),
                "note": (
                    "synthetic outcomes: validates plumbing only, not real accuracy"
                    if origin == "synthetic"
                    else None
                ),
            }
        )
    return summary
