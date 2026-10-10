"""Score stored requests with a predictor and compare predictions with observations.

Predictions are written to their own table (``normalized/predictions``) and are never
read back as observations or labels. Comparisons are always reported per data origin,
so test-fixture "accuracy" is never pooled with (or mistaken for) real accuracy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd

from seshat.data.io import read_parquet
from seshat.data.schemas.records import PredictionRecord
from seshat.data.storage import StorageLayout
from seshat.predictor import Prediction


class WordCountPredictor(Protocol):
    version: str

    def predict_from_word_count(
        self, word_count: int, model: str, effort: str | None
    ) -> Prediction: ...


def score_requests(root: str, predictor: WordCountPredictor, *, now: datetime | None = None) -> int:
    """(Re)build the predictions of ``predictor.version`` for every stored request."""
    layout = StorageLayout(root)
    requests = layout.read_frame("requests")
    created = now or datetime.now(UTC)
    existing = layout.read_frame("predictions")
    previous = (
        dict(zip(existing["prediction_id"], existing["created_at_utc"], strict=True))
        if not existing.empty
        else {}
    )
    records: list[PredictionRecord] = []
    for row in requests.drop_duplicates("request_id").itertuples(index=False):
        result = predictor.predict_from_word_count(
            int(row.prompt_word_count), row.requested_model, row.requested_effort
        )
        prediction_id = f"pred_{row.request_id}_{predictor.version}"
        records.append(
            PredictionRecord(
                prediction_id=prediction_id,
                request_id=row.request_id,
                created_at_utc=previous.get(prediction_id, created),
                predictor_version=result.predictor_version,
                predicted_output_tokens=result.predicted_output_tokens,
                lower_bound=result.lower_bound,
                upper_bound=result.upper_bound,
                confidence=result.confidence,
                interval_calibrated=False,
            )
        )
    layout.replace_source("predictions", f"predictor-{predictor.version}", records)
    return len(records)


def compare_predictions(root: str) -> pd.DataFrame:
    """Eligible observations joined with predictions for the same request."""
    layout = StorageLayout(root)
    predictions = layout.read_frame("predictions")
    observations = layout.read_frame("observations")
    decisions_path = layout.validation / "decisions.parquet"
    if predictions.empty or observations.empty or not decisions_path.exists():
        raise ValueError(
            "Need predictions, observations and validation decisions (run predict and validate)"
        )
    decisions = read_parquet(decisions_path)[
        ["observation_id", "eligible_for_training", "data_origin"]
    ]
    obs = observations.merge(decisions, on="observation_id")
    obs = obs[obs["eligible_for_training"]]
    frame = predictions.merge(
        obs[
            [
                "request_id",
                "observation_id",
                "provider",
                "requested_model",
                "actual_output_tokens",
                "is_truncated",
                "data_origin",
            ]
        ],
        on="request_id",
    )
    frame["error"] = frame["predicted_output_tokens"] - frame["actual_output_tokens"]
    frame["abs_error"] = frame["error"].abs()
    frame["within_interval"] = frame["actual_output_tokens"].between(
        frame["lower_bound"], frame["upper_bound"]
    )
    return pd.DataFrame(frame)


def summarize(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per (predictor version, data origin): MAE, median APE, interval coverage.

    Truncated observations are right-censored and excluded from error metrics.
    """
    rows: list[dict[str, Any]] = []
    for (version, origin), group in frame.groupby(["predictor_version", "data_origin"]):
        scored = group[~group["is_truncated"].fillna(False).astype(bool)]
        actual = scored["actual_output_tokens"].astype(float)
        ape = (scored["abs_error"] / actual.where(actual > 0)).dropna()
        rows.append(
            {
                "predictor_version": version,
                "data_origin": origin,
                "pairs": int(len(group)),
                "censored_excluded": int(len(group) - len(scored)),
                "mae_tokens": round(float(scored["abs_error"].mean()), 2) if len(scored) else None,
                "median_abs_pct_error": round(float(ape.median()), 4) if len(ape) else None,
                "interval_coverage": round(float(scored["within_interval"].mean()), 4)
                if len(scored)
                else None,
                "note": "test-fixture outcomes: validates plumbing only, not real accuracy"
                if origin == "synthetic"
                else None,
            }
        )
    return rows
