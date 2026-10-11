"""``prediction_outcomes``: every prediction joined to its observed outcome.

One row per (prediction, eligible observation of the same request), written to
``<root>/quality/prediction_outcomes/predictor=<version>/outcomes.parquet``.
The table is a deterministic function of the canonical ``predictions``,
``observations``, ``requests``, ``preflight`` and validation tables, so a rebuild
is idempotent (re-running a workflow never duplicates outcomes) and the raw
predicted and actual values remain the canonical evidence: every metric column
can be recomputed from them under a future :data:`METRICS_VERSION`.

``label_origin`` is ``real`` or ``synthetic`` and is never mixed downstream:
snapshots, drift and run reports are computed per origin, and only ``real``
outcomes may feed production calibration or reporting.

``valid_out_of_sample`` marks rows whose forecast did not see the observation:
preflight forecasts made before dispatch, backtests (protected test or
out-of-fold), and untrained heuristics. In-sample re-scoring is kept but must not
be reported as accuracy.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seshat.data.io import read_parquet
from seshat.data.storage import StorageLayout
from seshat.quality.metrics import METRICS_VERSION, add_row_metrics

OUTCOMES_DIR = Path("quality") / "prediction_outcomes"
OUT_OF_SAMPLE_MODES = {"preflight", "backtest_holdout", "backtest_out_of_fold"}
UNTRAINED_PREFIXES = ("baseline-",)
LENGTH_BANDS = [0, 50, 150, 400, 1000, 10**9]
LENGTH_LABELS = ["<=50w", "51-150w", "151-400w", "401-1000w", ">1000w"]
TOKEN_BANDS = [-1, 64, 256, 1024, 4096, 10**12]
TOKEN_LABELS = ["<=64", "65-256", "257-1024", "1025-4096", ">4096"]
CONTEXT_BANDS = [-1, 0, 2000, 10000, 10**12]
CONTEXT_LABELS = ["none", "<=2k chars", "2k-10k chars", ">10k chars"]

LINEAGE = [
    "prediction_id",
    "observation_id",
    "sample_id",
    "request_id",
    "run_id",
    "experiment_id",
    "timestamp_prediction_created",
    "timestamp_actual_observed",
    "predictor_version",
    "uncertainty_model_version",
    "calibration_version",
    "dataset_version",
    "feature_set_version",
    "pricing_version",
    "code_revision",
    "label_origin",
    "source_type",
    "prediction_mode",
    "evaluation_role",
    "valid_out_of_sample",
    "metrics_version",
]


def _band(values: pd.Series, bins: list[int], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins, labels=labels).astype(str)


def build_outcomes(root: str | Path) -> pd.DataFrame:
    """All prediction outcomes (every predictor version), with metrics."""
    layout = StorageLayout(root)
    predictions = layout.read_frame("predictions")
    observations = layout.read_frame("observations")
    decisions_path = layout.validation / "decisions.parquet"
    if predictions.empty or observations.empty or not decisions_path.exists():
        return pd.DataFrame()
    decisions = read_parquet(decisions_path)[
        ["observation_id", "eligible_for_training", "data_origin"]
    ]
    obs = observations.merge(decisions, on="observation_id", how="inner")
    obs = obs[obs["eligible_for_training"] & obs["actual_output_tokens"].notna()]
    obs = obs.rename(
        columns={
            "data_origin": "label_origin",
            "collected_at_utc": "timestamp_actual_observed",
            "calculated_cost_usd": "calculated_actual_cost_usd",
        }
    )
    obs_cols = [
        "observation_id",
        "sample_id",
        "request_id",
        "run_id",
        "provider",
        "requested_model",
        "resolved_model",
        "source_type",
        "label_origin",
        "label_quality",
        "timestamp_actual_observed",
        "actual_input_tokens",
        "actual_output_tokens",
        "actual_reasoning_tokens",
        "actual_visible_output_tokens",
        "actual_total_tokens",
        "actual_cached_input_tokens",
        "calculated_actual_cost_usd",
        "pricing_version",
        "is_truncated",
        "stop_reason",
        "response_status",
    ]
    frame = predictions.rename(columns={"created_at_utc": "timestamp_prediction_created"}).merge(
        obs[obs_cols], on="request_id", how="inner"
    )
    requests = layout.read_frame("requests").drop_duplicates("sample_id")
    request_cols = [
        "sample_id",
        "experiment_id",
        "requested_effort",
        "prompt_category",
        "prompt_subcategory",
        "prompt_word_count",
        "prompt_char_count",
        "message_count",
        "requested_output_format",
        "max_output_tokens",
        "tool_count",
        "context_character_count",
    ]
    frame = frame.merge(requests[request_cols], on="sample_id", how="left")
    preflight = layout.read_frame("preflight")
    if not preflight.empty:
        pre = preflight[preflight["preflight_input_tokens"].notna()].copy()
        pre["_rank"] = (pre["preflight_method"] != "provider_native").astype(int)
        pre = pre.sort_values(["sample_id", "_rank"]).drop_duplicates("sample_id")
        frame = frame.merge(
            pre[["sample_id", "preflight_input_tokens", "preflight_method"]],
            on="sample_id",
            how="left",
        )
    frame["task_group"] = _task_groups(root, frame["sample_id"])
    frame["target_is_censored"] = frame["is_truncated"].fillna(False).astype(bool) | (
        frame["label_quality"] == "censored"
    )
    frame["prediction_interval_lower"] = frame["lower_bound"].astype(float)
    frame["prediction_interval_upper"] = frame["upper_bound"].astype(float)
    frame["prediction_mode"] = frame["prediction_mode"].fillna("retrospective")
    untrained = frame["predictor_version"].astype(str).str.startswith(UNTRAINED_PREFIXES)
    frame["valid_out_of_sample"] = frame["prediction_mode"].isin(OUT_OF_SAMPLE_MODES) | untrained
    frame["metrics_version"] = METRICS_VERSION
    frame["provider_model"] = frame["provider"] + "/" + frame["requested_model"]
    frame["effort"] = frame["requested_effort"].fillna("unset")
    frame["prompt_length_band"] = _band(frame["prompt_word_count"], LENGTH_BANDS, LENGTH_LABELS)
    frame["predicted_output_band"] = _band(
        frame["predicted_output_tokens"], TOKEN_BANDS, TOKEN_LABELS
    )
    frame["actual_output_band"] = _band(frame["actual_output_tokens"], TOKEN_BANDS, TOKEN_LABELS)
    frame["context_band"] = _band(frame["context_character_count"], CONTEXT_BANDS, CONTEXT_LABELS)
    frame["tool_use"] = np.where(frame["tool_count"].fillna(0) > 0, "tools", "no_tools")
    for column in ("predicted_cost_usd", "calculated_actual_cost_usd"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["reasoning_usage_reported"] = frame["actual_reasoning_tokens"].notna()
    return add_row_metrics(frame).sort_values(
        ["predictor_version", "timestamp_actual_observed", "prediction_id"]
    )


def _task_groups(root: str | Path, sample_ids: pd.Series) -> list[str]:
    """Inference-time task group from the stored exact request (for segment statistics)."""
    from seshat.data.schemas.request import LLMRequest
    from seshat.modeling.dataset import stored_requests
    from seshat.modeling.features import model_features

    plans = stored_requests(root)
    cache: dict[str, str] = {}
    out = []
    for sid in sample_ids.astype(str):
        if sid not in cache:
            plan = plans.get(sid)
            cache[sid] = (
                str(model_features(LLMRequest.model_validate(plan["request"]))["task_group"])
                if plan
                else "unknown"
            )
        out.append(cache[sid])
    return out


def write_outcomes(root: str | Path, outcomes: pd.DataFrame) -> dict[str, int]:
    """Atomically replace each predictor version's outcome file. Returns rows per version."""
    base = Path(root) / OUTCOMES_DIR
    written: dict[str, int] = {}
    keep = set()
    for version, part in outcomes.groupby("predictor_version"):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(version))
        path = base / f"predictor={safe}" / "outcomes.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(".outcomes.parquet.tmp")
        _parquet_safe(part).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        written[str(version)] = int(len(part))
        keep.add(path)
    for stale in base.glob("predictor=*/outcomes.parquet"):
        if stale not in keep:
            stale.unlink()
    return written


def read_outcomes(root: str | Path) -> pd.DataFrame:
    parts = sorted((Path(root) / OUTCOMES_DIR).glob("predictor=*/outcomes.parquet"))
    return (
        pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        if parts
        else pd.DataFrame()
    )


def _parquet_safe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            sample: Any = out[column].dropna().head(1).tolist()
            if sample and not isinstance(sample[0], (str, bool)):
                out[column] = out[column].map(lambda v: None if v is None else str(v))
    return out
