"""Build leakage-safe request-time feature matrices and targets.

Label-origin policy: a feature matrix contains outcomes of exactly one origin.
The default is ``real``. Synthetic outcomes (mock adapter, simulations) are only
used when explicitly requested with ``label_origin="synthetic"``, for pipeline tests,
and are never mixed with real labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from seshat.data.feature_registry import TRAINING_FEATURES, assert_training_features
from seshat.data.io import read_parquet, write_parquet
from seshat.data.provenance import DataOrigin
from seshat.data.records import RecordType
from seshat.data.store import DatasetStore

IDENTIFIER_COLUMNS = (
    "observation_id",
    "run_id",
    "request_id",
    "condition_id",
    "prompt_id",
    "prompt_family_id",
    "experiment_id",
)
TARGET_COLUMN = "output_tokens"
LabelOrigin = Literal["real", "synthetic"]
UNCATEGORIZED = "uncategorized"


def _select_label_origin(frame: pd.DataFrame, label_origin: LabelOrigin) -> pd.DataFrame:
    origin = DataOrigin(label_origin)
    if "data_origin" not in frame.columns:
        raise ValueError(
            "Input has no data_origin column, so real and synthetic outcomes cannot be "
            "told apart. Re-run `seshat-data normalize` (or ingest with provenance)."
        )
    selected = frame[frame["data_origin"] == origin.value]
    if selected.empty:
        available = sorted(frame["data_origin"].dropna().astype(str).unique())
        hint = (
            " To build a pipeline-test dataset from synthetic outcomes, pass "
            "--label-origin synthetic."
            if origin == DataOrigin.REAL and DataOrigin.SYNTHETIC.value in available
            else ""
        )
        raise ValueError(
            f"No rows with data_origin={origin.value!r} (available: {available}).{hint}"
        )
    return selected


def build_feature_frame(
    normalized: pd.DataFrame,
    *,
    label_origin: LabelOrigin = "real",
) -> pd.DataFrame:
    """Feature matrix + target for PASS/WARNING rows of a single label origin."""
    if "quality_decision" in normalized.columns:
        eligible = normalized[normalized["quality_decision"].isin(["PASS", "WARNING"])].copy()
    else:
        eligible = normalized.copy()
    eligible = _select_label_origin(eligible, label_origin)
    feature_columns = [column for column in TRAINING_FEATURES if column in eligible.columns]
    assert_training_features(feature_columns)
    required = [column for column in IDENTIFIER_COLUMNS if column in eligible.columns]
    if TARGET_COLUMN not in eligible.columns:
        raise ValueError(f"Missing target column {TARGET_COLUMN}")
    output = eligible[required + feature_columns + [TARGET_COLUMN]].copy()
    output["target_is_censored"] = eligible.get("is_truncated", False)
    output["is_refusal"] = eligible.get("is_refusal", False)
    output["quality_decision"] = eligible.get("quality_decision", "PASS")
    output["label_origin"] = label_origin
    return output.reset_index(drop=True)


def store_usage_frame(store: DatasetStore) -> pd.DataFrame:
    """Join stored request features with their observed usage, one row per observation.

    Request-side columns supply the features; the observation supplies the target,
    quality decision and ``data_origin`` (the label origin is the origin of the
    *outcome*). Requests without a prompt family use their prompt hash, so identical
    prompts always share a split; missing categories become ``uncategorized``.
    """
    requests = store.read(RecordType.REQUEST_FEATURES)
    observations = store.read(RecordType.OBSERVED_USAGE)
    if requests.empty or observations.empty:
        raise ValueError(f"Store at {store.root} has no request features or observed usage")
    request_columns = [
        "request_id",
        "provider",
        "model",
        "effort",
        "max_output_tokens",
        "temperature",
        "category",
        "prompt_hash",
        "prompt_family_id",
        "experiment_id",
        *[column for column in TRAINING_FEATURES if column in requests.columns],
    ]
    request_side = requests[list(dict.fromkeys(request_columns))].rename(
        columns={"model": "model_requested"}
    )
    observation_columns = [
        "observation_id",
        "request_id",
        "status",
        "output_tokens",
        "input_tokens",
        "reasoning_tokens",
        "total_tokens",
        "is_truncated",
        "is_refusal",
        "quality_decision",
        "data_origin",
    ]
    frame = request_side.merge(
        observations[observation_columns], on="request_id", how="inner", validate="one_to_many"
    )
    family = frame["prompt_family_id"].astype("object")
    fallback = "pf_" + frame["prompt_hash"].astype(str).str[:16]
    frame["prompt_family_id"] = family.where(family.notna(), fallback)
    frame["category"] = frame["category"].astype("object").fillna(UNCATEGORIZED)
    return frame.drop(columns=["prompt_hash"])


def build_features_to_parquet(
    input_path: str | Path | None,
    output_path: str | Path,
    *,
    label_origin: LabelOrigin = "real",
    store_root: str | Path | None = None,
) -> pd.DataFrame:
    """Build features from a validated normalized table *or* from a store."""
    if (input_path is None) == (store_root is None):
        raise ValueError("Provide exactly one of input_path or store_root")
    source = (
        store_usage_frame(DatasetStore(store_root))
        if store_root is not None
        else read_parquet(input_path)  # type: ignore[arg-type]
    )
    frame = build_feature_frame(source, label_origin=label_origin)
    write_parquet(output_path, frame)
    return frame
