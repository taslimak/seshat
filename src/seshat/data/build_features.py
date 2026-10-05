"""Build leakage-safe request-time feature matrices and targets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from seshat.data.feature_registry import TRAINING_FEATURES, assert_training_features
from seshat.data.io import read_parquet, write_parquet

IDENTIFIER_COLUMNS = (
    "observation_id",
    "run_id",
    "condition_id",
    "prompt_id",
    "prompt_family_id",
    "experiment_id",
)
TARGET_COLUMN = "output_tokens"


def build_feature_frame(normalized: pd.DataFrame) -> pd.DataFrame:
    if "quality_decision" in normalized.columns:
        eligible = normalized[normalized["quality_decision"].isin(["PASS", "WARNING"])].copy()
    else:
        eligible = normalized.copy()
    feature_columns = [column for column in TRAINING_FEATURES if column in eligible.columns]
    assert_training_features(feature_columns)
    required = [column for column in IDENTIFIER_COLUMNS if column in eligible.columns]
    if TARGET_COLUMN not in eligible.columns:
        raise ValueError(f"Missing target column {TARGET_COLUMN}")
    output = eligible[required + feature_columns + [TARGET_COLUMN]].copy()
    output["target_is_censored"] = eligible.get("is_truncated", False)
    output["is_refusal"] = eligible.get("is_refusal", False)
    output["quality_decision"] = eligible.get("quality_decision", "PASS")
    return output


def build_features_to_parquet(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    frame = build_feature_frame(read_parquet(input_path))
    write_parquet(output_path, frame)
    return frame
