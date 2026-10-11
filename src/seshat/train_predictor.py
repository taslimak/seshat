"""Train and evaluate a simple real-label predictor without split leakage.

The model uses only fields available to the public prediction API: prompt
word count, model ID, and requested reasoning effort. No provider responses
or observed usage enter feature generation at inference time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["prompt_word_count", "requested_model", "requested_effort"]
MIN_TRAIN = 100
MIN_CALIBRATION = 20
MIN_TEST = 20


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FEATURES].copy()
    result["prompt_word_count"] = pd.to_numeric(
        result["prompt_word_count"], errors="coerce"
    ).fillna(0).clip(lower=0)
    for name in ("requested_model", "requested_effort"):
        result[name] = result[name].fillna("auto" if name == "requested_effort" else "unknown")
        result[name] = result[name].astype(str)
    return result


def _load_split(directory: Path, split: str) -> pd.DataFrame:
    path = directory / f"{split}.parquet"
    if not path.exists():
        raise ValueError(f"Missing leakage-safe split: {path}")
    return pd.read_parquet(path)


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    # Do not train on truncated outputs or synthetic, partial, unavailable labels.
    frame = frame[
        (frame["label_origin"] == "real")
        & (frame["source_type"] == "controlled_experiment")
        & (frame["label_quality"] == "verified")
        & (~frame["target_is_censored"].fillna(True).astype(bool))
    ].copy()
    frame["actual_output_tokens"] = pd.to_numeric(
        frame["actual_output_tokens"], errors="coerce"
    )
    return frame.loc[frame["actual_output_tokens"].notna()].copy()


def _predict(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    log_values: np.ndarray = model.predict(_features(frame))
    return np.maximum(0, np.expm1(np.clip(log_values, 0, 16)))


def train_from_dataset(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source = Path(dataset_dir)
    target = Path(output_dir)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("label_origin") != "real":
        raise ValueError("Refusing synthetic training dataset")
    if set(manifest.get("source_types", [])) != {"controlled_experiment"}:
        raise ValueError("Only controlled real API experiments supported for production models")
    frames = {name: _eligible(_load_split(source, name)) for name in (
        "train", "validation", "calibration", "test"
    )}
    if any(len(frames[name]) < threshold for name, threshold in (
        ("train", MIN_TRAIN), ("calibration", MIN_CALIBRATION), ("test", MIN_TEST)
    )):
        raise ValueError(
            "Insufficient verified, uncensored real observations: "
            + ", ".join(f"{k}={len(v)}" for k, v in frames.items())
            + f"; require train>={MIN_TRAIN}, calibration>={MIN_CALIBRATION}, test>={MIN_TEST}"
        )
    groups = [set(v["split_group_id"].dropna().astype(str)) for v in frames.values()]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Leaked prompt groups across splits")
    transform = ColumnTransformer([
        ("numeric", "passthrough", ["prompt_word_count"]),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["requested_model", "requested_effort"],
        ),
    ])
    model = Pipeline([("features", transform), ("regression", Ridge(alpha=10.0))])
    model.fit(
        _features(frames["train"]),
        np.log1p(frames["train"]["actual_output_tokens"].to_numpy(dtype=float)),
    )
    calibrate = frames["calibration"]
    calibration_errors = np.abs(
        calibrate["actual_output_tokens"].to_numpy(dtype=float) - _predict(model, calibrate)
    )
    # Conservative split-conformal 90% absolute-error radius.
    rank = int(np.ceil((len(calibration_errors) + 1) * 0.9))
    radius = float(np.sort(calibration_errors)[min(rank, len(calibration_errors)) - 1])
    test = frames["test"]
    actual = test["actual_output_tokens"].to_numpy(dtype=float)
    predicted = _predict(model, test)
    lower = np.maximum(0, predicted - radius)
    upper = predicted + radius
    metrics = {
        "train_rows": len(frames["train"]),
        "validation_rows": len(frames["validation"]),
        "calibration_rows": len(calibrate),
        "test_rows": len(test),
        "test_mae": float(mean_absolute_error(actual, predicted)),
        "test_rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "test_interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "nominal_interval_coverage": 0.9,
        "interval_radius_tokens": radius,
        "target": "actual_output_tokens",
        "input_features": FEATURES,
        "dataset_version": manifest["dataset_version"],
    }
    if target.exists():
        raise FileExistsError(f"Model directory already exists: {target}")
    target.mkdir(parents=True)
    joblib.dump(
        {"model": model, "radius": radius, "version": f"ridge-{manifest['dataset_version']}"},
        target / "predictor.joblib",
    )
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(train_from_dataset(args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
