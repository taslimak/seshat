"""Freeze an immutable, versioned dataset release with lineage and checksums."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.feature_registry import TRAINING_FEATURES, feature_registry
from seshat.data.io import read_parquet, write_json, write_parquet
from seshat.data.provenance import PIPELINE_VERSION, DataOrigin
from seshat.data.split import SPLIT_NAMES

SYNTHETIC_INTENDED_USE = (
    "Pipeline testing only. Labels are synthetic and must not be used to train or "
    "evaluate predictors of real provider usage."
)
REAL_INTENDED_USE = "Training and evaluation of output-token predictors."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(cwd: str | Path | None = None) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": sha, "dirty": dirty}


def _unique_strings(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def freeze_release(
    features_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    *,
    version: str,
    feature_pipeline_version: str = PIPELINE_VERSION,
) -> dict[str, object]:
    """Write train/validation/calibration/test Parquet plus manifest and schema.

    Releases are immutable: an existing release directory is never overwritten.
    Synthetic-label releases get a ``_synthetic`` suffix and an explicit
    pipeline-testing-only ``intended_use``.
    """
    features = read_parquet(features_path)
    assignments = read_parquet(split_path)
    origins = _unique_strings(features, "label_origin")
    if len(origins) != 1:
        raise ValueError(
            f"Features must have exactly one label_origin, found {origins or 'none'}. "
            "Rebuild them with `seshat-data features`."
        )
    label_origin = DataOrigin(origins[0])
    merged = features.merge(
        assignments[["prompt_family_id", "split"]],
        on="prompt_family_id",
        how="left",
        validate="many_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Some observations are missing split assignments")
    suffix = "_synthetic" if label_origin == DataOrigin.SYNTHETIC else ""
    target = Path(output_dir) / f"dataset_v{version}{suffix}"
    if target.exists():
        raise FileExistsError(
            f"Release {target} already exists; releases are immutable. Use a new version."
        )
    target.mkdir(parents=True)
    files: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split_name in SPLIT_NAMES:
        part = merged[merged["split"] == split_name].drop(columns=["split"])
        path = target / f"{split_name}.parquet"
        write_parquet(path, part)
        files[path.name] = _sha256(path)
        split_counts[split_name] = int(len(part))
    csv_path = target / "records.csv"
    merged.to_csv(csv_path, index=False)
    files[csv_path.name] = _sha256(csv_path)

    feature_columns = [column for column in TRAINING_FEATURES if column in features.columns]
    registry = feature_registry()
    schema = {
        "target": "output_tokens",
        "label_origin": label_origin.value,
        "feature_columns": feature_columns,
        "columns": {
            column: {
                "dtype": str(features[column].dtype),
                "role": registry[column].phase.value if column in registry else "METADATA",
            }
            for column in features.columns
        },
    }
    write_json(target / "schema.json", schema)
    files["schema.json"] = _sha256(target / "schema.json")

    manifest: dict[str, object] = {
        "dataset_version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "label_origin": label_origin.value,
        "intended_use": (
            SYNTHETIC_INTENDED_USE if label_origin == DataOrigin.SYNTHETIC else REAL_INTENDED_USE
        ),
        "feature_pipeline_version": feature_pipeline_version,
        "code_revision": git_revision(),
        "source_experiments": _unique_strings(features, "experiment_id"),
        "providers": _unique_strings(features, "provider"),
        "models": _unique_strings(features, "model_requested"),
        "observation_count": int(len(merged)),
        "prompt_family_count": int(merged["prompt_family_id"].nunique()),
        "split_counts": split_counts,
        "censored_target_count": int(merged.get("target_is_censored", pd.Series(dtype=bool)).sum()),
        "target": "output_tokens",
        "split_strategy": "prompt-family grouped, category-stratified",
        "input_files": {
            "features": _sha256(Path(features_path)),
            "splits": _sha256(Path(split_path)),
        },
        "files": files,
    }
    write_json(target / "manifest.json", manifest)
    return manifest
