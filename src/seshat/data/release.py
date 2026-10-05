"""Freeze a versioned dataset release with lineage and checksums."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from seshat.data.io import read_parquet, write_json, write_parquet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_release(
    features_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    *,
    version: str,
    feature_pipeline_version: str = "0.1.0",
) -> dict[str, object]:
    features = read_parquet(features_path)
    assignments = read_parquet(split_path)
    merged = features.merge(
        assignments[["prompt_family_id", "split"]],
        on="prompt_family_id",
        how="left",
        validate="many_to_one",
    )
    if merged["split"].isna().any():
        raise ValueError("Some observations are missing split assignments")
    target = Path(output_dir) / f"dataset_v{version}"
    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for split_name in ("train", "validation", "calibration", "test"):
        path = target / f"{split_name}.parquet"
        write_parquet(path, merged[merged["split"] == split_name].drop(columns=["split"]))
        files[path.name] = _sha256(path)
    csv_path = target / "records.csv"
    merged.to_csv(csv_path, index=False)
    files[csv_path.name] = _sha256(csv_path)
    manifest: dict[str, object] = {
        "dataset_version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "feature_pipeline_version": feature_pipeline_version,
        "observation_count": int(len(merged)),
        "prompt_family_count": int(merged["prompt_family_id"].nunique()),
        "target": "output_tokens",
        "split_strategy": "prompt-family grouped, category-stratified",
        "files": files,
    }
    write_json(target / "manifest.json", manifest)
    return manifest
