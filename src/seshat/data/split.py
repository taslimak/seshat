"""Prompt-family grouped dataset splitting."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from seshat.data.io import read_parquet, write_parquet

DEFAULT_RATIOS = {"train": 0.64, "validation": 0.12, "calibration": 0.08, "test": 0.16}


def _counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    raw = {name: n * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def assign_grouped_splits(
    frame: pd.DataFrame,
    *,
    seed: int = 1729,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    ratios = ratios or DEFAULT_RATIOS
    required = {"prompt_family_id", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    groups = frame[["prompt_family_id", "category"]].drop_duplicates()
    assignments: list[dict[str, str]] = []
    rng = random.Random(seed)
    for category, category_groups in groups.groupby("category", sort=True):
        family_ids = sorted(category_groups["prompt_family_id"].astype(str).tolist())
        rng.shuffle(family_ids)
        counts = _counts(len(family_ids), ratios)
        cursor = 0
        for split_name in ("train", "validation", "calibration", "test"):
            for family_id in family_ids[cursor : cursor + counts[split_name]]:
                assignments.append(
                    {"prompt_family_id": family_id, "category": str(category), "split": split_name}
                )
            cursor += counts[split_name]
    result = pd.DataFrame(assignments)
    if result["prompt_family_id"].duplicated().any():
        raise AssertionError("Prompt family leaked across splits")
    return result.sort_values(["split", "category", "prompt_family_id"]).reset_index(drop=True)


def split_to_parquet(
    features_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 1729,
) -> pd.DataFrame:
    assignments = assign_grouped_splits(read_parquet(features_path), seed=seed)
    write_parquet(output_path, assignments)
    return assignments
