"""Prompt-family grouped, approximately category-stratified dataset splitting.

Every row of a prompt family (repeats, variants) lands in the same split. Families are
allocated globally rather than per category, so small corpora still produce four
non-empty splits:

1. Within each category, families are shuffled (seeded) and given evenly spaced
   positions in [0, 1).
2. All families are sorted by position (seeded tie-break) and the global list is cut
   into train / validation / calibration / test by the target ratios.

Because each category's families are spread uniformly over [0, 1), every split
receives a roughly proportional share of each category. With very few families per
category, exact per-category proportions are impossible; the global ratios are kept
instead.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from seshat.data.io import read_parquet, write_parquet

SPLIT_NAMES = ("train", "validation", "calibration", "test")
DEFAULT_RATIOS = {"train": 0.64, "validation": 0.12, "calibration": 0.08, "test": 0.16}


def _counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of ``n`` families, with at least one per split."""
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Split ratios must define exactly {list(SPLIT_NAMES)}")
    if any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError("Every split ratio must be positive")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    if n < len(SPLIT_NAMES):
        raise ValueError(
            f"Need at least {len(SPLIT_NAMES)} prompt families to create non-empty "
            f"splits; got {n}"
        )
    raw = {name: n * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    for name in SPLIT_NAMES:
        if counts[name] == 0:
            donor = max(SPLIT_NAMES, key=lambda split: counts[split])
            counts[donor] -= 1
            counts[name] = 1
    return counts


def assign_grouped_splits(
    frame: pd.DataFrame,
    *,
    seed: int = 1729,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return one ``(prompt_family_id, category, split)`` row per prompt family."""
    ratios = ratios or DEFAULT_RATIOS
    required = {"prompt_family_id", "category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    groups = frame[["prompt_family_id", "category"]].astype(str).drop_duplicates()
    if groups.empty:
        raise ValueError("No prompt families to split (input has no eligible rows)")
    multi_category = groups.loc[groups["prompt_family_id"].duplicated(), "prompt_family_id"]
    if not multi_category.empty:
        raise ValueError(
            f"Prompt families span multiple categories: {sorted(set(multi_category))[:5]}"
        )

    rng = random.Random(seed)
    placed: list[tuple[float, float, str, str]] = []
    for category, category_groups in groups.groupby("category", sort=True):
        family_ids = sorted(category_groups["prompt_family_id"].tolist())
        rng.shuffle(family_ids)
        size = len(family_ids)
        for index, family_id in enumerate(family_ids):
            placed.append(((index + 0.5) / size, rng.random(), family_id, str(category)))
    placed.sort()

    counts = _counts(len(placed), ratios)
    assignments: list[dict[str, str]] = []
    cursor = 0
    for split_name in SPLIT_NAMES:
        for _, _, family_id, category in placed[cursor : cursor + counts[split_name]]:
            assignments.append(
                {"prompt_family_id": family_id, "category": category, "split": split_name}
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
