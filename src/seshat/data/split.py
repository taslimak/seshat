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
TRAIN_SPLITS = ("train", "validation", "calibration")
DEFAULT_RATIOS = {"train": 0.64, "validation": 0.12, "calibration": 0.08, "test": 0.16}


def _split_names(ratios: dict[str, float]) -> tuple[str, ...]:
    unknown = set(ratios) - set(SPLIT_NAMES)
    if unknown or len(ratios) < 2:
        raise ValueError(f"Split ratios must use at least two of {list(SPLIT_NAMES)}")
    return tuple(name for name in SPLIT_NAMES if name in ratios)


def _counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of ``n`` groups, with at least one per split."""
    names = _split_names(ratios)
    if any(ratios[name] <= 0 for name in names):
        raise ValueError("Every split ratio must be positive")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    if n < len(names):
        raise ValueError(
            f"Need at least {len(names)} prompt families to create non-empty splits; got {n}"
        )
    raw = {name: n * ratios[name] for name in names}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(names, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    for name in names:
        if counts[name] == 0:
            donor = max(names, key=lambda split: counts[split])
            counts[donor] -= 1
            counts[name] = 1
    return counts


def assign_grouped_splits(
    frame: pd.DataFrame,
    *,
    seed: int = 1729,
    ratios: dict[str, float] | None = None,
    group_column: str = "prompt_family_id",
    stratify_column: str = "category",
) -> pd.DataFrame:
    """Return one ``(group, stratum, split)`` row per group.

    A group's stratum is its most common ``stratify_column`` value (ties broken by
    name), so groups that span strata still get exactly one split.
    """
    ratios = ratios or DEFAULT_RATIOS
    required = {group_column, stratify_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    pairs = frame[[group_column, stratify_column]].astype(str)
    if pairs.empty:
        raise ValueError("No prompt families to split (input has no eligible rows)")
    if group_column == "prompt_family_id" and stratify_column == "category":
        multi = pairs.drop_duplicates().loc[lambda d: d[group_column].duplicated(), group_column]
        if not multi.empty:
            raise ValueError(f"Prompt families span multiple categories: {sorted(set(multi))[:5]}")
    groups = (
        pairs.value_counts()
        .reset_index(name="n")
        .sort_values([group_column, "n", stratify_column], ascending=[True, False, True])
        .drop_duplicates(group_column)[[group_column, stratify_column]]
        .rename(columns={group_column: "prompt_family_id", stratify_column: "category"})
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
    for split_name in _split_names(ratios):
        for _, _, family_id, category in placed[cursor : cursor + counts[split_name]]:
            assignments.append(
                {"prompt_family_id": family_id, "category": category, "split": split_name}
            )
        cursor += counts[split_name]

    result = pd.DataFrame(assignments)
    if result["prompt_family_id"].duplicated().any():
        raise AssertionError("Prompt family leaked across splits")
    result = result.sort_values(["split", "category", "prompt_family_id"]).reset_index(drop=True)
    return result.rename(columns={"prompt_family_id": group_column, "category": stratify_column})


def split_to_parquet(
    features_path: str | Path,
    output_path: str | Path,
    *,
    seed: int = 1729,
) -> pd.DataFrame:
    assignments = assign_grouped_splits(read_parquet(features_path), seed=seed)
    write_parquet(output_path, assignments)
    return assignments
