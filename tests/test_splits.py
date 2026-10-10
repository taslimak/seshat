import pandas as pd
import pytest

from seshat.data.split import assign_grouped_splits


def test_prompt_families_never_leak_across_splits() -> None:
    rows = []
    for index in range(50):
        for repeat in range(2):
            rows.append(
                {
                    "prompt_family_id": f"pf_{index:03d}",
                    "category": "coding" if index < 25 else "qa",
                    "repeat": repeat,
                }
            )
    assignments = assign_grouped_splits(pd.DataFrame(rows), seed=123)
    assert assignments["prompt_family_id"].is_unique
    assert set(assignments["split"]) == {"train", "validation", "calibration", "test"}


def _families(per_category: int, categories: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"prompt_family_id": f"pf_{c}_{i}", "category": f"cat_{c:02d}"}
            for c in range(categories)
            for i in range(per_category)
        ]
    )


def test_small_corpus_still_fills_every_split() -> None:
    # Mirrors the seed prompt bank: 18 categories x 2 families.
    assignments = assign_grouped_splits(_families(2, 18), seed=1729)
    counts = assignments["split"].value_counts().to_dict()
    assert counts == {"train": 23, "test": 6, "validation": 4, "calibration": 3}
    # Every category keeps at least one family in train.
    assert assignments.loc[assignments["split"] == "train", "category"].nunique() == 18


def test_minimum_corpus_gets_one_family_per_split() -> None:
    assignments = assign_grouped_splits(_families(1, 4), seed=7)
    assert sorted(assignments["split"]) == ["calibration", "test", "train", "validation"]


def test_splits_are_deterministic_for_a_seed() -> None:
    frame = _families(5, 6)
    assert assign_grouped_splits(frame, seed=3).equals(assign_grouped_splits(frame, seed=3))


def test_large_corpus_is_approximately_stratified() -> None:
    assignments = assign_grouped_splits(_families(50, 4), seed=11)
    shares = assignments.groupby("category")["split"].value_counts(normalize=True).unstack()
    assert ((shares["train"] - 0.64).abs() < 0.05).all()
    assert ((shares["test"] - 0.16).abs() < 0.05).all()


def test_empty_input_raises_clear_error() -> None:
    empty = pd.DataFrame(columns=["prompt_family_id", "category"])
    with pytest.raises(ValueError, match="No prompt families"):
        assign_grouped_splits(empty)


def test_too_few_families_raises() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        assign_grouped_splits(_families(1, 3))


def test_family_in_two_categories_raises() -> None:
    frame = pd.DataFrame(
        [
            {"prompt_family_id": "pf_1", "category": "a"},
            {"prompt_family_id": "pf_1", "category": "b"},
            *[{"prompt_family_id": f"pf_{i}", "category": "a"} for i in range(2, 6)],
        ]
    )
    with pytest.raises(ValueError, match="multiple categories"):
        assign_grouped_splits(frame)
