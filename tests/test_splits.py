import pandas as pd

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
