"""Report real-data training readiness and baseline error after collection.

Only validated, verified, uncensored, controlled real API observations qualify.
This command does not send API requests or train models on insufficient data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.io import read_parquet
from seshat.data.storage import StorageLayout


def readiness(root: str | Path) -> dict[str, Any]:
    layout = StorageLayout(root)
    observations = layout.read_frame("observations")
    decisions_path = layout.validation / "decisions.parquet"
    result: dict[str, Any] = {
        "status": "insufficient_data",
        "eligible_real_uncensored": 0,
        "distinct_prompt_families": 0,
        "providers": {},
        "minimum_total_usable": 140,
        "warning": "Readiness is not a guarantee of generalization or split balance.",
    }
    if observations.empty or not decisions_path.exists():
        return result
    decisions = read_parquet(decisions_path)
    valid = decisions.loc[decisions["eligible_for_training"] & (decisions["data_origin"] == "real"), ["observation_id"]]
    eligible = observations.merge(valid, on="observation_id", how="inner")
    eligible = eligible.loc[
        (eligible["source_type"] == "controlled_experiment")
        & (eligible["label_quality"] == "verified")
        & (~eligible["is_truncated"].fillna(True).astype(bool))
        & eligible["actual_output_tokens"].notna()
    ]
    result["eligible_real_uncensored"] = int(len(eligible))
    result["providers"] = {str(k): int(v) for k, v in eligible["provider"].value_counts().items()}
    requests = layout.read_frame("requests")
    if not eligible.empty and not requests.empty:
        joined = eligible[["sample_id"]].merge(
            requests[["sample_id", "prompt_family_id"]].drop_duplicates("sample_id"),
            on="sample_id", how="left"
        )
        result["distinct_prompt_families"] = int(joined["prompt_family_id"].nunique())
    if result["eligible_real_uncensored"] >= 140 and result["distinct_prompt_families"] >= 4:
        result["status"] = "candidate_for_split_validation"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = readiness(args.root)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
