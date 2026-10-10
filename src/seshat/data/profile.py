"""Dataset health summaries used before model training."""

from __future__ import annotations

from typing import Any

import pandas as pd


def profile_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "observations": int(len(frame)),
        "columns": list(frame.columns),
        "missing_rate": {key: float(value) for key, value in frame.isna().mean().items()},
    }
    for column in ("provider", "model_requested", "category", "effort", "quality_decision"):
        if column in frame.columns:
            profile[f"by_{column}"] = {
                str(key): int(value)
                for key, value in frame[column].value_counts(dropna=False).items()
            }
    if "output_tokens" in frame.columns:
        output = pd.to_numeric(frame["output_tokens"], errors="coerce").dropna()
        if not output.empty:
            profile["output_tokens"] = {
                "mean": float(output.mean()),
                "median": float(output.median()),
                "std": float(output.std(ddof=1)) if len(output) > 1 else 0.0,
                "min": float(output.min()),
                "max": float(output.max()),
                "p10": float(output.quantile(0.10)),
                "p90": float(output.quantile(0.90)),
            }
    if "run_id" in frame.columns:
        profile["duplicate_run_rate"] = float(frame["run_id"].duplicated().mean())
    return profile
