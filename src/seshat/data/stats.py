"""Collection statistics per provider/model from the normalized layer."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.storage import StorageLayout


def collection_stats(root: str | Path) -> dict[str, Any]:
    layout = StorageLayout(root)
    obs = layout.read_frame("observations")
    if obs.empty:
        return {"observations": 0, "by_provider_model": {}}
    obs = obs.assign(
        cost=obs["calculated_cost_usd"].map(
            lambda v: None if v is None or pd.isna(v) else float(Decimal(str(v)))
        )
    )
    out: dict[str, Any] = {"observations": int(len(obs)), "by_provider_model": {}}
    for (provider, model), group in obs.groupby(["provider", "requested_model"]):
        latency = group["total_latency_ms"].dropna()
        out["by_provider_model"][f"{provider}/{model}"] = {
            "observations": int(len(group)),
            "source_types": sorted(group["source_type"].unique().tolist()),
            "completed": int((group["response_status"] == "completed").sum()),
            "incomplete": int((group["response_status"] == "incomplete").sum()),
            "failed": int((group["response_status"] == "failed").sum()),
            "uncertain": int((group["response_status"] == "uncertain").sum()),
            "retries": int(group["retry_count"].sum()),
            "rate_limited": int(group["rate_limit_encountered"].sum()),
            "input_tokens": int(group["actual_input_tokens"].fillna(0).sum()),
            "output_tokens": int(group["actual_output_tokens"].fillna(0).sum()),
            "reasoning_tokens": int(group["actual_reasoning_tokens"].fillna(0).sum()),
            "cached_input_tokens": int(group["actual_cached_input_tokens"].fillna(0).sum()),
            "estimated_cost_usd": round(float(group["cost"].fillna(0).sum()), 6),
            "latency_ms_p50": float(latency.median()) if len(latency) else None,
            "latency_ms_p95": float(latency.quantile(0.95)) if len(latency) else None,
        }
    return out
