"""Flatten immutable raw observations into a provider-neutral table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from seshat.data.io import load_jsonl, write_parquet
from seshat.data.models import RawObservation


def normalize_raw_observations(raw_path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for observation in load_jsonl(raw_path, RawObservation):
        usage = observation.normalized_usage
        row: dict[str, object] = {
            "observation_id": observation.observation_id,
            "run_id": observation.run_id,
            "condition_id": observation.condition_id,
            "experiment_id": observation.experiment_id,
            "prompt_id": observation.prompt_id,
            "prompt_family_id": observation.prompt_family_id,
            "category": observation.category,
            "timestamp": observation.timestamp.isoformat(),
            "provider": observation.provider,
            "model_requested": observation.model_requested,
            "model_resolved": observation.model_resolved,
            "effort": observation.effort,
            "repetition": observation.repetition,
            "max_output_tokens": observation.max_output_tokens,
            "temperature": observation.temperature,
            "provider_request_id": observation.provider_request_id,
            "finish_reason": observation.finish_reason,
            "response_char_count": observation.response_char_count,
            "latency_ms": observation.latency_ms,
            "is_refusal": observation.is_refusal,
            "is_truncated": observation.is_truncated,
            "status": observation.status.value,
            "attempt_count": observation.attempt_count,
            "error_type": observation.error_type,
            "error_message": observation.error_message,
            "provider_usage_json": json.dumps(
                observation.provider_usage_raw, sort_keys=True, separators=(",", ":")
            ),
        }
        for key, value in observation.request_features.items():
            row[key] = value
        if usage is not None:
            row.update(usage.model_dump())
        else:
            for column in (
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "cached_input_tokens",
                "cache_creation_tokens",
                "tool_prompt_tokens",
                "total_tokens",
                "billable_tokens",
            ):
                row[column] = None
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_to_parquet(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    frame = normalize_raw_observations(raw_path)
    write_parquet(output_path, frame)
    return frame
