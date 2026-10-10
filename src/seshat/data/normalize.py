"""Flatten immutable raw observations into a provider-neutral table."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from seshat.data.io import load_jsonl, write_parquet
from seshat.data.models import NormalizedUsage, RawObservation, RunStatus
from seshat.data.provenance import Provenance
from seshat.providers.registry import provenance_for_provider


def select_final_attempts(observations: list[RawObservation]) -> list[tuple[RawObservation, int]]:
    """Collapse retried runs to one observation per ``run_id``.

    The raw file is append-only, so a run retried with ``--retry-failed`` has several
    lines. The latest successful attempt wins; otherwise the latest failure. Returns
    ``(observation, number_of_raw_records_for_run)`` in first-seen run order.
    """
    by_run: dict[str, list[RawObservation]] = defaultdict(list)
    for observation in observations:
        by_run[observation.run_id].append(observation)
    selected: list[tuple[RawObservation, int]] = []
    for attempts in by_run.values():
        final = max(
            attempts,
            key=lambda item: (item.status == RunStatus.SUCCESS, item.timestamp),
        )
        selected.append((final, len(attempts)))
    return selected


def observation_provenance(observation: RawObservation) -> Provenance:
    """Recorded provenance, or the adapter's provenance for legacy raw records."""
    if observation.provenance is not None:
        return observation.provenance
    return provenance_for_provider(observation.provider, observation.experiment_id)


def normalize_raw_observations(raw_path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    observations = load_jsonl(raw_path, RawObservation)
    for observation, raw_record_count in select_final_attempts(observations):
        usage = observation.normalized_usage
        provenance = observation_provenance(observation)
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
            "raw_record_count": raw_record_count,
            "error_type": observation.error_type,
            "error_message": observation.error_message,
            "data_origin": provenance.data_origin.value,
            "data_source": provenance.source.value,
            "provider_usage_json": json.dumps(
                observation.provider_usage_raw, sort_keys=True, separators=(",", ":")
            ),
        }
        for key, value in observation.request_features.items():
            row[key] = value
        if usage is not None:
            row.update(usage.model_dump())
        else:
            for column in NormalizedUsage.model_fields:
                row[column] = None
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_to_parquet(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    frame = normalize_raw_observations(raw_path)
    write_parquet(output_path, frame)
    return frame
