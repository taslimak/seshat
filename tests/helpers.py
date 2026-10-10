"""Shared payload builders for tests (imported as ``helpers``)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from seshat.data.provenance import DataOrigin, DataSource, Provenance

REAL_PRODUCTION = Provenance(data_origin=DataOrigin.REAL, source=DataSource.PRODUCTION_LOG)


def observation_payload(**overrides: Any) -> dict[str, Any]:
    """A valid real production observed-usage payload (JSON form)."""
    payload: dict[str, Any] = {
        "observation_id": "obs_1",
        "request_id": "req_1",
        "observed_at": datetime(2026, 9, 1, tzinfo=UTC).isoformat(),
        "provider": "openai",
        "model_requested": "model-a",
        "provider_request_id": "resp_1",
        "status": "success",
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        "provenance": REAL_PRODUCTION.model_dump(mode="json"),
    }
    payload.update(overrides)
    return payload


def request_payload(**overrides: Any) -> dict[str, Any]:
    """A valid real production request-feature payload carrying prompt text."""
    payload: dict[str, Any] = {
        "request_id": "req_1",
        "created_at": datetime(2026, 9, 1, tzinfo=UTC).isoformat(),
        "provider": "openai",
        "model": "model-a",
        "effort": "low",
        "prompt_text": "Summarize the attached report for jane@example.com.",
        "provenance": REAL_PRODUCTION.model_dump(mode="json"),
    }
    payload.update(overrides)
    return payload
