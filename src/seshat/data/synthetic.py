"""Deterministic synthetic request-feature and observed-usage records.

Synthetic records exercise ingestion, validation, storage, splitting and comparison
without provider spend. They are always labelled ``data_origin=synthetic`` /
``source=simulation``, and the feature builder refuses them as labels unless
``label_origin="synthetic"`` is requested explicitly.

Output is a pure function of ``(prompts, n, seed, start)``: IDs, timestamps and
usage are all derived from those inputs, so regenerating the example dataset is
byte-for-byte reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from seshat.data.models import EffortLevel, PromptRecord, RunStatus
from seshat.data.privacy import RetentionPolicy
from seshat.data.provenance import DataOrigin, DataSource, Provenance
from seshat.data.records import ObservedUsageRecord, RequestFeatureRecord, stable_id
from seshat.providers.mock import MockAdapter

GENERATOR = "seshat.data.synthetic/1"
DEFAULT_START = datetime(2026, 1, 1, tzinfo=UTC)
MODELS = ("mock-small", "mock-large")
EFFORTS: tuple[EffortLevel, ...] = ("low", "medium", "high")
MAX_OUTPUT_TOKENS = (256, 1024)


def synthetic_provenance() -> Provenance:
    return Provenance(
        data_origin=DataOrigin.SYNTHETIC,
        source=DataSource.SIMULATION,
        generator=GENERATOR,
    )


def generate_synthetic_records(
    prompts: Sequence[PromptRecord],
    *,
    n: int,
    seed: int = 1729,
    start: datetime = DEFAULT_START,
) -> tuple[list[RequestFeatureRecord], list[ObservedUsageRecord]]:
    """Generate ``n`` request/observation pairs drawn from ``prompts``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not prompts:
        raise ValueError("At least one prompt is required")
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    rng = random.Random(seed)
    ordered = sorted(prompts, key=lambda prompt: prompt.prompt_id)
    adapter = MockAdapter()
    provenance = synthetic_provenance()
    policy = RetentionPolicy(store_prompt_text=False)
    requests: list[RequestFeatureRecord] = []
    observations: list[ObservedUsageRecord] = []
    for index in range(n):
        prompt = rng.choice(ordered)
        model = rng.choice(MODELS)
        effort = rng.choice(EFFORTS)
        max_output_tokens = rng.choice(MAX_OUTPUT_TOKENS)
        created_at = start + timedelta(minutes=index)
        request_id = stable_id("req", f"synthetic|{seed}|{index}")
        requests.append(
            RequestFeatureRecord.from_prompt(
                request_id=request_id,
                prompt=prompt.text,
                provider="mock",
                model=model,
                provenance=provenance,
                created_at=created_at,
                policy=policy,
                hash_key="",
                effort=effort,
                max_output_tokens=max_output_tokens,
                temperature=None,
                category=prompt.category,
                prompt_family_id=prompt.prompt_family_id,
            )
        )
        result = adapter.execute(
            prompt=prompt.text,
            model=model,
            effort=effort,
            max_output_tokens=max_output_tokens,
            temperature=None,
        )
        latency_ms = round(150 + 4.0 * (result.usage.output_tokens or 0) + rng.uniform(0, 50), 1)
        observations.append(
            ObservedUsageRecord(
                observation_id=stable_id("obs", f"synthetic-observation|{seed}|{index}"),
                request_id=request_id,
                observed_at=created_at + timedelta(milliseconds=latency_ms),
                provider="mock",
                model_requested=model,
                model_resolved=model,
                provider_request_id=stable_id(
                    "mockreq", f"synthetic-provider-request|{seed}|{index}"
                ),
                status=RunStatus.SUCCESS,
                usage=result.usage,
                finish_reason=result.finish_reason,
                is_truncated=result.is_truncated,
                is_refusal=False,
                latency_ms=latency_ms,
                response_char_count=len(result.response_text or ""),
                provenance=provenance,
                metadata={"generator_seed": seed, "generator_index": index},
            )
        )
    return requests, observations
