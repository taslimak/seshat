"""Resumable controlled experiment runner."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from seshat.data.io import append_jsonl, load_jsonl
from seshat.data.models import ManifestRow, PromptRecord, RawObservation, RunStatus
from seshat.features import extract_prompt_features
from seshat.providers.anthropic import AnthropicAdapter
from seshat.providers.base import ProviderAdapter, is_retriable_exception
from seshat.providers.mock import MockAdapter
from seshat.providers.openai import OpenAIAdapter

AdapterFactory = Callable[[], ProviderAdapter]

PROVIDERS: dict[str, AdapterFactory] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "mock": MockAdapter,
}


def _observation_id(run_id: str) -> str:
    return f"obs_{hashlib.sha256(run_id.encode()).hexdigest()[:20]}"


def _request_features(prompt: str) -> dict[str, int | float | str | bool | None]:
    return extract_prompt_features(prompt).__dict__.copy()


def run_manifest(
    manifest: pd.DataFrame,
    prompts: list[PromptRecord],
    raw_path: str | Path,
    *,
    max_attempts: int = 4,
    retry_failed: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    target = Path(raw_path)
    existing: dict[str, RawObservation] = {}
    if target.exists():
        existing = {item.run_id: item for item in load_jsonl(target, RawObservation)}
    adapters: dict[str, ProviderAdapter] = {}
    written = 0
    for payload in manifest.to_dict(orient="records"):
        row = ManifestRow.model_validate(payload)
        prior = existing.get(row.run_id)
        if prior is not None and (prior.status == RunStatus.SUCCESS or not retry_failed):
            continue
        prompt = prompt_by_id.get(row.prompt_id)
        if prompt is None:
            raise ValueError(f"Manifest references unknown prompt_id {row.prompt_id}")
        adapter = adapters.get(row.provider)
        if adapter is None:
            factory = PROVIDERS.get(row.provider)
            if factory is None:
                raise ValueError(f"Unsupported provider {row.provider}")
            adapter = factory()
            adapters[row.provider] = adapter
        result = None
        error: Exception | None = None
        attempts = 0
        for attempts in range(1, max_attempts + 1):
            try:
                result = adapter.execute(
                    prompt=prompt.text,
                    model=row.model,
                    effort=row.effort,
                    max_output_tokens=row.max_output_tokens,
                    temperature=row.temperature,
                )
                error = None
                break
            except Exception as exc:
                error = exc
                if attempts >= max_attempts or not is_retriable_exception(exc):
                    break
                sleep(min(16.0, 2.0 ** (attempts - 1)))
        if result is not None:
            observation = RawObservation(
                observation_id=_observation_id(row.run_id),
                run_id=row.run_id,
                condition_id=row.condition_id,
                experiment_id=row.experiment_id,
                prompt_id=row.prompt_id,
                prompt_family_id=row.prompt_family_id,
                category=row.category,
                timestamp=datetime.now(UTC),
                provider=row.provider,
                model_requested=row.model,
                model_resolved=result.model_resolved,
                effort=row.effort,
                repetition=row.repetition,
                max_output_tokens=row.max_output_tokens,
                temperature=row.temperature,
                request_features=_request_features(prompt.text),
                provider_usage_raw=result.provider_usage_raw,
                normalized_usage=result.usage,
                provider_response_raw=result.provider_response_raw,
                provider_request_id=result.provider_request_id,
                finish_reason=result.finish_reason,
                response_char_count=(
                    len(result.response_text) if result.response_text is not None else None
                ),
                latency_ms=result.latency_ms,
                is_refusal=result.is_refusal,
                is_truncated=result.is_truncated,
                status=RunStatus.SUCCESS,
                attempt_count=attempts,
            )
        else:
            assert error is not None
            observation = RawObservation(
                observation_id=_observation_id(row.run_id),
                run_id=row.run_id,
                condition_id=row.condition_id,
                experiment_id=row.experiment_id,
                prompt_id=row.prompt_id,
                prompt_family_id=row.prompt_family_id,
                category=row.category,
                timestamp=datetime.now(UTC),
                provider=row.provider,
                model_requested=row.model,
                effort=row.effort,
                repetition=row.repetition,
                max_output_tokens=row.max_output_tokens,
                temperature=row.temperature,
                request_features=_request_features(prompt.text),
                status=RunStatus.FAILED,
                attempt_count=attempts,
                error_type=type(error).__name__,
                error_message=str(error)[:1000],
            )
        append_jsonl(target, observation)
        existing[row.run_id] = observation
        written += 1
    return written
