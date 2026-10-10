from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from seshat.experiments.config import CollectionConfig

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]


def provider_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / "provider_responses" / name).read_text(encoding="utf-8"))


@pytest.fixture
def small_config(tmp_path: Path) -> Callable[..., CollectionConfig]:
    """Factory for a small offline config rooted in ``tmp_path``."""

    def make(**overrides: Any) -> CollectionConfig:
        raw: dict[str, Any] = {
            "experiment_id": "test_exp",
            "seed": 3,
            "prompts": {"generate": {"count": 34, "seed": 3}},
            "targets": [
                {
                    "provider": "mock-anthropic",
                    "model": "mock-claude",
                    "efforts": ["low", "high"],
                    "max_output_tokens": [256, 2048],
                    "caching": [False, True],
                    "cache_ttl": "5m",
                },
                {
                    "provider": "mock-openai",
                    "model": "mock-gpt",
                    "efforts": ["low", "high"],
                    "max_output_tokens": [256, 2048],
                },
                {
                    "provider": "mock-gemini",
                    "model": "mock-gemini",
                    "efforts": ["low", "high"],
                    "max_output_tokens": [256, 2048],
                },
            ],
            "sampling": {"configs_per_prompt": 3},
            "repetitions": {"fraction": 0.2, "count": 3},
            "budget": {"max_requests": 150},
            "execution": {"backoff_base_seconds": 0, "backoff_max_seconds": 0, "concurrency": 6},
            "storage": {"root": str(tmp_path / "data")},
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(raw.get(key), dict):
                raw[key] = {**raw[key], **value}
            else:
                raw[key] = value
        return CollectionConfig.model_validate(raw)

    return make
