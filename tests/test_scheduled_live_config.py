"""Safety and variation tests for the scheduled paid collection configuration."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scripts.prepare_scheduled_live import make_config


def test_limits_and_rotation() -> None:
    eastern = ZoneInfo("America/New_York")
    morning = make_config(
        provider="openai",
        model="gpt-6.1-sol",
        now=datetime(2026, 10, 10, 9, 17, tzinfo=eastern),
        storage_root="/var/lib/seshat/data",
    )
    evening = make_config(
        provider="openai",
        model="gpt-6.1-sol",
        now=datetime(2026, 10, 10, 21, 17, tzinfo=eastern),
        storage_root="/var/lib/seshat/data",
    )
    assert morning["experiment_id"] != evening["experiment_id"]
    assert morning["budget"] == {"max_requests": 4, "max_cost_usd": "0.15"}
    assert morning["execution"]["max_retries"] == 0
    assert morning["storage"]["root"] == "/var/lib/seshat/data"
    assert morning["targets"][0]["max_output_tokens"] == [256]


def test_rejects_unverified_model_and_ephemeral_paths() -> None:
    now = datetime.now(ZoneInfo("America/New_York"))
    with pytest.raises(ValueError):
        make_config(
            provider="openai",
            model="not-an-api-model",
            now=now,
            storage_root="/var/lib/seshat/data",
        )
    with pytest.raises(ValueError, match="absolute"):
        make_config(
            provider="openai",
            model="gpt-6.1-sol",
            now=now,
            storage_root="data",
        )
