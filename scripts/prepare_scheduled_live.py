"""Generate a bounded, reproducible live experiment for one Eastern-time slot.

This program never reads or prints provider API keys and never dispatches calls.
The live collector has separate explicit opt-in and spend authorization gates.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from seshat.providers.catalog import load_catalog

EASTERN = ZoneInfo("America/New_York")
MAX_REQUESTS = 4
MAX_RUN_COST_USD = "0.15"


def make_config(
    *, provider: str, model: str, now: datetime, storage_root: str
) -> dict[str, object]:
    if provider not in {"openai", "anthropic", "gemini"}:
        raise ValueError("Unsupported real provider")
    if not model or len(model) > 100 or model.strip() != model:
        raise ValueError("Select an explicit API model")
    capabilities = load_catalog().capabilities(provider, model)
    if not capabilities.verified:
        raise ValueError("Unverified API model: update catalog before live use")
    if not Path(storage_root).is_absolute():
        raise ValueError("Storage root must be an absolute persistent filesystem path")
    local = now.astimezone(EASTERN)
    slot = "morning" if local.hour < 15 else "evening"
    day = local.strftime("%Y%m%d")
    experiment_id = f"scheduled_{day}_{slot}_{provider}"
    # A new seed rotates deterministic template variants across days.
    seed = int(day) + (0 if slot == "morning" else 1)
    return {
        "experiment_id": experiment_id,
        "description": "Cost-controlled scheduled real-usage collection",
        "seed": seed,
        "prompts": {
            "generate": {"count": 17, "seed": seed},
            "bank_files": [],
            "require_approved_for_live": True,
        },
        "targets": [{
            "provider": provider,
            "model": model,
            "efforts": [None],
            "max_output_tokens": [256],
        }],
        "sampling": {"configs_per_prompt": 1},
        "repetitions": {"fraction": 0, "count": 1},
        "budget": {"max_requests": MAX_REQUESTS, "max_cost_usd": MAX_RUN_COST_USD},
        "execution": {
            "concurrency": 1,
            "per_provider_concurrency": 1,
            "timeout_seconds": 60,
            "max_retries": 0,
            "preflight": "approximate",
            "rate_limits": {provider: {"requests_per_minute": 5}},
        },
        "privacy": {"store_prompt_text": False, "store_response_text": False},
        "storage": {"root": storage_root},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = make_config(
        provider=os.environ.get("SESHAT_COLLECTION_PROVIDER", ""),
        model=os.environ.get("SESHAT_COLLECTION_MODEL", ""),
        now=datetime.now(EASTERN),
        storage_root=os.environ.get("SESHAT_PERSISTENT_DATA_ROOT", ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("Wrote safe scheduled experiment configuration; no API calls sent.")


if __name__ == "__main__":
    main()
