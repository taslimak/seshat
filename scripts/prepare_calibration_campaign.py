"""Prepare a diversified, manually authorized, budget-bounded live data batch."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scripts.prepare_scheduled_live import make_config

EASTERN = ZoneInfo("America/New_York")


def campaign_config(*, provider: str, model: str, run_id: str, root: str) -> dict[str, object]:
    if not run_id.isdecimal() or not 1 <= len(run_id) <= 20:
        raise ValueError("Expected a numeric GitHub run ID")
    config = make_config(
        provider=provider,
        model=model,
        now=datetime.now(EASTERN),
        storage_root=root,
    )
    seed = int(run_id) % 2_000_000_000
    config["experiment_id"] = f"calibration_{run_id}_{provider}"
    config["seed"] = seed
    config["description"] = "Distinct manual calibration batch; not auto-trained"
    config["prompts"] = {
        "generate": {"count": 160, "seed": seed},
        "bank_files": [],
        "require_approved_for_live": True,
    }
    config["targets"] = [
        {
            "provider": provider,
            "model": model,
            "efforts": [None],
            "max_output_tokens": [512],
        }
    ]
    config["sampling"] = {"configs_per_prompt": 1}
    config["budget"] = {"max_requests": 40, "max_cost_usd": "0.75"}
    config["execution"] = {
        "concurrency": 1,
        "per_provider_concurrency": 1,
        "timeout_seconds": 90,
        "max_retries": 0,
        "preflight": "approximate",
        "rate_limits": {provider: {"requests_per_minute": 5}},
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = campaign_config(
        provider=args.provider,
        model=args.model,
        run_id=args.run_id,
        root=str(Path(args.root).resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("Calibration config ready. No paid calls performed by this script.")


if __name__ == "__main__":
    main()
