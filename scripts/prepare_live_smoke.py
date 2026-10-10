"""Build a tightly bounded config for an explicitly approved live API smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from seshat.providers.catalog import load_catalog
from seshat.providers.registry import adapter_class

APPROVED_PROVIDERS = ("openai", "anthropic", "gemini")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=APPROVED_PROVIDERS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    model = args.model.strip()
    if model != args.model or not model or len(model) > 100:
        raise ValueError("Provide a valid exact API model identifier.")
    if not adapter_class(args.provider).makes_external_calls:
        raise ValueError("A real provider must be selected.")
    caps = load_catalog().capabilities(args.provider, model)
    if not caps.verified:
        raise ValueError("Model is not verified in the capabilities catalog.")
    source = yaml.safe_load(Path("configs/experiments/live_pilot.yaml").read_text())
    source["experiment_id"] = args.experiment_id
    source["prompts"] = {
        "generate": {"count": 17, "seed": 7},
        "bank_files": [],
        "require_approved_for_live": True,
    }
    source["targets"] = [
        {
            "provider": args.provider,
            "model": model,
            "efforts": [None],
            "max_output_tokens": [128],
        }
    ]
    source["sampling"] = {"configs_per_prompt": 1}
    source["repetitions"] = {"fraction": 0.0, "count": 1}
    source["budget"] = {"max_requests": 2, "max_cost_usd": "0.20"}
    source["execution"] = {
        "concurrency": 1,
        "per_provider_concurrency": 1,
        "timeout_seconds": 60,
        "max_retries": 0,
        "preflight": "approximate",
        "rate_limits": {args.provider: {"requests_per_minute": 5}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
