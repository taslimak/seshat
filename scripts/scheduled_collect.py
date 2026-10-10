"""Bounded, opt-in Seshat collection job for GitHub Actions.

This script never sends paid requests unless a repository owner explicitly
enables them and confirms provider-side budget controls. The request cap
protects against unbounded request volume, not against unexpected dollar charges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from seshat.data.csv_view import export_csv
from seshat.data.io import write_parquet
from seshat.data.models import ExperimentConfig, ProviderExperimentConfig
from seshat.data.normalize import normalize_to_parquet
from seshat.data.validate import validate_dataframe
from seshat.experiments.manifest import build_manifest, load_prompts
from seshat.experiments.runner import run_manifest

MAX_REQUESTS_PER_RUN = 2
MAX_OUTPUT_TOKENS = 256
PROMPT_FILE = Path("prompts/hand_authored.jsonl")


def collect(
    *,
    mode: str,
    output_dir: Path,
    now: datetime | None = None,
) -> dict[str, object]:
    """Execute a tiny deterministic batch, saving even failed observations."""
    if mode not in {"mock", "paid"}:
        raise ValueError("mode must be mock or paid")

    provider, model = "mock", "mock-v1"
    if mode == "paid":
        if os.environ.get("SESHAT_ENABLE_PAID_COLLECTION") != "true":
            raise RuntimeError(
                "Paid collection disabled: set repository variable "
                "SESHAT_ENABLE_PAID_COLLECTION=true."
            )
        if os.environ.get("SESHAT_PROVIDER_BUDGET_CONFIRMED") != "yes":
            raise RuntimeError("Confirm an external provider billing cap before paid collection.")
        provider = os.environ.get("SESHAT_COLLECTION_PROVIDER", "").strip()
        model = os.environ.get("SESHAT_COLLECTION_MODEL", "").strip()
        if provider not in {"openai", "anthropic"} or not model:
            raise RuntimeError("Select openai or anthropic and an explicit supported API model.")
        key_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        if not os.environ.get(key_var):
            raise RuntimeError(f"Missing {key_var}; configure as a GitHub Actions secret.")
    local = (now or datetime.now(ZoneInfo("America/New_York"))).astimezone(
        ZoneInfo("America/New_York")
    )
    slot = "am" if local.hour < 15 else "pm"
    batch_id = f"{local:%Y%m%d}-{slot}-{provider}"
    experiment_id = f"scheduled_{batch_id}"
    seed = int(hashlib.sha256(experiment_id.encode()).hexdigest()[:8], 16)
    prompts = load_prompts([PROMPT_FILE])
    config = ExperimentConfig(
        experiment_id=experiment_id,
        prompt_sources=[str(PROMPT_FILE)],
        providers={
            provider: ProviderExperimentConfig(
                models=[model], efforts=["auto"], max_output_tokens=MAX_OUTPUT_TOKENS
            )
        },
        repetitions=1,
        max_requests=MAX_REQUESTS_PER_RUN,
        seed=seed,
        metadata={"run_kind": "scheduled", "mode": mode},
    )
    manifest = build_manifest(config, prompts)
    if len(manifest) > MAX_REQUESTS_PER_RUN:
        raise RuntimeError("Refusing to exceed the scheduled request cap.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(output_dir / "manifest.parquet", manifest)
    raw = output_dir / "observations.jsonl"

    # NO retry on a paid API call: a timed-out call may still be billed.
    written = run_manifest(
        manifest, prompts, raw, max_attempts=1, retry_failed=False
    )
    normalized = normalize_to_parquet(raw, output_dir / "normalized.parquet")
    validated = validate_dataframe(normalized)
    write_parquet(output_dir / "validated.parquet", validated)
    export_csv(output_dir / "validated.parquet", output_dir / "records.csv")
    quality_counts = validated["quality_decision"].value_counts().to_dict()
    result: dict[str, object] = {
        "batch_id": batch_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "manifest_requests": len(manifest),
        "new_observations": written,
        "successful_observations": int((validated["status"] == "success").sum()),
        "quality_counts": {str(k): int(v) for k, v in quality_counts.items()},
        "notes": (
            "API request count is capped. This script does NOT enforce an "
            "authoritative provider-dollar or monthly billing cap."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one safe scheduled Seshat batch")
    parser.add_argument("--mode", choices=["mock", "paid"], default="mock")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(mode=args.mode, output_dir=args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
