"""Validate a reviewed batch config (from ``seshat-data plan-batch``) for a paid run.

The config must be committed under ``configs/experiments/batches/`` (so it was
reviewed in a pull request), target only the provider/model chosen in the
workflow, use a catalog-verified model and stay within the $2.00 per-run cap.
The GitHub run id is appended to ``experiment_id`` so a re-dispatch can never
resume (and silently re-pay for) another run's samples on a fresh runner.
No API calls are made here.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from decimal import Decimal
from pathlib import Path

import yaml

from seshat.experiments.config import CollectionConfig
from seshat.providers.catalog import load_catalog

BATCH_DIR = Path("configs/experiments/batches")
MAX_RUN_COST_USD = Decimal("2.00")
# Must finish well inside the workflow's timeout-minutes (120).
MAX_PACED_MINUTES = 90


def prepare(
    config_path: str, *, provider: str, model: str, run_id: str, data_root: str
) -> dict[str, object]:
    cleaned = config_path.strip().strip("'\"")
    path = Path(cleaned)
    resolved = path.resolve()
    if BATCH_DIR.resolve() not in resolved.parents or path.suffix not in {".yaml", ".yml"}:
        raise ValueError(
            f"Batch configs must be YAML files under {BATCH_DIR}/ (received {config_path!r})"
        )
    if not path.is_file():
        available = sorted(p.as_posix() for p in BATCH_DIR.glob("*.y*ml"))
        raise ValueError(
            f"{cleaned} does not exist on this branch (was its PR merged into main?). "
            f"Available: {available or 'none'}"
        )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)], capture_output=True, check=False
    )
    if tracked.returncode != 0:
        raise ValueError(f"{path} is not committed; only reviewed configs may run")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = CollectionConfig.model_validate(raw)
    for target in config.targets:
        if (target.provider, target.model) != (provider, model):
            raise ValueError(
                f"Config targets {target.provider}/{target.model}, workflow selected "
                f"{provider}/{model}"
            )
    if not load_catalog().capabilities(provider, model).verified:
        raise ValueError("Unverified API model: update the catalog before live use")
    cap = config.budget.max_cost_usd
    if cap is None or cap > MAX_RUN_COST_USD:
        raise ValueError(f"budget.max_cost_usd must be set and <= {MAX_RUN_COST_USD}")
    if config.privacy.store_prompt_text or config.privacy.store_response_text:
        raise ValueError("Paid batches must not store prompt or response text")
    slowest = min(
        (limit.requests_per_minute for limit in config.execution.rate_limits.values()),
        default=None,
    )
    if slowest is not None and config.budget.max_requests / slowest > MAX_PACED_MINUTES:
        raise ValueError(
            f"{config.budget.max_requests} requests at {slowest}/min need "
            f"{config.budget.max_requests / slowest:.0f} min, over the {MAX_PACED_MINUTES} min "
            "the workflow allows; raise requests_per_minute or split the batch"
        )
    if not run_id.isdecimal():
        raise ValueError("Expected a numeric GitHub run id")
    raw["experiment_id"] = f"{config.experiment_id}_r{run_id}"
    raw["storage"] = {"root": str(Path(data_root).resolve())}
    CollectionConfig.model_validate(raw)
    return dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = prepare(
        args.config,
        provider=args.provider,
        model=args.model,
        run_id=args.run_id,
        data_root=args.root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Validated reviewed batch {config['experiment_id']}. No paid calls performed.")


if __name__ == "__main__":
    main()
