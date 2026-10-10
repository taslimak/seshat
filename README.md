# Seshat

Seshat is a token-usage prediction service for AI prompts. It estimates likely output-token usage before a request is sent, returns a confidence interval, and learns from prediction error after actual usage is observed.

## Implemented vertical slices

1. FastAPI prediction service with the existing baseline predictor.
2. Data collection and engineering pipeline (schema `1.0.0`): deterministic prompt
   generation, balanced experiment design, a resumable budget-capped collector for
   Anthropic, OpenAI and Gemini (official SDKs) plus offline mock providers,
   consented telemetry and authorized log import, provider-specific usage
   normalization, validation, leakage-safe training datasets and CSV export.
   See [architecture](docs/DATA_PIPELINE_ARCHITECTURE.md),
   [collection guide](docs/DATA_COLLECTION_GUIDE.md) and [schema](docs/DATA_SCHEMA.md).

```text
generate / prompt bank / authorized logs / consented telemetry
-> experiment design (balanced, stratified, repetitions, budget)
-> collector (dry-run default; live needs --execute-live --max-cost-usd)
-> raw append-only event log (raw provider usage kept)
-> normalized Parquet: requests | preflight | observations | predictions
-> validation + quality report
-> leakage-safe, versioned training datasets
-> CSV export + data dictionary
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,data]"
uvicorn seshat.api:app --reload
```

For live provider experiments (official Anthropic, OpenAI and Gemini SDKs):

```bash
pip install -e ".[dev,data,providers]"
```

Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and/or `GEMINI_API_KEY` in your
environment (see `.env.example`). Never commit keys. Paid calls are **off by
default**: `seshat-data collect` sends nothing to a real provider unless you pass
both `--execute-live` and `--max-cost-usd`.

## Test

```bash
pytest
ruff check .
mypy src
```

Live adapter tests are skipped unless `SESHAT_LIVE_TESTS=1` and the provider key
are set.

## Run the data pipeline without API spend

```bash
seshat-data collect --config configs/data_collection.yaml --dry-run   # plan + cost estimate
seshat-data collect --config configs/data_collection.yaml             # mock providers, offline
seshat-data validate
seshat-data export --format csv
seshat-data prepare-training --version 0.1.0 --label-origin synthetic
seshat-data stats
```

Mock-provider outcomes are test fixtures (`label_quality=synthetic`).
`prepare-training` uses real labels by default and refuses mock data unless you
pass `--label-origin synthetic`, which writes `data/training/dataset_v0.1.0_synthetic/`.
Example outputs are committed in `data/fixtures/offline_example/`.

## Live pilot

```bash
seshat-data collect --config configs/experiments/live_pilot.yaml --dry-run
seshat-data collect --config configs/experiments/live_pilot.yaml --execute-live --max-cost-usd 5
```

Check `configs/pricing/` against the providers' pricing pages first.

## Dataset invariants

- Raw collection events are append-only.
- Provider-native usage payloads are preserved.
- Post-inference fields cannot enter training features.
- Repeats and prompt-family variants stay in the same split.
- Truncated outputs are marked as censored.
- Every record carries its `source_type` and `label_quality`; synthetic or mock outcomes never become real labels.
- Preflight token counts and predictions are never stored as observed usage.
- Prompt text is not stored by default (keyed hash + request-time features only).
- Customer telemetry does not automatically become training data.
- Protected test data is not used for iterative tuning.

See [Data Pipeline Architecture](docs/DATA_PIPELINE_ARCHITECTURE.md), [Data Collection Guide](docs/DATA_COLLECTION_GUIDE.md), [Data Schema](docs/DATA_SCHEMA.md) and the original design rationale in [Dataset Creation Workflow](docs/DATASET_CREATION_WORKFLOW.md).
