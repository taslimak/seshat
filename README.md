# Seshat

Seshat is a token-usage prediction service for AI prompts. It estimates likely output-token usage before a request is sent, returns a confidence interval, and learns from prediction error after actual usage is observed.

## Implemented vertical slices

1. FastAPI prediction service with the existing baseline predictor.
2. Controlled-experiment data factory for reproducible training/evaluation data.

The data path is:

```text
prompt bank
-> experiment manifest
-> provider adapter
-> immutable JSONL observations
-> normalized Parquet
-> quality validation
-> leakage-safe feature matrix
-> prompt-family grouped splits
-> versioned release
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,data]"
uvicorn seshat.api:app --reload
```

For paid provider experiments:

```bash
pip install -e ".[providers]"
```

Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in your environment. Never commit keys.

## Run the complete data factory without API spend

```bash
seshat-data build-manifest --config configs/experiments/pilot_001.json --output data/manifests/pilot_001.parquet
seshat-data run --manifest data/manifests/pilot_001.parquet --prompts prompts/hand_authored.jsonl --output data/raw/pilot_001.jsonl
seshat-data normalize --input data/raw/pilot_001.jsonl --output data/normalized/pilot_001.parquet
seshat-data validate --input data/normalized/pilot_001.parquet --output data/normalized/pilot_001.validated.parquet
seshat-data features --input data/normalized/pilot_001.validated.parquet --output data/features/pilot_001.parquet
seshat-data split --input data/features/pilot_001.parquet --output data/splits/pilot_001.parquet
seshat-data profile --input data/normalized/pilot_001.validated.parquet --output data/profiles/pilot_001.json
seshat-data release --features data/features/pilot_001.parquet --splits data/splits/pilot_001.parquet --version 0.0.1
seshat-data export-csv --input data/normalized/pilot_001.validated.parquet --output data/records.csv --limit 100
```

`data/records.csv` is a human-readable view. Parquet/JSONL remain canonical.

## Dataset invariants

- Raw observations are append-only.
- Provider-native usage payloads are preserved.
- Post-inference fields cannot enter training features.
- Repeats and prompt-family variants stay in the same split.
- Truncated outputs are marked as censored.
- Customer telemetry does not automatically become training data.
- Protected test data is not used for iterative tuning.

See [Dataset Creation Workflow](docs/DATASET_CREATION_WORKFLOW.md) and [Dataset Implementation](docs/DATASET_IMPLEMENTATION.md).
