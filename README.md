# Seshat

Seshat is a token-usage prediction service for AI prompts. It estimates likely output-token usage before a request is sent, returns a confidence interval, and learns from prediction error after actual usage is observed.

## Implemented vertical slices

1. FastAPI prediction service with the existing baseline predictor.
2. Controlled-experiment data factory for reproducible training/evaluation data.
3. Versioned record store (request features, predictions, observed usage) with
   explicit real/synthetic provenance, deduplication, quarantine and privacy-safe
   retention. See [Dataset Pipeline](docs/DATASET_PIPELINE.md).

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
Real provider calls are **disabled by default**. `seshat-data run` refuses a manifest
with a non-mock provider unless you pass `--allow-provider-calls` or set
`SESHAT_ALLOW_PROVIDER_CALLS=1`.

## Test

```bash
pytest
ruff check .
```

## Run the complete data factory without API spend

```bash
seshat-data build-manifest --config configs/experiments/pilot_001.json --output data/manifests/pilot_001.parquet
seshat-data run --manifest data/manifests/pilot_001.parquet --prompts prompts/hand_authored.jsonl --output data/raw/pilot_001.jsonl
seshat-data normalize --input data/raw/pilot_001.jsonl --output data/normalized/pilot_001.parquet
seshat-data validate --input data/normalized/pilot_001.parquet --output data/normalized/pilot_001.validated.parquet
seshat-data features --input data/normalized/pilot_001.validated.parquet --output data/features/pilot_001.parquet --label-origin synthetic
seshat-data split --input data/features/pilot_001.parquet --output data/splits/pilot_001.parquet
seshat-data profile --input data/normalized/pilot_001.validated.parquet --output data/profiles/pilot_001.json
seshat-data release --features data/features/pilot_001.parquet --splits data/splits/pilot_001.parquet --version 0.0.1
```

Mock-adapter outcomes are **synthetic**. The feature builder uses real labels by
default, so the mock pilot needs `--label-origin synthetic`, and its release is
written as `data/releases/dataset_v0.0.1_synthetic/`, marked "pipeline testing only".

## Record store quick start (no API spend)

```bash
seshat-data ingest --record-type request_features --input examples/synthetic_v1/request_features.jsonl
seshat-data ingest --record-type observed_usage --input examples/synthetic_v1/observed_usage.jsonl
seshat-data ingest-experiment --raw data/raw/pilot_001.jsonl --prompts prompts/hand_authored.jsonl
seshat-data predict                # baseline predictions from stored, privacy-safe features
seshat-data compare                # predicted vs observed, reported per data origin
seshat-data store-summary
```

`data/records.csv` is a human-readable CSV view of the example observations.
Parquet/JSONL remain canonical.

## Dataset invariants

- Raw observations are append-only.
- Provider-native usage payloads are preserved.
- Post-inference fields cannot enter training features.
- Repeats and prompt-family variants stay in the same split.
- Truncated outputs are marked as censored.
- Every record is labelled `real` or `synthetic`; synthetic outcomes never become real labels.
- Prompt text is not stored by default (keyed hash + request-time features only).
- Customer telemetry does not automatically become training data.
- Protected test data is not used for iterative tuning.

See [Dataset Creation Workflow](docs/DATASET_CREATION_WORKFLOW.md), [Dataset Implementation](docs/DATASET_IMPLEMENTATION.md) and [Dataset Pipeline](docs/DATASET_PIPELINE.md).
