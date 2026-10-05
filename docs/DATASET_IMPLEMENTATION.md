# Dataset Implementation

The repository now implements the architecture in `DATASET_CREATION_WORKFLOW.md`.

## Connected workflow

```text
prompts/*.jsonl
-> configs/experiments/*.json
-> seshat.experiments.manifest
-> data/manifests/*.parquet
-> seshat.experiments.runner
-> provider adapter
-> data/raw/*.jsonl
-> seshat.data.normalize
-> data/normalized/*.parquet
-> seshat.data.validate
-> PASS / WARNING / QUARANTINE / REJECT
-> seshat.data.build_features
-> hard leakage guard
-> seshat.data.split
-> prompt-family grouped assignments
-> seshat.data.release
-> train / validation / calibration / test Parquet + checksums
```

## Integration with the prediction service

`src/seshat/features.py` is shared by the online baseline predictor and experiment runner. This prevents basic request-time feature definitions from drifting between training data and prediction.

The primary v0 label is `output_tokens`. Targets and post-inference metadata are outside the training whitelist in `seshat.data.feature_registry`.

## Provider handling

Adapters preserve both normalized usage fields and the native provider usage payload. Nullable means “not reported,” not zero. Requested and resolved model identifiers are recorded separately.

The Anthropic adapter intentionally does not pretend OpenAI-style reasoning effort has identical semantics. Provider-specific effort experiments require an explicit mapping.

## Resuming and retries

Manifest `condition_id` and `run_id` values are deterministic hashes. The runner scans an existing raw JSONL file and skips completed run IDs. Transient failures use bounded exponential backoff; deterministic failures are recorded and later quarantined.

## CSV view

`data/records.csv` is a small checked-in mock preview. Regenerate a current view with:

```bash
seshat-data export-csv --input data/normalized/pilot_001.validated.parquet --output data/records.csv --limit 100
```

CSV is inspection-only; JSONL/Parquet remain canonical.

## Dataset v0.1

`configs/experiments/dataset_v0_1.json` is the experiment skeleton. The checked-in prompt bank is deliberately only a seed corpus. Expand and review it to the planned coverage before executing a full paid Dataset v0.1.

Run the mock pilot in CI first, then a small paid-provider pilot, inspect usage normalization and cost, and only then scale.
