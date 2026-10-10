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
-> train / validation / calibration / test Parquet + manifest (lineage, label origin) + schema.json + checksums (immutable)
```

## Integration with the prediction service

`src/seshat/features.py` is shared by the online baseline predictor and experiment runner. This prevents basic request-time feature definitions from drifting between training data and prediction.

The primary v0 label is `output_tokens`. Targets and post-inference metadata are outside the training whitelist in `seshat.data.feature_registry`.

## Provider handling

Adapters preserve both normalized usage fields and the native provider usage payload. Nullable means “not reported,” not zero. Requested and resolved model identifiers are recorded separately.

The Anthropic adapter intentionally does not pretend OpenAI-style reasoning effort has identical semantics. Provider-specific effort experiments require an explicit mapping.

## Resuming and retries

Manifest `condition_id` and `run_id` values are deterministic hashes. The runner scans an existing raw JSONL file and skips completed run IDs. Transient failures use bounded exponential backoff; deterministic failures are recorded and later quarantined. With `--retry-failed`, retries append new raw lines, and `normalize` keeps the latest successful attempt per `run_id` (`raw_record_count` shows how many lines existed).

## Provenance and label policy

Every raw observation carries a `provenance` label from its adapter: mock runs are
`synthetic`, while OpenAI and Anthropic runs are `real` / `controlled_experiment`.
`normalize` carries this into a `data_origin` column, inferring it for legacy raw
files. `features` uses real outcomes only, unless `--label-origin synthetic` is
passed for pipeline tests, and never mixes origins. Releases record `label_origin`,
and synthetic releases get a `_synthetic` suffix. See
[DATASET_PIPELINE.md](DATASET_PIPELINE.md).

## Paid-call safety gate

`seshat-data run` refuses manifests containing a real provider unless
`--allow-provider-calls` or `SESHAT_ALLOW_PROVIDER_CALLS=1` is given. The check runs
before any request or file write.

## CSV view

`data/records.csv` is a small CSV view of the synthetic example observations. It is
regenerated from `examples/synthetic_v1/` (see `data/README.md`). CSV is
inspection-only; JSONL/Parquet remain canonical.

## Dataset v0.1

`configs/experiments/dataset_v0_1.json` is the experiment skeleton. The checked-in prompt bank is deliberately only a seed corpus. Expand and review it to the planned coverage before executing a full paid Dataset v0.1.

Run the mock pilot in CI first, then a small paid-provider pilot, inspect usage normalization and cost, and only then scale.
