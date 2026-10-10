# Dataset Pipeline: Versioned Records, Provenance and Storage

This guide covers the record layer that sits beside the controlled-experiment factory
described in [DATASET_IMPLEMENTATION.md](DATASET_IMPLEMENTATION.md). Experiments,
production logs, imports and synthetic generators all produce the same three
versioned record types. Those records land in one Parquet store and feed the same
feature, split and release code.

```text
controlled experiment ──run──► raw JSONL ──ingest-experiment──┐
production / import logs (JSONL) ──────────ingest─────────────┤
synthetic generator ──synth──► JSONL ──────ingest─────────────┤
                                                              ▼
                     validate · deduplicate · redact · set retention · quarantine
                                                              ▼
                 data/store/{request_features,observed_usage,prediction}/part-*.parquet
                   │                 │                            │
         features --from-store   predict (BaselinePredictor)   export-csv / purge-expired
                   ▼                 ▼
          split ─► release      compare (predicted vs observed, per data origin)
```

## 1. Versioned records (`seshat.data.records`)

Every record has `record_type` and `schema_version` (`"1.0"`). Unknown fields are
rejected, and all timestamps must carry a timezone.

| Record | Answers | Key fields |
|---|---|---|
| `RequestFeatureRecord` | What was known **before** the request? | `request_id`, `provider`, `model`, `effort`, `max_output_tokens`, `category`, `prompt_hash` + `prompt_hash_scheme`, `features` (the 9 `seshat.features` values, `feature_set_version="prompt_features/1"`), `provenance`, `retain_until` |
| `PredictionRecord` | What did Seshat predict? | `prediction_id`, `request_id`, `predictor_version`, `predicted_output_tokens`, `lower_bound ≤ prediction ≤ upper_bound`, `confidence`, `interval_calibrated` |
| `ObservedUsageRecord` | What did the provider report **after** the request? | `observation_id`, `request_id`, optional `prediction_id`, `status`, `usage` (`NormalizedUsage`), `finish_reason`, `is_truncated`, `is_refusal`, `latency_ms`, `provenance`, `retain_until`, `metadata` |

A prediction can never be read as an observation. `record_type` is a fixed literal,
so a prediction payload fails `ObservedUsageRecord` validation, and predictions live
in their own store table.

IDs are caller-supplied, or deterministic via `stable_id(prefix, value)`. Experiment
runs use `request_id = run_id`. Predictions use
`prediction_id = stable_id("pred", request_id | predictor_version)`.

## 2. Provenance: real vs synthetic (`seshat.data.provenance`)

```json
{"data_origin": "real|synthetic", "source": "controlled_experiment|production_log|import|mock_adapter|simulation",
 "generator": "...", "experiment_id": "...", "pipeline_version": "0.2.0"}
```

The guarantees, and what enforces each one:

| Guarantee | Enforced by |
|---|---|
| Mock and simulation output is always `synthetic` | `Provenance` validator (`source=mock_adapter`/`simulation` + `real` is a schema error) |
| Experiment runs are labelled by their adapter | `ProviderAdapter.data_origin` → `provenance_for_provider` → stamped on every `RawObservation` |
| Legacy raw files without provenance still get the correct label | `normalize` infers it from the adapter registry |
| Nothing is assumed real | `ingest` quarantines records without provenance unless `--origin`/`--source` are passed explicitly |
| Synthetic outcomes are never used as real labels | `features` defaults to `--label-origin real` and refuses if no real rows exist; one origin per matrix |
| A release declares what its labels are | `manifest.json` `label_origin` + `intended_use`; synthetic releases are named `dataset_v<version>_synthetic` and marked *pipeline testing only* |
| Accuracy is never pooled across origins | `compare` summarizes per `(predictor_version, data_origin)` and annotates synthetic results |

The label origin of a store row is the origin of the **observed outcome**.

## 3. Privacy-safe retention (`seshat.data.privacy`)

| Data | Default handling |
|---|---|
| Prompt text | Not stored. Ingestion computes `prompt_hash` and the request features from the text, then drops it (`prompt_text_dropped` in the report). `--keep-prompt-text` opts in. |
| Prompt hash | HMAC-SHA256 with `SESHAT_PROMPT_HASH_KEY` when set (recommended for production; unkeyed hashes of short prompts can be guessed), otherwise SHA-256. The scheme is stored with the hash. |
| Error messages, string metadata | E-mail addresses, bearer tokens, `sk-`/`pk-`/`rk-` keys, AWS key IDs and 12–19-digit numbers are redacted; text is capped at 500 characters |
| Quarantine files | Same redaction; `prompt_text`/`text`/`prompt` keys are removed unless prompt text retention is enabled; Pydantic errors are recorded without input values |
| Response text | Never stored in the record layer (only `response_char_count`) |
| Retention | `retain_until` = event time + 90 days for `production_log` and `import` records (`--production-days`, `--import-days`). Controlled-experiment and synthetic records do not expire. `seshat-data purge-expired` deletes expired rows and their predictions, and logs counts only. |

The experiment raw files under `data/raw/` keep the run's full provider payload
(immutable by design). They use project-authored prompts and stay gitignored.

## 4. Ingestion, validation and deduplication (`seshat.data.ingest`)

Each ingested record goes through these steps in order. Every skipped record is
either counted as an exact duplicate or written to
`data/quarantine/<batch>-<type>.jsonl` with its reason.

| Step | Outcome | Quarantine reason |
|---|---|---|
| JSON parse | — | `invalid_json`, `invalid_record` |
| `record_type` matches the target table | — | `wrong_record_type` |
| Provenance present (or explicit default) | — | `missing_provenance` |
| Schema validation (versioned Pydantic model) | — | `schema_error` (with field locations) |
| Same ID **and** same content already stored | skipped, counted in `duplicates` (idempotent re-ingest) | — |
| Same ID, different content | — | `conflicting_duplicate_id` |
| Provider request ID already used by another observation | — | `duplicate_provider_request_id` |
| Quality rules (`seshat.data.validate`: failed run, missing/zero/negative tokens, total < input + output, refusal, truncation) | PASS/WARNING stored with `quality_decision` + `quality_flags` | `quality_quarantine`, `quality_reject` |

Content identity is `record_hash`, a SHA-256 of the canonical record excluding
`retain_until`. Each batch is written atomically as a new immutable Parquet part.
`_ingestion_log.jsonl` records counts per batch and purge, never content.

## 5. Store (`seshat.data.store`)

```text
data/store/
  request_features/part-<batch>.parquet
  observed_usage/part-<batch>.parquet
  prediction/part-<batch>.parquet
  _ingestion_log.jsonl
```

Rows are the flattened records (`seshat.data.records.to_row`). Usage fields become
columns (`output_tokens`, …), provenance becomes `data_origin`, `data_source`,
`data_generator`, `experiment_id` and `pipeline_version`, and features become the
same column names the feature registry uses. `data/store/` is gitignored.

## 6. Reproducible experiments

- **Manifests:** run, condition and observation IDs are deterministic hashes. Prompt sources resolve from the project root (nearest `pyproject.toml`) or `--root`.
- **Runs:** **mock only by default.** Manifests with a real provider fail before any request unless `--allow-provider-calls` or `SESHAT_ALLOW_PROVIDER_CALLS=1` is set.
- **Retries:** the raw file stays append-only. `normalize` and `ingest-experiment` keep the latest successful attempt per `run_id` (`raw_record_count` shows how many raw lines existed).
- **Splits:** prompt-family grouped, approximately category-stratified, seeded.
- **Releases:** immutable (an existing version is refused). `manifest.json` records the git commit and dirty flag, `label_origin`, source experiments, providers, models, split counts, censored-target count and checksums of inputs and outputs. `schema.json` records the role of every column from the feature registry.
- **Synthetic data:** `seshat-data synth` is a pure function of `(prompt bank, n, seed)`; the committed example is regenerated and compared byte-for-byte in tests.

## 7. CLI reference

```bash
# Synthetic example → store → predictions → comparison (no API spend)
seshat-data synth --prompts prompts/hand_authored.jsonl --n 48 --seed 1729 --output-dir examples/synthetic_v1
seshat-data ingest --record-type request_features --input examples/synthetic_v1/request_features.jsonl
seshat-data ingest --record-type observed_usage   --input examples/synthetic_v1/observed_usage.jsonl
seshat-data predict
seshat-data compare --output data/compare.csv
seshat-data store-summary

# Controlled experiment → store
seshat-data ingest-experiment --raw data/raw/pilot_001.jsonl --prompts prompts/hand_authored.jsonl

# External logs without embedded provenance must say what they are
seshat-data ingest --record-type observed_usage --input usage.jsonl --origin real --source production_log

# Store → features → split → release (real labels by default)
seshat-data features --from-store data/store --output data/features/store.parquet [--label-origin synthetic]
seshat-data split --input data/features/store.parquet --output data/splits/store.parquet
seshat-data release --features data/features/store.parquet --splits data/splits/store.parquet --version 0.1.0

# Inspection and retention
seshat-data export-csv --input data/store/observed_usage --output data/observed_usage.csv [--columns ...] [--limit N]
seshat-data purge-expired [--as-of 2026-12-31T00:00:00+00:00]
```

Every command exits with code 2 and a one-line `error:` message for invalid input,
refused releases, blocked provider calls or missing data.

CSV export accepts a Parquet file, a JSONL file or a store table directory. Text
cells beginning with `=`, `+`, `-`, `@`, tab or CR are prefixed with `'` so
spreadsheets do not execute them.

## 8. Example data

- `examples/synthetic_v1/` — 48 synthetic request/observation pairs (`seed=1729`), labelled `synthetic`/`simulation`.
- `examples/edge_cases/` — 12 hand-written lines that exercise every ingestion guard, with `expected_report.json`.
- `data/records.csv` — CSV view of `examples/synthetic_v1/observed_usage.jsonl` after ingestion (regenerate as in `data/README.md`).

There is **no real observed data in the repository**. Real data enters only through
a configured provider run or an explicit `--origin real` ingestion.

## 9. Configuration

| Variable | Effect |
|---|---|
| `SESHAT_ALLOW_PROVIDER_CALLS=1` | Permits real provider requests in `seshat-data run` |
| `SESHAT_PROMPT_HASH_KEY` | Secret for HMAC prompt hashing |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | Provider SDK credentials (only read when calls are allowed) |

## 10. Known limitations

- The baseline predictor ignores `model` and its interval is not calibrated (`interval_calibrated=false`). `compare` measures it but doesn't fix it.
- `compare` excludes truncated (right-censored) observations from error metrics but still counts them.
- The store is file-based and single-writer. Concurrent ingestion into one store is not supported. A transactional database becomes worthwhile once the API or SDK ingests live traffic.
- `/v1/predict` does not yet persist `PredictionRecord`s or return a `prediction_id`. That is the next integration step for the actual-vs-predicted loop.
