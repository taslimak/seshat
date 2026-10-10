# Seshat data pipeline architecture

Status: implemented (schema `1.0.0`, pipeline `1.0.0`). Supersedes the
manifest/runner "data factory" and the PR #5 record store; see
[module mapping](#what-replaced-what) below.

The pipeline exists to produce **trustworthy, provider-reported usage labels** for
training Seshat's token and cost predictors. Everything else (generation, design,
validation, splitting) is there to make those labels comparable, leakage-safe and
honest about where they came from.

```text
                       ┌──────────────────────── acquisition ───────────────────────┐
 generate (templates) ─┤ 1. controlled experiments  collect --dry-run | --execute-live│
 prompt bank (JSONL) ──┤ 2. consented telemetry     TelemetryRecorder (opt-in)        │
                       │ 3. authorized log import   import --kind usage-log           │
 public prompt sets ───┤    (prompts only)          import --kind prompts → unlabeled  │
                       └────────────────────────────────┬───────────────────────────┘
                                                        ▼
  RAW        data/raw/<experiment>/events.jsonl   append-only, fsync'd, raw usage kept
             data/raw/imports/<batch>/            sanitized rows + rejection report
                                                        ▼  normalize (deterministic, idempotent)
  NORMALIZED data/normalized/{requests,preflight,observations,predictions}/
             date=YYYY-MM-DD/provider=<p>/part-<source>.parquet   (Arrow schema 1.0.0)
                                                        ▼  validate
             data/normalized/_validation/{decisions.parquet, quality_report.{md,json}}
                                                        ▼  prepare-training
  TRAINING   data/training/dataset_v<version>[_synthetic]/
             {train,validation,calibration,test}.parquet + manifest/schema/split_stats
                                                        ▼  export
  EXPORTS    data/exports/observations.csv + data_dictionary.md   (human-readable only)
```

## Design principles

1. **Observed usage is the only label.** Labels come from provider usage payloads
   (`usage`, `usage_metadata`). Preflight counts, predictions and estimates are
   stored in their own tables and are never written into `actual_*` columns.
2. **Provenance on every row.** `source_type` says where a row came from; only
   `controlled_experiment`, `consented_api_telemetry` and `authorized_log_import`
   can carry real labels. Mock-provider output is `test_fixture` with
   `label_quality=synthetic` and can only enter a training set built with
   `--label-origin synthetic`, written to a directory suffixed `_synthetic`.
   Production-derived real labels (telemetry, log imports) enter a training set
   only when included explicitly with `--include-source`.
3. **No leakage by construction.** Request records hold only fields known before
   dispatch; post-inference fields live in `observations`. `feature_registry.py`
   assigns every column a phase, and `prepare-training` refuses any
   `POST_INFERENCE` column as a feature.
4. **Provider semantics are not assumed to match.** Each provider has its own
   normalizer that follows its documented accounting (cached ⊂ input for OpenAI and
   Gemini; Anthropic input excludes cache reads/writes; Gemini output excludes
   thoughts). Reasoning and cached tokens are never added on top of a total that
   already contains them. Contradictions are flagged, not repaired.
5. **Safe by default.** No paid calls without `--execute-live` *and*
   `--max-cost-usd`; no prompt or response text stored unless the privacy config
   allows it and (for third-party data) consent is `granted`; no API keys in code.
6. **Reproducible.** Prompt generation, plan construction, sample IDs and splits
   are deterministic functions of config + seed. The normalized layer is a pure
   function of the raw layer, rebuilt per source atomically.

## Components

| Concern | Module | Notes |
|---|---|---|
| Canonical schema | `seshat.data.schemas` (`enums`, `request`, `records`, `arrow`) | Pydantic models + matching Arrow schemas. `LLMRequest` is the provider-neutral request. Field docs: [DATA_SCHEMA.md](DATA_SCHEMA.md) (generated, tested for drift). |
| Request features | `seshat.data.features.extract` | `extract_request_features()` computes request-time features only; `FEATURE_DEFINITIONS` documents each. Prompt text is hashed with HMAC (`SESHAT_PROMPT_HASH_KEY`) and never stored by default. |
| Prompt generation | `seshat.data.generators` | 17 templates (≥14 required categories), deterministic by seed, length strata from config, near-duplicate families merged so grouped splits stay leak-free. |
| Prompt bank / imports | `seshat.prompts` | License/consent policy (`configs/prompt_collection.json`), PII checks, dedup (exact + Jaccard shingles), review workflow. Imported prompts are `synthetic_unexecuted`/unlabeled. |
| Experiment design | `seshat.experiments.design` | Balanced stratified assignment of configurations to prompts (two-stage: target, then configuration), repeated cells for variance, capability filtering (`configs/model_catalog.json`), budget trim, imbalance report. Never a full Cartesian product. |
| Provider adapters | `seshat.providers` | `ProviderAdapter` interface: `list_supported_models`, `get_capabilities`, `count_input_tokens`, `execute_request`, `normalize_usage`. Live adapters use the official SDKs (`anthropic`, `openai`, `google-genai`) with SDK retries disabled; mocks emit realistic payloads parsed by the *same* parsers and normalizers. |
| Collector | `seshat.data.collectors.collector` | asyncio, global + per-provider semaphores, minimum-spacing rate limits per provider, exponential backoff with jitter, retry limits, timeouts, error classification (retryable / uncertain / fatal), budget reserve→settle, append-only event log, resume, dry-run. |
| Normalization | `seshat.data.collectors.normalize`, `seshat.providers.normalization` | Raw events → `RequestRecord`, `PreflightRecord`, `ObservationRecord`; cost from versioned price tables. |
| Telemetry | `seshat.data.collectors.telemetry` | Opt-in `TelemetryRecorder` (`start` before the call, `finish` after). Disabled by default. |
| Log import | `seshat.data.collectors.log_import` | Requires `--authorization-ref`; validation, deterministic dedup, redacted rejection report. |
| Storage | `seshat.data.storage` | Partitioned Parquet, atomic per-source replacement, retention purge. |
| Validation | `seshat.data.validators` | PASS / WARNING / QUARANTINE / REJECT per observation with reasons; quality report (Markdown + JSON). |
| Training prep | `seshat.data.training` | Point-in-time join, eligibility, union-find leakage groups (prompt family, prompt hash, conversation), category-stratified group split, optional temporal holdout, split statistics. |
| Pricing | `seshat.data.pricing`, `configs/pricing/*.json` | Dated official price tables; cost is never back-dated before a table's `effective_from`. |
| Exports / stats | `seshat.data.exports`, `seshat.data.stats` | CSV with plain columns + data dictionary; per provider/model statistics. |
| CLI | `seshat.data.cli` (`seshat-data`) | `generate`, `collect`, `normalize`, `import`, `validate`, `export`, `prepare-training`, `stats`, `predict`, `compare`, `purge-expired`, `prompts …`. |

## Collector state machine

Each planned sample has a deterministic `sample_id` (UUIDv5 of experiment,
prompt, configuration and repetition). The event log records:

```text
planned ──► dispatched ──► completed
                │      └─► failed (fatal, or retries exhausted)  ── --retry-failed
                └────────► uncertain (timeout / connection drop after send)
                                                              ── --retry-uncertain only
```

- `dispatched` is fsync'd **before** the request is sent, so a crash leaves
  evidence that a paid call may have happened.
- On resume, `completed` samples are skipped and `uncertain` samples are **not**
  re-sent automatically (that could double-bill); the operator decides with
  `--retry-uncertain`.
- Client-side errors raised before any HTTP request (bad arguments, missing
  credentials) are classified as *not sent*, so they never become uncertain.
- The budget reserves the worst-case cost of a request (`max_cost_estimate`:
  every input token at the most expensive applicable input rate, plus the full
  `max_output_tokens`, under the long-context tier if it could apply) before dispatch and
  settles to the observed cost afterwards; the run stops when the next reservation
  would exceed `--max-cost-usd`.
- Live adapters are instantiated before dispatch; a missing SDK or key aborts the
  run with exit code 2 instead of producing failures.

## Integration decisions

- **Reused**: the prompt bank and its review workflow, `privacy.py` (HMAC hashing,
  PII detection, redaction, retention), `provenance.py`, `feature_registry.py`,
  `split.py` (now generic over the grouping column), `csv_view.py` (formula
  neutralization), the baseline predictor (via `predict_from_word_count`) and the
  provider adapter package.
- **Replaced**: the earlier factory used JSON manifests, a sequential runner and a
  single "observed usage" record shape that could not express preflight counts,
  cache-write breakdowns, censoring or uncertain outcomes. Keeping both would have
  created two schemas for the same facts, so the old modules were removed and
  their tests rewritten against the new pipeline.
- **The online prediction service** (`seshat.api`, `seshat.predictor`) is unchanged
  apart from a small helper; the pipeline produces data for it but does not change
  its behaviour.

## What replaced what

| Before (PR #2/#5) | Now |
|---|---|
| `seshat.cli` (`seshat-data build-manifest/run/…`) | `seshat.data.cli` (`seshat-data generate/collect/…`) |
| `experiments/manifest.py`, `configs/experiments/*.json` | `experiments/design.py`, `experiments/config.py`, `configs/data_collection.yaml` |
| `experiments/runner.py` | `data/collectors/collector.py` (+ `events.py`, `control.py`) |
| `data/normalize.py` | `providers/normalization.py`, `providers/payloads.py`, `data/collectors/normalize.py` |
| `data/records.py`, `configs/dataset_schema.json` | `data/schemas/` (schema 1.0.0) |
| `data/store.py`, `data/ingest.py` | `data/storage/layers.py`, `data/collectors/log_import.py`, `telemetry.py` |
| `data/validate.py`, `data/profile.py` | `data/validators/` |
| `data/build_features.py`, `data/release.py` | `data/features/`, `data/training.py` |
| `data/synthetic.py`, `examples/` | mock providers (`providers/mock.py`) + `data/fixtures/offline_example/` |
| `data/compare.py` | `data/predictions.py` |
| `scripts/0*_*.py` | CLI subcommands |
| `data/records.csv` | `data/fixtures/offline_example/sample_export.csv` |

## Known limitations

- No live provider call has been made from this repository yet; live adapters are
  tested against recorded-shape fixtures and SDK request builders. Run
  `configs/experiments/live_pilot.yaml` with a small cap to validate them.
- Prices are copied from official pricing pages as of 2026-10-10 and must be
  re-checked before each live run; cost is an estimate (`price_table_estimate`),
  not an invoice.
- Mock-provider usage is a test fixture: its distributions are plausible but not
  real, and no model accuracy can be inferred from it.
- Native preflight counting differs by provider (Anthropic's `count_tokens` is an
  estimate; Gemini's `count_tokens` excludes system instruction/tools in this
  adapter) and is recorded as such (`preflight_complete`).
- The storage layer assumes a single writer per source; it is not a concurrent
  database.
