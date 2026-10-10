# Data collection guide

How to produce, validate and export Seshat training data. Architecture and design
decisions: [DATA_PIPELINE_ARCHITECTURE.md](DATA_PIPELINE_ARCHITECTURE.md). Every
field: [DATA_SCHEMA.md](DATA_SCHEMA.md).

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"            # offline pipeline + tests
pip install -e ".[dev,data,providers]"  # + official Anthropic / OpenAI / Gemini SDKs
cp .env.example .env                     # fill in locally; .env is gitignored
```

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic Messages API (live runs only) |
| `OPENAI_API_KEY` | OpenAI Responses API (live runs only) |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Gemini `generate_content` (live runs only) |
| `SESHAT_PROMPT_HASH_KEY` | HMAC key for prompt hashes. Set it for any real data; without it hashes are unkeyed SHA-256. |
| `SESHAT_LIVE_TESTS=1` | Enables the three gated live adapter tests (each also needs its key). |

Keys are read by the SDKs from the environment. None are needed for the offline
workflow.

## 2. Offline end-to-end run (no API spend)

`configs/data_collection.yaml` runs 120 generated prompts against three mock
providers that emit realistic OpenAI, Anthropic and Gemini usage payloads.

```bash
seshat-data generate --count 120 --seed 42                 # optional: inspect the prompts
seshat-data collect --config configs/data_collection.yaml --dry-run
seshat-data collect --config configs/data_collection.yaml  # mocks only: runs offline
seshat-data validate
seshat-data export --format csv
seshat-data prepare-training --version 0.1.0 --label-origin synthetic
seshat-data stats
```

Mock output is labelled `source_type=test_fixture`, `label_quality=synthetic`.
`prepare-training` without `--label-origin synthetic` refuses it, by design. The
committed result of this run (25-row CSV sample, data dictionary, quality report,
design report, split and collection statistics) is in
[`data/fixtures/offline_example/`](../data/fixtures/offline_example/).

## 3. Data sources

| Method | Command | `source_type` | Labels |
|---|---|---|---|
| Controlled experiments | `collect` | `controlled_experiment` (live) / `test_fixture` (mock) | real / synthetic |
| Consented API telemetry | `TelemetryRecorder` (Python) | `consented_api_telemetry` | real |
| Authorized usage-log import | `import --kind usage-log` | `authorized_log_import` | real |
| Public / licensed prompt datasets | `import --kind prompts` | `public_permitted_dataset` | **none** until executed |
| Generated prompts | `generate` | `synthetic_unexecuted` | **none** until executed |

Not supported, deliberately: scraping chatbot conversations, browser sessions or
any data without a license or authorization trail.

### 3.1 Controlled experiments

Edit or copy `configs/data_collection.yaml`:

- `prompts`: generated count/seed and/or `bank_files`. Live runs only use prompts
  whose `review_status` is `approved` (generated template prompts are approved by
  code review; imported prompts must be reviewed with `seshat-data prompts review`).
- `strata`: word-count thresholds for short / medium / long prompts.
- `targets`: provider, model and the levels of each factor (effort, thinking mode,
  thinking budget, max output tokens, caching, verbosity, temperature, structured
  output). `null` means "leave unset". Combinations a model does not support (per
  `configs/model_catalog.json`) are excluded and counted in the design report.
- `sampling.configs_per_prompt`, `repetitions` (fraction of cells repeated and how
  often, to measure output variance), `budget`, `execution`, `privacy`, `storage`.

Always start with `--dry-run`: it writes the plan and design report
(`data/raw/<experiment>/design_report.json`, including imbalance statistics) and
prints a worst-case cost estimate without sending anything.

### 3.2 Live runs (paid)

```bash
seshat-data collect --config configs/experiments/live_pilot.yaml --dry-run
seshat-data collect --config configs/experiments/live_pilot.yaml --execute-live --max-cost-usd 5
```

- Both flags are required. The config's `budget.max_cost_usd` is an additional cap;
  the lower one wins.
- Before running, check `configs/pricing/<date>.json` against the providers'
  pricing pages and add a new dated file if prices changed. Costs are never
  computed with a table whose `effective_from` is later than the request.
- Each live run records the SDK version, resolved model, provider request ID and
  the full raw usage payload.

### 3.3 Resuming, retries and uncertain samples

Re-running the same `collect` command resumes from `data/raw/<experiment>/events.jsonl`:
completed samples are skipped. Retryable errors (429, 5xx, overload) are retried
with backoff up to `execution.max_retries`. Samples whose outcome is unknown
(timeout or dropped connection after sending) are marked `uncertain` and are not
re-sent unless you pass `--retry-uncertain`, because the provider may already
have billed them. `--retry-failed` re-attempts samples that failed permanently.

Budget, execution and privacy settings may change between resumed runs. If the
*design* changes (prompts, targets, sampling, repetitions, strata or seed), the
collector refuses to resume; use a new `experiment_id`.

### 3.4 Consented telemetry

```python
from seshat.data.collectors.telemetry import TelemetryRecorder
from seshat.data.schemas.enums import ConsentStatus

recorder = TelemetryRecorder(root="data", enabled=True)  # disabled by default
handle = recorder.start(request, request_id="r1", consent_status=ConsentStatus.GRANTED)
response = client.messages.create(...)
recorder.finish(handle, response.model_dump(mode="json"), latency_ms=820.0)
recorder.flush()
```

Features are captured before the call and usage after it. Prompt text is kept only
if `capture_prompt_text=True` and consent is `granted`. Telemetry rows get a
`retain_until` date from the retention policy; delete expired rows with
`seshat-data purge-expired`.

### 3.5 Authorized usage-log import

```bash
seshat-data import --kind usage-log --file exports/usage.jsonl \
  --authorization-ref "DPA-2026-014 / approved by data-owner@example.com" \
  --dataset-origin "support-bot-prod"
```

One JSON object per line: `request` (an `LLMRequest`), `response` (the provider's
response object including usage), `timestamp_utc`, and optional `latency_ms`,
`conversation_id`, `parent_request_id`, `prompt_category`, `consent_status`. The
format is documented in `seshat/data/collectors/log_import.py`. Rows are validated
and deduplicated by provider response ID; rejected rows are written, redacted, to
`data/raw/imports/<batch>/rejections.jsonl`.

### 3.6 Public prompt datasets

```bash
seshat-data import --kind prompts --file some_dataset.jsonl --source "<name>" \
  --source-detail "<url / attribution>" --license-id CC-BY-4.0 --commercial-use true
seshat-data prompts check
seshat-data prompts review --ids <prompt ids> --status approved --reviewer <you>
```

Licenses must be allowed in `configs/prompt_collection.json`. Imported prompts
carry no usage labels; they become labelled data only after being executed in a
controlled experiment.

## 4. Validation, training data and exports

- `seshat-data validate` writes `data/normalized/_validation/decisions.parquet`
  and `quality_report.{md,json}`. Nothing is deleted or edited; each observation
  gets a decision and flags (full list in `seshat/data/validators/checks.py`):
  **PASS**; **WARNING** — usable but flagged (censored at the output limit,
  preflight/actual discrepancy above tolerance, unexpected model routing, the same
  prompt in several families); **QUARANTINE** — kept for inspection, excluded from
  training (contradictory token or cost accounting, failed / uncertain / no usage,
  timestamps out of order); **REJECT** — structurally invalid (orphan or duplicate
  observation, provider/model or provenance mismatch).
- `seshat-data prepare-training --version X` builds
  `data/training/dataset_vX/` from real labels only, and by default only from
  controlled experiments: production data (telemetry, log imports) never becomes
  training data automatically. Add it with `--include-source consented_api_telemetry`
  and/or `--include-source authorized_log_import`. Other options:
  `--temporal-holdout 0.2` (newest groups become test), `--group-by-template`,
  `--seed`. Repeats, prompt families, identical prompts and conversations always
  land in the same split. `split_stats.json` reports sizes and distributions.
- `seshat-data export --format csv` writes `data/exports/observations.csv` and
  `data_dictionary.md`. CSV is for reading; Parquet remains canonical.

## 5. Where data lives

| Path | Contents | In Git? |
|---|---|---|
| `data/raw/` | event logs, plans, design reports, import batches | no |
| `data/normalized/` | canonical Parquet tables + validation output | no |
| `data/training/` | versioned training datasets | no |
| `data/exports/` | CSV exports | no |
| `data/fixtures/offline_example/` | small mock-only example outputs | yes |
| `configs/` | collection, pricing and model catalog configs | yes |
| `prompts/` | reviewed prompt bank | yes |

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Output says `ran as a dry run` | The plan has live providers but `--execute-live` was not passed; nothing was sent. |
| `error: live collection requires --max-cost-usd` | Live runs need an explicit cap. |
| `error: … different design …` | The config's design changed under an existing `experiment_id`; pick a new one. |
| Run aborts before dispatch with an SDK/key error | Install `.[providers]` and export the provider key. Nothing was sent. |
| Many `excluded_combinations` in the design report | A factor level is unsupported for that model in `configs/model_catalog.json`. |
| `stopped_reason: max_cost_reached` / `max_requests_reached` | The next request's worst-case reservation would exceed the cap. Raise the cap or lower `max_output_tokens`. |
| `stopped_reason: cost_unbounded` | A cost cap is set but no price is known for a target model. Add it to `configs/pricing/`. |
| `cost_usd` empty | No price table covers the model or date. Add one under `configs/pricing/`. |
| Rows quarantined as `inconsistent` | Provider usage fields contradict each other; inspect `raw_usage` and `billing_dimensions.accounting_issues`. |
| `No eligible rows with real labels` | Only mock data exists. Use `--label-origin synthetic` for pipeline tests. |
| Samples stuck in `uncertain` | Check provider dashboards for the request, then decide on `--retry-uncertain`. |
