# Dataset Creation Workflow

> **Status (2026-10-10):** this is the original design rationale. The implemented
> pipeline is described in [DATA_PIPELINE_ARCHITECTURE.md](DATA_PIPELINE_ARCHITECTURE.md),
> [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) and [DATA_SCHEMA.md](DATA_SCHEMA.md).
> Where file names, commands or schemas below differ from those documents (for
> example `build-manifest`/`run` → `collect`, JSON manifests → `configs/data_collection.yaml`),
> the newer documents are authoritative. The principles in section 1 still apply.

This document is the canonical implementation guide for creating Seshat's training and evaluation datasets.

The objective is to build the **smallest scientifically valid data factory** that can turn controlled LLM requests into reproducible, versioned ML datasets without manually editing rows.

## 1. Principles

1. **Raw observations are immutable.** Never edit provider responses in place.
2. **Prediction features must exist before inference.** Post-response fields may be targets or evaluation metadata, never prediction-time inputs.
3. **Provider-native usage is preserved.** Normalize common fields, but keep the original provider payload.
4. **Every experiment is reproducible.** Prompts, configuration, code version, model identifier, and run IDs must be recorded.
5. **Repeated runs stay grouped during splitting.** Near-duplicate conditions must not leak across train/test.
6. **Dataset quality matters more than dataset size.**
7. **Production observations do not automatically become training data.**
8. **Protected evaluation data stays protected.** Do not tune against the final test set.

## 2. v0 Tooling

Use the existing Seshat Python project as the base.

| Need | v0 choice | Notes |
|---|---|---|
| IDE | Cursor or VS Code | AI-assisted coding is fine; human review remains required |
| Language | Python 3.11+ | Matches the current project |
| Schemas | Pydantic | Already a project dependency |
| DataFrames | pandas initially | Already a project dependency; Polars can be evaluated later |
| Raw storage | JSONL | Appendable and preserves nested provider data |
| ML dataset storage | Parquet | Typed, compressed, efficient for analysis/training |
| Local analytics | DuckDB | Query Parquet directly; add when dataset profiling begins |
| Config | YAML or JSON | Human-readable experiment definitions |
| Tests | pytest | Already configured |
| Version control | Git/GitHub | Commit schemas/configs/code, not large raw datasets |
| Secrets | environment variables / .env locally | Never commit provider API keys |

Likely dependencies to add when implementation begins:

- provider SDK for the first target provider
- `pyarrow` for Parquet support
- `duckdb` for local analytical queries
- `pyyaml` if experiment configs use YAML
- optionally `typer` for a cleaner CLI

Do not introduce PostgreSQL, Kafka, Spark, a warehouse, or Kubernetes for the controlled-experiment phase.

## 3. Target Repository Layout

The current repository remains the root. Add dataset-specific modules incrementally:

```text
seshat/
├── configs/
│   ├── dataset_schema.yaml
│   ├── providers.yaml
│   ├── models.yaml
│   └── experiments/
│       ├── pilot_001.yaml
│       └── pilot_002.yaml
│
├── prompts/
│   ├── hand_authored.jsonl
│   ├── coding.jsonl
│   ├── reasoning.jsonl
│   ├── summarization.jsonl
│   └── imported/
│
├── src/seshat/
│   ├── schemas.py
│   ├── features.py
│   ├── data/
│   │   ├── models.py
│   │   ├── normalize.py
│   │   ├── validate.py
│   │   ├── build_features.py
│   │   ├── split.py
│   │   └── release.py
│   ├── experiments/
│   │   ├── manifest.py
│   │   └── runner.py
│   └── providers/
│       ├── base.py
│       └── <provider>.py
│
├── scripts/
│   ├── 01_import_prompts.py
│   ├── 02_build_manifest.py
│   ├── 03_run_experiment.py
│   ├── 04_normalize.py
│   ├── 05_validate.py
│   ├── 06_build_features.py
│   ├── 07_split_dataset.py
│   ├── 08_profile_dataset.py
│   └── 09_freeze_dataset.py
│
├── data/
│   ├── raw/
│   ├── manifests/
│   ├── normalized/
│   ├── features/
│   ├── splits/
│   ├── releases/
│   └── quarantine/
│
├── notebooks/
│   ├── 01_dataset_profile.ipynb
│   └── 02_variance_analysis.ipynb
│
└── tests/
    ├── test_data_schemas.py
    ├── test_normalization.py
    ├── test_data_validation.py
    ├── test_feature_roles.py
    └── test_splits.py
```

Large/raw generated data directories remain gitignored. Commit schemas, configs, small synthetic fixtures, and dataset manifests.

## 4. End-to-End Data Flow

```text
Human / public / synthetic prompts
              |
              v
       Prompt bank (JSONL)
              |
              v
      Experiment config
              |
              v
       Manifest builder
              |
              v
       Manifest (Parquet)
              |
              v
      Experiment runner
              |
              v
         Provider API
              |
              v
       Raw observations
       (immutable JSONL)
              |
              v
         Normalization
              |
              v
   Normalized observations
          (Parquet)
              |
              v
      Quality validation
          /       \
       PASS     QUARANTINE
        |
        v
    Feature pipeline
        |
        v
    Features (Parquet)
        |
        v
     Split assignment
    /       |       |       \
 train  validation  test  calibration
    \       |       |       /
              v
       Versioned release
              |
              v
       Model development
```

## 5. Stage 1 — Build the Prompt Bank

Prompt files use JSONL: one JSON object per line.

Example:

```json
{"prompt_id":"p_000001","category":"coding","text":"Write a Python function that merges overlapping intervals.","source":"hand_authored","language":"en","complexity":"medium"}
{"prompt_id":"p_000002","category":"summarization","text":"Summarize the following passage in three bullet points: ...","source":"hand_authored","language":"en","complexity":"low"}
```

Minimum prompt fields:

- `prompt_id` — permanent stable identifier
- `text` or structured messages
- `category`
- `source`
- `language`
- `complexity`
- provenance/license metadata when imported

Do not use row numbers as IDs.

### Initial corpus

Start with approximately **200–500 reviewed prompts**, not tens of thousands.

Cover roughly:

- coding/debugging/refactoring
- summarization
- reasoning
- mathematics
- translation
- extraction
- business/general writing
- structured output
- research synthesis
- agentic/tool-oriented requests

Vary prompt length, requested output length, complexity, format, and conversation depth.

Synthetic prompts may fill coverage gaps but must be marked as synthetic and should not dominate the corpus.

## 6. Stage 2 — Define the Experiment

Experiment files describe how observations will be generated.

Example:

```yaml
experiment_id: pilot_001

prompt_sources:
  - prompts/hand_authored.jsonl

categories:
  - coding
  - summarization
  - reasoning

providers:
  first_provider:
    models:
      - model_a
      - model_b

settings:
  effort:
    - low
    - high

repetitions: 2
max_requests: 400
```

An experiment configuration is a **recipe**, not an observation.

Do not automatically execute every possible Cartesian product. Sampling should control cost while maintaining useful coverage.

## 7. Stage 3 — Build an Explicit Manifest

`02_build_manifest.py` converts the experiment config into explicit planned runs.

Canonical file:

```text
data/manifests/pilot_001.parquet
```

Example columns:

| run_id | prompt_id | provider | model | effort | repetition |
|---|---|---|---|---|---:|
| r_000001 | p_000001 | provider_a | model_a | low | 1 |
| r_000002 | p_000001 | provider_a | model_a | low | 2 |
| r_000003 | p_000001 | provider_a | model_a | high | 1 |

The manifest makes experiment cost, coverage, retries, and reproducibility inspectable **before** API calls are made.

A CSV preview may be generated for humans, but Parquet is canonical.

## 8. Stage 4 — Execute Controlled Runs

`03_run_experiment.py` reads manifest rows and:

1. loads the prompt/configuration;
2. validates the request;
3. computes request-time metadata;
4. calls the provider;
5. captures provider-native usage metadata;
6. records timestamps and request IDs;
7. persists an immutable raw observation;
8. records failures without destroying completed work.

Conceptual loop:

```python
for run in manifest:
    request = build_request(run)
    started_at = now()

    try:
        response = provider.execute(request)
        observation = build_success_observation(
            run=run,
            request=request,
            response=response,
            started_at=started_at,
        )
    except Exception as exc:
        observation = build_failure_observation(
            run=run,
            error=exc,
            started_at=started_at,
        )

    append_raw_observation(observation)
```

The runner **does not perform ML feature engineering**. It records reality faithfully.

### Retry rules

- Rate-limit / transient network error: retry with bounded exponential backoff.
- Deterministic request validation failure: do not retry.
- Provider refusal/safety response: record as an observation with an explicit status; do not silently discard.
- Unknown/malformed usage metadata: preserve the raw response and quarantine downstream.
- Every retry receives trace metadata so accidental duplicate observations can be detected.

## 9. Stage 5 — Raw Observation Format

Raw experiment output uses JSONL:

```text
data/raw/<experiment_id>/<provider>.jsonl
```

Representative record:

```json
{
  "observation_id": "obs_000421",
  "run_id": "r_000421",
  "experiment_id": "pilot_001",
  "prompt_id": "p_000053",
  "timestamp": "2026-10-05T02:11:12Z",
  "provider": "provider_a",
  "model": "model_a",
  "effort": "high",
  "repetition": 2,
  "request_metadata": {
    "message_count": 1,
    "tools_enabled": false
  },
  "provider_usage_raw": {},
  "normalized_usage": {
    "input_tokens": 713,
    "output_tokens": 1821,
    "reasoning_tokens": null,
    "cached_tokens": null,
    "total_tokens": 2534
  },
  "provider_request_id": "...",
  "status": "success"
}
```

### Immutability rule

Nothing edits files in `data/raw/`.

If normalization or parsing is wrong, fix the processing code and rebuild downstream artifacts.

## 10. Stage 6 — Normalize Provider Data

`04_normalize.py` transforms raw provider-specific records into typed rectangular records:

```text
data/normalized/<experiment_id>.parquet
```

Core normalized fields:

- observation_id
- run_id
- prompt_id
- experiment_id
- timestamp
- provider
- model
- provider model/version identifier where available
- effort/configuration
- repetition
- provider-reported input tokens
- output tokens
- reasoning/thinking tokens where applicable
- cached tokens where applicable
- total/billable usage where applicable
- finish reason
- latency
- provider request ID
- status

Preserve `provider_usage_raw` separately. Do not force semantically different provider fields into false equivalence.

For Dataset v0.1, the likely primary target is:

```text
output_tokens
```

Retain the other usage measures for future analysis.

## 11. Stage 7 — Data Quality Validation

`05_validate.py` applies explicit rules.

Check for:

- missing IDs
- duplicate run IDs
- duplicate provider request IDs
- missing provider/model
- negative token counts
- impossible totals
- missing usage metadata
- failed requests
- timeouts
- truncation
- refusal/safety outcomes
- unknown models
- malformed provider metadata
- schema violations

Each observation receives one of:

- `PASS`
- `WARNING`
- `QUARANTINE`
- `REJECT`

Quarantined records are written to:

```text
data/quarantine/
```

Never silently delete anomalous observations.

## 12. Stage 8 — Build Prediction-Time Features

`06_build_features.py` combines prompt metadata, experiment metadata, and normalized observations.

Example feature dataset:

```text
data/features/features_v001.parquet
```

Candidate v0 features:

- provider
- model
- effort/reasoning setting
- input token count
- input character count
- input word count
- message count
- conversation depth
- task category
- prompt complexity bucket
- requested output format
- requested output length signals
- tools enabled
- tool count

Target:

- `output_tokens`

### Leakage invariant

For every feature ask:

> Could Seshat know this value at the exact moment prediction is requested?

If no, it cannot be a prediction-time feature.

Never allow fields such as:

- actual output tokens
- actual response characters
- actual latency
- actual tool calls
- actual finish reason

into `X`.

## 13. Explicit Column Roles

Maintain a version-controlled schema such as `configs/dataset_schema.yaml`:

```yaml
features:
  - input_tokens
  - input_characters
  - input_words
  - provider
  - model
  - effort
  - task_category
  - complexity
  - requested_output_format

targets:
  - output_tokens

post_inference_only:
  - actual_response_characters
  - latency_ms
  - finish_reason
  - reasoning_tokens
  - total_tokens

identifiers:
  - observation_id
  - prompt_id
  - experiment_id

forbidden_features:
  - output_tokens
  - actual_response_characters
  - latency_ms
  - finish_reason
```

Add a test that fails if any `forbidden_features` column enters the model feature matrix.

## 14. Stage 9 — Assign Train/Validation/Test/Calibration Splits

`07_split_dataset.py` creates:

```text
data/splits/split_assignments_v001.parquet
```

Example:

| observation_id | split |
|---|---|
| obs_001 | train |
| obs_002 | train |
| obs_003 | validation |
| obs_004 | test |
| obs_005 | calibration |

### Split rules

- Repeated runs from the same underlying prompt/configuration stay in the same split.
- Near-duplicate prompt families stay together when possible.
- Do not manually move rows after seeing model performance.
- Prefer time-aware evaluation once data spans provider/model changes.
- Keep the protected test set out of iterative feature/model tuning.
- Use a separate calibration set when conformal intervals are introduced.

The exact v0 split strategy should be documented in the dataset release manifest.

## 15. Stage 10 — Profile the Dataset

`08_profile_dataset.py` should produce coverage and quality summaries before any model is trained.

At minimum inspect:

- observations by provider/model
- observations by task category
- effort/configuration coverage
- prompt length distribution
- output-token distribution
- repeated-run counts
- invalid/quarantine rates
- missing fields
- duplicate rates
- source distribution
- category imbalance

DuckDB can query Parquet directly for ad-hoc analysis.

Example:

```sql
SELECT
    model,
    effort,
    COUNT(*) AS n,
    AVG(output_tokens) AS avg_output_tokens
FROM 'data/features/features_v001.parquet'
GROUP BY model, effort;
```

## 16. Stage 11 — Freeze a Dataset Release

`09_freeze_dataset.py` creates an immutable release:

```text
data/releases/
└── dataset_v0.1/
    ├── train.parquet
    ├── validation.parquet
    ├── test.parquet
    ├── calibration.parquet
    ├── manifest.json
    ├── schema.json
    └── README.md
```

Example `manifest.json`:

```json
{
  "dataset_version": "0.1.0",
  "created_at": "...",
  "observation_count": 4832,
  "prompt_count": 612,
  "providers": ["provider_a"],
  "models": ["model_a", "model_b"],
  "target": "output_tokens",
  "feature_pipeline_version": "0.1.0",
  "split_strategy": "grouped",
  "source_experiments": ["pilot_001", "pilot_002"]
}
```

Every trained model must be traceable to:

```text
dataset version
+ feature pipeline version
+ training code version
+ model configuration
```

## 17. Dataset Storage Policy

### Commit to Git

- schemas
- prompt metadata that is safe/licensed to commit
- experiment configs
- small synthetic fixtures
- scripts
- tests
- release manifests
- summary statistics

### Do not commit directly

- secrets
- raw customer prompts
- large raw experiment outputs
- generated Parquet datasets
- provider response bodies containing sensitive material

For the controlled phase, local/object-storage files are sufficient.

PostgreSQL becomes useful later when the SDK/API begins ingesting live production observations transactionally.

## 18. Pilot 0 — Validate the Data Factory

Before generating a serious training dataset, run a small engineering pilot.

Target:

```text
~50 prompts
x 2 model/configurations
x 2 effort/settings where applicable
x 2 repetitions
≈ 400 requests
```

The goal is **not model accuracy**.

The goal is to prove:

- requests execute reliably;
- provider usage parses correctly;
- retries do not silently duplicate records;
- raw observations are immutable;
- normalization is deterministic;
- leakage checks work;
- quality rules behave correctly;
- split generation is reproducible;
- a versioned release can be produced without manual row editing.

Fix pipeline problems before scaling.

## 19. Dataset v0.1

After Pilot 0 succeeds, target approximately:

- 500 unique reviewed prompts
- 8–10 task categories
- 2–4 initial model/configuration combinations
- selective repetitions on ~10–20% of conditions
- approximately 3,000–8,000 total observations

Do **not** execute the full Cartesian product unless analysis shows it is useful.

Dataset v0.1 exists to answer:

1. How strongly does input length predict output usage?
2. How much predictive value comes from model/provider/effort?
3. How much does task category matter?
4. What is the within-condition stochastic variance?
5. Which segments have materially larger errors/variance?
6. Is there enough predictable structure to justify Seshat?
7. What additional data should Dataset v0.2 collect?

## 20. Start With One Provider

The first implementation should support only one provider end-to-end:

```text
prompt
  -> provider adapter
  -> API request
  -> provider usage
  -> raw JSONL
  -> normalized Parquet
  -> validated feature dataset
  -> versioned release
```

Once this path is reliable, extract the provider interface and add additional adapters.

This keeps early failures attributable and prevents three integrations from obscuring basic pipeline bugs.

## 21. Repeated-Run Protocol

Use repeated identical conditions to estimate stochastic output variance.

For selected prompt/model/configuration combinations:

- run the identical visible request multiple times;
- retain every observation independently;
- compute mean, median, variance, standard deviation, quantiles, and range;
- group repeats together during train/test splitting.

Repeated runs help separate:

- model prediction error
- from irreducible variation in LLM output behavior.

## 22. Recommended Commands

During early implementation the scripts may be invoked directly:

```bash
python scripts/02_build_manifest.py pilot_001
python scripts/03_run_experiment.py pilot_001
python scripts/04_normalize.py pilot_001
python scripts/05_validate.py pilot_001
python scripts/06_build_features.py
python scripts/07_split_dataset.py
python scripts/08_profile_dataset.py
python scripts/09_freeze_dataset.py --version 0.1.0
```

A later CLI may expose:

```bash
seshat experiment build pilot_001
seshat experiment run pilot_001
seshat dataset validate
seshat dataset build
seshat dataset release 0.1.0
```

## 23. Human Review vs AI Delegation

AI coding agents can implement much of the mechanical code, including:

- Pydantic model boilerplate
- JSONL/Parquet IO
- provider adapter scaffolding
- CLI commands
- retries
- validation plumbing
- tests
- profiling queries

Human technical review is required for:

- canonical observation schema
- target definition
- feature whitelist
- leakage rules
- experimental sampling
- split methodology
- data-quality policy
- privacy/retention rules
- benchmark protection

Do not accept an AI-generated training pipeline merely because it runs.

## 24. First Implementation Tickets

Execute in this order:

1. **DATA-001 — Canonical schemas**  
   Define `Prompt`, `ExperimentRun`, `RawObservation`, and `NormalizedObservation` using Pydantic.

2. **DATA-002 — Prompt-bank loader**  
   Read/version JSONL prompt sources and validate stable IDs/provenance.

3. **DATA-003 — Experiment manifest builder**  
   Convert a config into deterministic planned runs and cost/coverage summary.

4. **DATA-004 — Provider interface**  
   Define one provider-neutral execution contract without erasing provider-specific metadata.

5. **DATA-005 — First provider adapter**  
   Execute one request and capture native usage.

6. **DATA-006 — Immutable raw writer**  
   Append success/failure observations safely to JSONL.

7. **DATA-007 — Normalizer**  
   Convert raw observations to typed Parquet.

8. **DATA-008 — Data-quality validator**  
   Implement PASS/WARNING/QUARANTINE/REJECT rules.

9. **DATA-009 — Feature builder + leakage guard**  
   Build request-time features and fail if forbidden fields enter `X`.

10. **DATA-010 — Split + release builder**  
    Group repeats/prompt families, create split assignments, and freeze Dataset v0.0.1.

## 25. Definition of the First Milestone

The Seshat data factory exists when the project can:

> Take ~50 prompts, execute ~400 controlled provider calls, and reproducibly produce versioned `train.parquet`, `validation.parquet`, `test.parquet`, and `calibration.parquet` files without manually editing any observation.

Only after that milestone should the team spend meaningful effort comparing ML predictors.
