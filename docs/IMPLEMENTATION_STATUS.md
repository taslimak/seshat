# Seshat — Implementation Status

| | |
|---|---|
| Audit date | 2026-10-10 |
| `main` revision audited | `454451d` (merge of PR #1, `feature/demo-ui`) |
| Open branch audited | `feature/dataset-factory-v0` @ `ce9aec7` (PR #2, open, CI failing) |
| Scope | Every tracked file on `main` and every file changed on PR #2 |
| Production code changed by this audit | None |

Every claim below was checked against the code or a command run during the audit. Where something was checked by running code, the command or reproduction is noted. "Planned" means it is described in a doc but has no code.

---

## 1. Executive summary

1. **`main` is a working walking skeleton.** It has a FastAPI service (`/health`, `/v1/predict`, demo UI at `/`), a hand-tuned heuristic predictor, a deterministic prompt-feature extractor, and 4 tests. All 4 tests pass, ruff is clean, `mypy --strict` is clean, and line coverage is 100%. The predictor is not trained on anything. It ignores the `model` field and returns a constant `confidence` of 0.35.
2. **The dataset factory exists only on PR #2, and that branch is broken.** `src/seshat/data/io.py` has two `"\n"` literals committed as real line breaks, which is a `SyntaxError`. That stops 3 of 7 new test modules from importing, and GitHub CI on PR #2 failed at the Lint step.
3. **PR #2 has deeper bugs behind the syntax error.** With the syntax error patched in a scratch copy, the branch's own CI smoke pipeline still quarantines **144 of 144** mock observations, writes **0** training rows, and then crashes in `split`. The seed corpus also produces empty validation and calibration splits, and successful retries get rejected.
4. **Nothing on the cost side is built yet.** There is no pricing, cost estimation, input/reasoning-token prediction, feedback endpoint, database, model training, interval calibration, auth, or SDK.
5. **Critical path:** fix and merge PR #2 → run Pilot 0 → build a trained baseline with calibrated intervals → serve it from the API → add pricing → add the feedback loop. See §8.

---

## 2. Test and check results (actually executed)

Environment: Python 3.13.16 in a fresh venv. CI uses 3.11, so 3.11 was **not** exercised locally. Resolved versions: fastapi 0.143.0, pydantic 2.14.0, pandas 2.3.3, pyarrow 25.0.1, pytest 8.4.2, ruff 0.17.0, mypy 1.20.2.

### 2.1 `main` @ `454451d` — `pip install -e ".[dev]"`

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -v` | **4 passed**, 0 failed, 1 warning (`StarletteDeprecationWarning`: `httpx` with `starlette.testclient` is deprecated) |
| Coverage | `pytest --cov=seshat` | **100%** (71/71 statements) |
| Lint | `ruff check .` | **All checks passed** |
| Types | `mypy src` (strict, from `pyproject.toml`) | **Success: no issues in 5 files**. Note that CI does **not** run mypy. |
| GitHub CI | Actions history | All `main` push runs and the PR #1 run: success |

### 2.2 PR #2 `feature/dataset-factory-v0` @ `ce9aec7` — `pip install -e ".[dev,data]"`

| Check | Result as committed |
|---|---|
| Tests | **Collection interrupted: 3 errors** (`test_manifest.py`, `test_runner.py`, `test_splits.py`), all `SyntaxError: unterminated string literal` at `src/seshat/data/io.py:47` |
| Lint | **22 errors**: 10 `invalid-syntax` (io.py lines 47 and 58), 8 `I001` unsorted imports (`scripts/*.py`), 3 `UP035`, 1 `UP037`. 12 can be auto-fixed. |
| Types | **mypy aborts** on the syntax error |
| GitHub CI (run 2026-10-05) | **failure** at `Lint`. `Test` and `Data factory smoke test` were **skipped**. |
| `[providers]` extra | Resolves (dry run: openai 3.28.0, anthropic 1.13.0). Not installed or exercised, and no live provider calls were made. |

**Diagnostic run in a scratch copy only.** I replaced the two broken literals with `"\\n"` and changed nothing else. This was not committed.

| Check | Result |
|---|---|
| Tests | **15 passed** |
| Coverage | 30% when collection fails. Not re-measured after patching. `cli.py`, `normalize.py`, `release.py`, `profile.py`, `build_features.py` and the OpenAI/Anthropic adapters have no direct unit tests. |
| mypy | **15 errors**: 11× missing `pandas-stubs`, 2× `openai`/`anthropic` not installed, 3 real typing errors (`models.py:42`, `manifest.py:56`, `manifest.py:100`) |
| CI smoke pipeline (the exact `ci.yml` sequence) | `build-manifest` 144 runs ✓ → `run` 144 obs ✓ → `normalize` ✓ → `validate`: **QUARANTINE 144** → `features`: **0 eligible rows** → `split`: **`KeyError: 'prompt_family_id'`** ✗ |

So PR #2 would still fail CI after the syntax fix, because the smoke test would crash.

---

## 3. Architecture map

### 3.1 On `main`

```text
Browser (src/seshat/static/index.html)
   │  POST /v1/predict  {prompt, model, effort}
   ▼
api.py  (FastAPI app, module-level BaselinePredictor singleton)
   │  validates with schemas.PredictionRequest
   ▼
predictor.BaselinePredictor.predict(prompt, model, effort)
   │  model is accepted but unused
   ▼
features.extract_prompt_features(prompt) -> PromptFeatures (9 numeric features)
   │
   ▼
estimate = max(32, round((48 + 0.55·word_count) · effort_multiplier))
interval = ±max(24, 0.65·estimate);  confidence = 0.35 (constant)
```

### 3.2 Added by PR #2 (unmerged)

```text
prompts/*.jsonl ─┐
configs/experiments/*.json ─► experiments.manifest ─► data/manifests/*.parquet
                                                         │
                     providers.{openai,anthropic,mock} ◄─ experiments.runner ──► data/raw/*.jsonl (append-only)
                                                         │   (reuses seshat.features ✓)
data.normalize ─► data/normalized/*.parquet ─► data.validate (PASS/WARNING/QUARANTINE/REJECT)
   ─► data.build_features (+ feature_registry leakage guard) ─► data.split (family-grouped, category-stratified)
   ─► data.release (train/validation/calibration/test parquet + sha256 manifest) ; data.profile ; data.csv_view
Entry points: `seshat-data` CLI (cli.py) and scripts/02–09 thin wrappers
```

**Nothing connects the two halves.** The predictor never reads a release or a model artifact, and no training module exists.

---

## 4. Verified inventory

Status key: ✅ works and is tested · 🟡 partial or has defects · 🔴 broken · ⬜ planned only (doc, no code)

### 4.1 API endpoints (`main`, `src/seshat/api.py`)

| Route | Status | Notes |
|---|---|---|
| `GET /` | ✅ | Serves the demo UI. Excluded from OpenAPI. |
| `GET /health` | ✅ | Static `{"status":"ok"}`. Doesn't check model or artifact readiness. |
| `POST /v1/predict` | 🟡 | Contract works and is validated: empty prompt → 422, `effort="xhigh"` → 422. **No prompt size limit**: a 2,000,000-character prompt returned 200. No auth, rate limit, cost, or input/reasoning-token fields. |
| `/docs`, `/redoc`, `/openapi.json` | ✅ | FastAPI defaults |
| Observation / feedback ingestion | ⬜ | Not designed |

### 4.2 Prediction and ML components

| Component | Location | Status | Notes |
|---|---|---|---|
| Prompt feature extractor | `features.py` | ✅ | 9 deterministic features, used by both the API and the PR #2 runner (no duplication) |
| Baseline predictor | `predictor.py` | 🟡 | Heuristic. **`model` is ignored** (verified: same output for different models). `auto` == `medium`. Interval is ±65% and not calibrated. `confidence=0.35` is a constant, not a coverage level. |
| Pre-request token counting | — | ⬜ | `tiktoken` is a dependency but **never imported** |
| Trained model / training code | — | ⬜ | `scikit-learn`, `joblib`, `numpy` are declared but **never imported** |
| Model artifact loading | — | ⬜ | `SESHAT_MODEL_PATH` (`.env.example`) is read by nothing |
| Interval calibration (conformal) | — | ⬜ | PR #2 does reserve a `calibration` split |
| Evaluation metrics | — | ⬜ | |
| Input / reasoning / total token prediction | — | ⬜ | Only output tokens are predicted |
| Pricing / cost estimation | — | ⬜ | No price table, no cost field |

### 4.3 Dataset generation (PR #2 only)

| Stage | Module | Status | Verified defect |
|---|---|---|---|
| Schemas | `data/models.py` | 🟡 | `NormalizedUsage` raises on `total < input + output`, so an anomalous provider response becomes a *failed run* and the usage is lost instead of being quarantined (reproduced). 1 mypy error. |
| IO | `data/io.py` | 🔴 | **SyntaxError** (lines 47 and 58) |
| Prompt bank | `prompts/hand_authored.jsonl` | 🟡 | 36 prompts across 18 categories (2 each). Below Pilot 0's ~50 and the workflow doc's 200–500. `01_import_prompts` is missing. |
| Manifest builder | `experiments/manifest.py` | 🟡 | Deterministic hashed IDs ✓. Prompt paths are resolved as `config_path.parents[2]`, so a config outside `configs/experiments/` fails (reproduced: `FileNotFoundError: '/prompts/hand_authored.jsonl'`). |
| Runner | `experiments/runner.py` | 🟡 | Resume and backoff ✓. **`--retry-failed` appends a second record with the same `run_id`, and validation then REJECTs both the failure and the successful retry** (reproduced). |
| Provider: mock | `providers/mock.py` | 🔴 | `provider_request_id` hashes only the prompt, so 4 rows per prompt share an ID. **All 144 pilot rows are quarantined as `duplicate_provider_request_id`** (reproduced; 36 distinct IDs for 144 rows). |
| Provider: OpenAI | `providers/openai.py` | 🟡 | Never sets `finish_reason`. Never detects refusals. Untested. Model IDs in configs (`gpt-5.6-luna`, `gpt-5.6-sol`) not verified against the provider. |
| Provider: Anthropic | `providers/anthropic.py` | 🟡 | Rejects every effort except `auto`/`none` (intentional). No reasoning/thinking tokens, no total, no refusal detection. Untested. Model IDs in configs (`claude-fable-5`, `claude-sonnet-5`) not verified. |
| Normalizer | `data/normalize.py` | 🟡 | Works. Has no attempt deduplication (see runner). No unit test. |
| Validator | `data/validate.py` | 🟡 | Rules are sound. The `inconsistent_total_tokens` rule can't trigger on adapter data because the schema raises first. The `provider_refusal` rule can't trigger because no adapter sets `is_refusal`. |
| Feature registry / leakage guard | `data/feature_registry.py` | 🟡 | Hard guard ✓ and tested. Marks `input_tokens` as post-inference, which is correct for the provider-reported value, but no pre-request token-count feature exists to replace it. |
| Feature builder | `data/build_features.py` | 🟡 | Works. Includes `category`, which the API never receives (train/serve skew). |
| Splitter | `data/split.py` | 🟡 | Family-grouped ✓. With 2 families per category, rounding gives **train 18 / test 18 / validation 0 / calibration 0** families (reproduced). Crashes with `KeyError` on empty input. |
| Profiler | `data/profile.py` | ✅* | Simple and reasonable. No test. |
| Release | `data/release.py` | 🟡 | Checksums ✓. **Overwrites an existing version silently** (reproduced). Manifest lacks git SHA, source experiments, providers/models, config hashes, and `schema.json` (all required by `DATASET_CREATION_WORKFLOW.md` §16). |
| CSV view | `data/csv_view.py`, `data/records.csv` | 🟡 | The checked-in `records.csv` shows 5 `PASS` rows, which the current code can't produce (every row is quarantined). It was not generated by this pipeline version. |
| CLI | `cli.py`, `scripts/02–09` | 🟡 | Works once io.py is fixed. No tests. |

### 4.4 Database schemas

**None.** There's no database, ORM or migrations on either branch. Persistence is file-based (JSONL + Parquet) on PR #2, which matches `DATASET_CREATION_WORKFLOW.md` §2/§17. That doc defers PostgreSQL until live observation ingestion.

### 4.5 CI/CD (`.github/workflows/ci.yml`)

| Item | `main` | PR #2 |
|---|---|---|
| Trigger | push to `main`, all PRs | same |
| Python | 3.11 only | same |
| Steps | install `.[dev]` → ruff → pytest | install `.[dev,data]` → ruff → pytest → 7-step mock pipeline smoke |
| mypy | **not run** (strict config exists) | not run |
| Coverage gate | none | none |
| CD / deploy / container | none | none |
| Status | green | **red** (Lint) |

### 4.6 Frontend (`src/seshat/static/index.html`, single file, no build step)

| Feature | Status | Notes |
|---|---|---|
| Prompt, model and effort input; calls the real `/v1/predict` | ✅ | Tested by `test_demo_ui` (string-presence check only) |
| Prediction, interval, "confidence" display | 🟡 | Shows `35% confidence`, which reads as a calibrated probability but is a constant |
| Model dropdown (GPT-5.6, GPT-6, Claude-class, Gemini-class) | 🟡 | Misleading: the predictor ignores the model, so every option gives the same number |
| Effort comparison table | 🟡 | Sends **5** HTTP requests per click (selected + 4 efforts) |
| Cost display | ⬜ | Requirement 8 |
| Actual-vs-predicted comparison | ⬜ | Requirement 8 |
| Header badge "Live baseline", "API connected" | 🟡 | Static text until the first call |

### 4.7 Documentation

| Doc | Branch | Status |
|---|---|---|
| `README.md` | main | Accurate for `main` |
| `README.md` | PR #2 | Rewritten. **Drops the `pytest` / `ruff` "Test" section.** |
| `docs/DATASET_CREATION_WORKFLOW.md` | both | Canonical design. Specifies YAML configs, but the implementation uses JSON. Specifies a release `schema.json` and `README.md`, which are not produced. |
| `docs/DATASET_IMPLEMENTATION.md` | PR #2 | Claims the workflow is implemented. Overstated, given §2.2. |
| `data/README.md`, `artifacts/README.md` | both | Fine |

---

## 5. Incomplete integrations and duplicated logic

### 5.1 Incomplete integrations

1. **Data factory → model → API.** No training code consumes a release, and the API never loads an artifact.
2. **Train/serve feature skew.** Training features include `provider`, `model_requested`, `category`, `max_output_tokens` and `temperature`. The API request has only `prompt`, `model`, `effort`, so `category`, `provider`, `max_output_tokens` and `temperature` are unavailable at serving time. The field names also differ (`model` vs `model_requested`).
3. **Effort vocabularies disagree.** API: `auto|low|medium|high` (`schemas.py`). Data factory: `auto|none|low|medium|high|xhigh|max` (`data/models.py`). The mock adapter treats `auto` as 1.0, OpenAI omits the reasoning parameter for `auto`, and Anthropic accepts only `auto`/`none`.
4. **Settings.** The `SESHAT_ENV`, `SESHAT_MODEL_PATH` and `SESHAT_LOG_LEVEL` keys in `.env.example` are read by nothing, and there is no settings module or logging configuration.
5. **Config files no code reads.** `configs/providers.json` and `configs/dataset_schema.json` are never loaded (grep: 0 references). The runner hard-codes `PROVIDERS`, and `feature_registry.py` hard-codes feature roles.
6. **Adapter fields the validator depends on are never populated.** `is_refusal` (all adapters) and `finish_reason` (OpenAI) are never set.

### 5.2 Duplicated or competing sources of truth

| Concept | Copies | Risk |
|---|---|---|
| Effort level enum | `schemas.EffortLevel`, `data.models.EffortLevel` | Already diverged |
| Feature roles / forbidden features | `data/feature_registry.py`, `configs/dataset_schema.json` | Currently consistent, but nothing enforces it |
| Provider registry | `experiments/runner.PROVIDERS`, `configs/providers.json` | JSON is decorative |
| Token-total plausibility | `NormalizedUsage` validator, `validate.py` rule | Schema copy pre-empts the quality rule and loses data |
| Effort multipliers | `predictor.py` (0.8/1.0/1.35), `providers/mock.py` (0.8/1.0/1.25) | Mock data partly mirrors the heuristic it would evaluate |
| CLI vs scripts | `cli.py` and 8 wrappers in `scripts/` | Low (thin wrappers), but two entry points to document |
| CSV views | `data/records.csv` (checked in), `release/*/records.csv` | Checked-in copy is stale or hand-made |

---

## 6. Technical debt

Severity: **H** = blocks the milestone or produces wrong data · **M** = correctness or maintainability risk · **L** = hygiene

| ID | Sev | Debt | Where |
|---|---|---|---|
| TD-01 | H | Syntax error breaks the whole data package | PR #2 `data/io.py:47,58` |
| TD-02 | H | Mock request IDs collide, so 100% of rows are quarantined and the pipeline crashes | PR #2 `providers/mock.py:31` |
| TD-03 | H | Validation and calibration splits are empty at current corpus size; no guard | PR #2 `data/split.py` |
| TD-04 | H | `--retry-failed` causes the successful retry to be REJECTed | PR #2 `runner.py` + `validate.py` |
| TD-05 | H | Provider anomalies are lost instead of quarantined (schema raises) | PR #2 `data/models.py:83` |
| TD-06 | H | Releases are mutable (same version silently overwritten); lineage incomplete | PR #2 `data/release.py:39` |
| TD-07 | H | Predictor ignores `model`; constant "confidence" presented as a percentage in the UI | `predictor.py`, `index.html` |
| TD-08 | M | No request size limit on `/v1/predict` | `schemas.py` |
| TD-09 | M | mypy configured strict but not in CI; PR #2 has 3 real type errors + missing `pandas-stubs` | `ci.yml`, PR #2 |
| TD-10 | M | CI tests only Python 3.11; no coverage gate | `ci.yml` |
| TD-11 | M | Unused runtime dependencies: `tiktoken`, `joblib`, `scikit-learn`, `numpy` (and `pandas` on `main`) | `pyproject.toml` |
| TD-12 | M | Config path resolution hard-wired to directory depth | PR #2 `manifest.py:98` |
| TD-13 | M | Provider adapters and the CLI have no tests (no SDK fakes) | PR #2 |
| TD-14 | M | Duplicated sources of truth (§5.2) | both |
| TD-15 | M | Docs overstate completeness; README loses test instructions; YAML/JSON drift | PR #2 docs |
| TD-16 | L | `httpx` + `starlette.testclient` deprecation warning | tests |
| TD-17 | L | 12 auto-fixable ruff findings | PR #2 |
| TD-18 | L | UI sends 5 requests per click; uses `innerHTML` for server and option values | `index.html` |
| TD-19 | L | Provider model IDs in configs unverified against live model lists | PR #2 `configs/` |

---

## 7. Missing requirements (vs. project requirements in CLAUDE.md)

| # | Requirement | Status | Gap |
|---|---|---|---|
| 1 | Predict input, output, reasoning, total tokens | 🟡 | Output only, heuristic. No input-token count (tiktoken unused, no provider `count_tokens`). No reasoning/total. No double-counting policy for reasoning inside output. |
| 2 | Single prediction + confidence + prediction intervals | 🟡 | Point estimate + ad hoc ±65% band. Not calibrated. "Confidence" is a constant. |
| 3 | Cost from provider-specific pricing | ⬜ | Nothing |
| 4 | Modular provider adapters | 🟡 | Exists only for *data collection* on PR #2. Prediction path is provider-agnostic only because it ignores the provider. |
| 5 | Learn from errors and observed usage | ⬜ | No feedback endpoint, storage, prediction IDs, or retraining |
| 6 | REST API | 🟡 | One prediction endpoint. No auth, limits, versioned errors or observation ingestion. |
| 7 | SDKs, IDE extensions, chatbot integrations | ⬜ | Future |
| 8 | Web UI: predictions, confidence, cost, actual-vs-predicted | 🟡 | Predictions + interval only |
| — | Never label a prediction as an observation | 🟡 | Checked-in `data/records.csv` shows PASS "observations" the current code can't generate. Label or regenerate it. |
| — | Reproducibility / lineage | 🟡 | Deterministic IDs and seeds ✓. Release lacks code SHA and config hashes. |

---

## 8. Dependency-ordered backlog

Each item lists its prerequisites. Items in the same phase with no mutual dependency can run in parallel. Phase 0 is the shortest path to a green, merged PR #2.

### Phase 0 — Make PR #2 correct and mergeable

| ID | Task | Depends on | Done when |
|---|---|---|---|
| B-01 | Restore the two `"\n"` literals in `data/io.py` | — | `python -m py_compile` passes; 15 tests collect |
| B-02 | `ruff --fix`; fix 3 mypy errors; add `pandas-stubs` to dev; mypy override for optional `openai`/`anthropic` | B-01 | ruff + mypy clean |
| B-03 | Make mock `provider_request_id` unique per call (include model, effort, repetition / run nonce) | B-01 | Pilot smoke run yields PASS rows |
| B-04 | Splitter: allocate families so every split is non-empty (or fail loudly with a clear message); handle empty input | B-01 | Seed corpus produces 4 non-empty splits; test added |
| B-05 | Retry semantics: record attempts explicitly; normalizer selects the final attempt per `run_id`; duplicate-run rule applies only to true duplicates | B-01 | Retry test: failed→success yields one PASS row |
| B-06 | Move total-token plausibility out of the `NormalizedUsage` validator; keep it only as a quality rule | B-01 | Anomalous usage lands in QUARANTINE, raw preserved |
| B-07 | Release immutability + lineage: refuse an existing version; add git SHA, source experiments, providers/models, config hashes, `schema.json` | B-01 | Test: second freeze of the same version errors |
| B-08 | Resolve prompt paths from an explicit `--root` / repo root, not `parents[2]` | B-01 | Config outside `configs/experiments/` works |
| B-09 | End-to-end mock pipeline test in pytest (mirrors the CI smoke); unit tests for normalize, release, CLI, adapters (with SDK fakes) | B-03…B-08 | Coverage of `data/` and `experiments/` ≥ 80% |
| B-10 | Regenerate `data/records.csv` from the real pipeline (or remove it); restore README test section; align docs (JSON vs YAML, release contents) | B-03 | Docs match code |
| B-11 | Merge PR #2 | B-01…B-10 | CI green |
| B-12 | CI: add mypy, Python 3.11 + 3.13 matrix, coverage floor; resolve the httpx deprecation | B-11 | CI enforces types and coverage |

### Phase 1 — One source of truth

| ID | Task | Depends on |
|---|---|---|
| B-13 | Single shared `EffortLevel` + explicit per-provider effort mapping; API and data factory use the same type | B-11 |
| B-14 | Make `feature_registry.py` authoritative; generate or verify `dataset_schema.json` from it in a test. Either load `providers.json` in the runner or delete it. | B-11 |
| B-15 | Settings module (`pydantic-settings` or plain env parsing) consuming `SESHAT_*`; logging config | B-11 |
| B-16 | Adapter completeness: OpenAI `finish_reason`; refusal detection (OpenAI + Anthropic); verify model IDs against live model lists | B-11 |

### Phase 2 — Pilot 0 data milestone (`DATASET_CREATION_WORKFLOW.md` §18)

| ID | Task | Depends on |
|---|---|---|
| B-17 | Pre-request input-token counting module (tiktoken for OpenAI; provider count-tokens endpoints where available) exposed as a `REQUEST_DERIVED` feature | B-13, B-14 |
| B-18 | Expand the reviewed prompt bank to ≥ 50 (target 200–500); add `01_import_prompts` with provenance and license checks | B-11 |
| B-19 | Spend preflight: estimated cost of a manifest before `run` (uses B-17 + a price table stub) | B-17 |
| B-20 | Run the paid pilot (`pilot_openai_001`), inspect normalization, freeze Dataset v0.0.1 | B-16, B-18, B-19, B-07 |

### Phase 3 — First trained predictor

| ID | Task | Depends on |
|---|---|---|
| B-21 | Evaluation harness: MAE, MAPE, pinball loss, interval coverage and width, per-segment breakdown; protected test set | B-20 |
| B-22 | Statistical baseline from the release (per model × effort × category quantiles) with a versioned artifact + lineage | B-21 |
| B-23 | Split conformal calibration on the `calibration` split; replace constant `confidence` with a nominal coverage level + measured coverage | B-22 |
| B-24 | Serve the artifact: predictor loads `SESHAT_MODEL_PATH`; API request gains the fields training uses (`provider`, optional `category`, `max_output_tokens`) to close the skew; `/health` reports model version | B-22, B-13, B-15 |
| B-25 | Candidate regressors (scikit-learn GBM / quantile models) compared against B-22 | B-21 |

### Phase 4 — Cost and multi-quantity prediction

| ID | Task | Depends on |
|---|---|---|
| B-26 | Versioned provider price table + `pricing` module (input / cached / output / reasoning rates); cost in `/v1/predict` | B-17, B-24 |
| B-27 | Predict input, reasoning and total tokens explicitly, with a documented no-double-counting rule per provider | B-17, B-22 |
| B-28 | API hardening: prompt size limit (can ship immediately), API keys, rate limiting, structured errors | B-15 |

### Phase 5 — Feedback loop and product surface

| ID | Task | Depends on |
|---|---|---|
| B-29 | Prediction IDs + `POST /v1/observations`; persistence (SQLite → PostgreSQL per workflow doc); predicted-vs-actual join | B-24 |
| B-30 | UI: cost, actual-vs-predicted view, honest calibration labelling, single batch request for the effort table | B-26, B-29 |
| B-31 | Error monitoring, drift detection, residual correction / scheduled retraining (production data is *not* auto-promoted to training data) | B-29, B-23 |
| B-32 | Python SDK, then IDE / chatbot integrations | B-26, B-29 |

### Quick wins (no dependencies, any time)

- B-28a: `max_length` on `PredictionRequest.prompt`.
- Remove or adopt the unused dependencies (TD-11).
- Mark the demo UI's confidence and model selector as "uncalibrated heuristic" until B-23 and B-24 land.

---

## 9. Recommended next task

**B-01 → B-04 on the PR #2 branch, as one small PR.** Fix the syntax error, make mock request IDs unique, and make splits non-empty. Together these turn the branch's own CI smoke pipeline from crash to green and unblock every later phase. The scratch-copy reproductions in §2.2 and §4.3 can be turned into regression tests directly.
