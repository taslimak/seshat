# Forecast quality: outcomes, confidence margins and the closed loop

Seshat stores not only *request → prediction → actual* but the realized error of
every forecast, and uses that evidence to correct, calibrate, monitor and retrain
itself:

```
request ─► point forecast (A) ─► bias correction (B) ─► residual/scale model (C)
        ─► conformal calibration (D) ─► preflight response with confidence margin
actual usage ─► prediction_outcomes (E) ─► snapshots · drift · run reports
             ─► challenger training / recalibration / adaptive collection (F)
```

## Sign convention and targets

`error = prediction − actual`. Positive = over-prediction, negative = under-prediction.

| Target | Definition | Censored (truncated) rows |
|---|---|---|
| **Latent output tokens** | what the model would generate without a cap | only a lower bound (`true ≥ observed = cap`); never an exact label |
| **Realized output tokens** | `min(latent, max_output_tokens)` — what is generated and billed | **exact** (the response produced exactly the cap) |

Forecast of realized usage: `min(point forecast, max_output_tokens)` (the cap is
known before dispatch). Because realized usage is observed for every row it has
no selection bias, so it is the **primary model-selection and promotion target**.
Latent metrics are reported on uncensored rows only, with lower-bound diagnostics
for censored rows. Cost and total-token errors use realized usage.

> Caveat: when a cap binds, any forecast ≥ cap is exactly right for realized
> usage. Realized metrics are therefore flattered on data where caps often bind;
> always read them next to the "cap not binding" view in run reports.

## Per-prediction metrics (`prediction_outcomes`)

Stored at `quality/prediction_outcomes/predictor=<version>/outcomes.parquet` in the
store, rebuilt deterministically from `predictions` + `observations` (idempotent;
reruns never duplicate). Raw predicted and actual values are always retained.

| Column(s) | Meaning |
|---|---|
| `signed_error_tokens`, `signed_percentage_error` | `p − y`; `100 (p − y) / y` (latent, uncensored) |
| `absolute_error_tokens`, `absolute_percentage_error` | `|p − y|`; `100 |p − y| / y` |
| `symmetric_absolute_percentage_error` | `100 · 2|p − y| / (|p| + |y|)`, bounded 0–200 |
| `squared_error_tokens2` | `(p − y)²` (RMSE, catastrophic analysis) |
| `log_error`, `multiplicative_error` | `log((p+1)/(y+1))`; `exp(log_error)` (1.20 ≈ 20% too high) |
| `error_direction` | `overprediction` / `underprediction` / `near_exact` (`|e| ≤ max(2, 2% of y)`) |
| `pct_denominator_too_small` | `y < 10` tokens: percentages undefined (NaN), sMAPE still reported |
| `target_is_censored`, `observed_lower_bound_tokens` | truncated; observed value is a lower bound |
| `prediction_below_known_lower_bound`, `lower_bound_error_tokens/_percentage` | censored row where `p < y`: definitely under-predicted, by at least `y − p` |
| `interval_position` | `below_interval` / `inside_interval` / `above_interval` / `unknown_censored` |
| `interval_hit`, `interval_miss_tokens/_percentage`, `interval_width_tokens/_percentage` | as named; miss = distance outside `[L, U]` |
| `realized_*` | the same metrics for realized usage (exact for every row), incl. `realized_interval_hit` with bounds capped at `max_output_tokens` |
| `cost_*`, `squared_cost_error_usd2` | predicted vs calculated cost (price-table estimate, **not** an invoice) |
| `total_*`, `reasoning_*`, `preflight_input_*` | total-token, reasoning-token (where reported) and preflight-input-counter errors |

Lineage columns: prediction/observation/sample/request/run/experiment ids,
prediction and observation timestamps, predictor / uncertainty-model /
calibration / dataset / feature-set / pricing versions, code revision,
`label_origin`, `source_type`, `prediction_mode`, `evaluation_role`,
`valid_out_of_sample`, `metrics_version`.

## Aggregate metrics (`aggregate_metrics`, snapshots, run reports)

Central error (MAE, median AE, RMSE; WAPE, median APE, sMAPE, nRMSE = RMSE / mean
actual), bias (mean/median signed error; mean signed % error; aggregate % bias =
Σ(p − y)/Σy), error percentiles (p50/75/90/95/99, tokens and %), directional rates
and magnitudes, catastrophic counts and rates (> 25 / 50 / 100 % APE; > 250 / 500 /
1000 tokens; `quality_metrics/1.0.0`), interval performance (hits, misses,
coverage, coverage bounds counting censored unknowns as hit/miss, below/above
rates, widths, interval score), confidence-margin averages/medians, cost accuracy,
censoring counts and rate — each with `n_observations`.

**Snapshots** (`quality/model_quality_snapshots.parquet`): per predictor version ×
calibration × label origin × window (lifetime, last 50/100/500, trailing 7/30 days)
× slice (provider/model, effort, category, subcategory, task group, prompt-length
band, predicted/actual output band, response format, tool use, context band,
experiment). Slices with `n < 20` are kept but `reliable = false`, with
`fallback_slice` naming the parent (model+category+effort → model+category →
model → global) to display instead.

**Run reports** (`quality/run_reports/<experiment>__<predictor>.md`): the
per-run summary (real outcomes only).

## Confidence level vs confidence margin

* **`confidence_level`** (e.g. 0.90) is the nominal coverage the interval targets:
  about 90% of actual values should fall inside it. It says nothing about precision.
* **`confidence_margin_tokens` / `confidence_margin_pct`** is how far the actual
  may plausibly fall from the forecast *at that level*: a calibrated forecast-error
  margin, not a probability that the forecast is right.

Request A `1,000 ± 80` and request B `1,000 ± 500` can both be 90% intervals;
A is far more predictable. Intervals are asymmetric (log scale), so responses
carry `lower_margin_*` and `upper_margin_*`; the display margin is
`max(lower, upper)` so `forecast ± margin` covers the calibrated interval. The
heuristic baseline reports `interval_calibrated = false` and no confidence level.

## Methodology

* **A — point forecast**: heteroscedastic censored log-normal regression
  (`log1p(y) ~ N(Xβ, exp(Xγ)²)`, right-censored at `log1p(cap)`; MAP with L2 penalties;
  σ smoothly bounded to [0.15, 1.5]). Features are computed from the exact request
  by `seshat.modeling.features.model_features` — the same function at training and
  serving time. Design labels (category) are never features.
* **B — bias correction**: Kaplan–Meier median of out-of-fold log residuals per
  `provider/model → effort → task group`, shrunk toward the parent with weight
  `n / (n + 15)`, capped at ±1 on the log scale; kept only if it lowers OOF
  realized MAE.
* **C — residual/scale model**: the σ(x) part of A (requests with inherently
  unpredictable length get wider margins).
* **D — calibration**: split-conformal quantiles of σ-normalised OOF log residuals,
  estimated with Kaplan–Meier (censoring-aware) at `α/2` and `1 − α/2` with the
  finite-sample correction; per-segment quantiles only for segments with
  `≥ 2/α − 1` rows, shrunk to global; global vs segment chosen by OOF interval score.
* **E — outcome capture**: `seshat-data quality-refresh` after every store sync.
* **F — learning**: challengers are trained when ≥ 50 new real observations
  arrive or drift is detected; the batch planner weights categories by realized
  WAPE, coverage shortfall and catastrophic rate (`forecast_quality_priority` in the
  dry-run report).

## Evaluation and promotion

`seshat-data train-forecaster --root <store> --output-dir <dir> [--register]`:

1. Protected test = leakage groups whose smallest prompt-family SHA-256 bucket
   is < 0.2 — stable as data grows; scored once per evaluation.
2. Grouped 5-fold CV on the rest → OOF forecasts for every candidate (baseline,
   group median, uncensored ridge, uncensored GBM, censored log-normal penalty grid).
3. Selection by OOF realized MAE; B and D fitted on OOF residuals.
4. All candidates and the baseline scored on the same protected test rows.
5. Gates (realized target): enough test rows, lower WAPE and MAE than the
   baseline, |bias| ≤ 15%, catastrophic (> 100%) rate not worse, coverage within
   7.5 pp of nominal, median interval width ≤ 400%, no category's MAE > 25% worse.

`evaluation.json` (immutable, also appended to `quality/model_evaluations.jsonl`)
records dataset/predictor/uncertainty/calibration versions, counts, hyper-
parameters, OOF selection table, protected-test metrics for every candidate with
slices, gate results and protocol notes. **A PASS is a recommendation;** the API
serves a forecaster only when an operator sets `SESHAT_FORECASTER_ARTIFACT`.

## Drift

`quality/drift/<predictor>.json`: recent outcomes vs the model's calibration
baseline (OOF + protected test), or the newest experiment vs earlier ones when only
backtests exist. Tests: Mann–Whitney on |realized log error|, Welch t on mean log
error (bias), one-sided binomial on coverage, Fisher exact on catastrophic rate.
Flags need ≥ 30 outcomes per side and p < 0.01; otherwise `insufficient_data`.

## Leakage and isolation safeguards

* Group- and family-disjoint dev/test; folds never split a leakage group.
* Preprocessing, bias correction and calibration fitted on training/OOF data only.
* `valid_out_of_sample` marks forecasts that did not see their observation;
  only those enter snapshots, drift and reports.
* Historical quality statistics are **not** used as model features (no
  point-in-time join is needed for training); they drive collection priority and
  retraining decisions only.
* `label_origin` separates real from synthetic outcomes everywhere; synthetic
  outcomes never train, calibrate or appear in real-world reporting.

## Live shadow evaluation (forward validation)

A batch config with a `shadow:` block runs an **observation-only** forecaster on
every paid request:

```
request -> provider-native input count (preflight) -> shadow forecast + confidence margin
        -> shadow_prediction event (persisted) -> dispatch -> response -> observation
        -> PredictionRecord (live_shadow_preflight / forward_validation) -> outcomes
```

* **Pre-inference only.** `CalibratedPredictor.predict_request(request,
  preflight_input_tokens=…)` sees the planned request (a deep copy) and the
  preflight result — nothing from the response. Training and serving share
  `model_features`.
* **One input count.** The provider-native preflight count is passed to the
  forecaster and used for `predicted_total_tokens` and `predicted_cost_usd`. If
  native counting fails, the approximation is used and labelled
  `approximate_fallback_after_native_failure` — never `provider_native`.
* **Persisted before dispatch.** The forecast is an event in `events.jsonl`
  written before `request_dispatched`; normalization keeps it only if its
  timestamp precedes the first dispatch, and outcomes mark it valid only if it
  precedes the observation (`prediction_precedes_observation`).
* **No effect on execution.** Request, cap, model, routing and budget are
  unchanged. If a *required* shadow forecast cannot be produced, that request is
  skipped rather than run without a forecast; an unverifiable artifact stops the
  run before any request.
* **Never activated.** The API keeps the baseline unless an operator sets
  `SESHAT_FORECASTER_ARTIFACT`.
* **Artifact handling.** The public config pins `artifact_sha256` and versions;
  the workflow fetches `forecaster.json` from the private store with the
  read-only secret `SESHAT_STORE_READ_TOKEN` into `$RUNNER_TEMP` (outside the
  uploaded `data/`), verifies the digest, and refuses to run without it.

Prediction modes and evaluation regimes (never pooled):

| `prediction_mode` | `evaluation_role` | `evaluation_regime` | Meaning |
|---|---|---|---|
| `live_shadow_preflight` | `forward_validation` | `true_forward_shadow` | forecast persisted before the request ran |
| `backtest_out_of_fold` | `cross_validation` | `out_of_fold_backtest` | retrospective, model did not see the row |
| `backtest_holdout` | `protected_test` | `protected_test` | retrospective, protected test set |
| `retrospective` | – | `untrained_heuristic` / `in_sample` | re-scored stored request |

`quality/forward_validation/<predictor>.{json,md}` reports the three regimes side
by side (MAE, median AE, WAPE, median APE, sMAPE, RMSE, bias, p50/p90/p95,
under/over-prediction, catastrophic rates, nominal vs actual coverage, margins,
cost WAPE, preflight input error) with slices by effort, category, output band and
prompt-size band. Only `true_forward_shadow` answers "how well did Seshat predict
new requests before they ran?".
