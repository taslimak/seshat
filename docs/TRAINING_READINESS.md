# Training readiness

`seshat-data readiness --root <store> [--markdown]` (also written to
`reports/training_readiness.*` after each consolidation) answers *can the
registered trainer run on this data, and if not, what is missing?*

## How it counts

1. Rows are selected exactly as `prepare-training` selects them (validated,
   real, controlled-experiment) and grouped with the same union-find leakage
   groups (prompt family, prompt hash, conversation hash, conversation id).
2. The **exact** seeded group-to-split assignment is simulated.
3. Per split it counts **usable** rows — what `train_predictor` accepts:
   verified, uncensored, with an observed output count.
4. It compares them with the trainer's minimums (train ≥ 100, calibration ≥ 20,
   test ≥ 20) for two ratio presets.

| Preset | train / validation / calibration / test | Usable rows implied by the minimums |
|---|---|---:|
| `default` (historical) | 0.64 / 0.12 / 0.08 / 0.16 | **250** (calibration is binding) |
| `real_label` (new) | 0.60 / 0.10 / 0.15 / 0.15 | **167** |

The earlier readiness script treated 140 usable rows as sufficient, which is
not reachable under either preset. Use `prepare-training --ratios real_label`
for real-label datasets.

Groups are allocated by count, not size, and censored rows still occupy
groups, so actual per-split usable counts vary; the report shows them and
estimates the additional usable rows needed (`shortfall / ratio` for the
worst split, assuming new independent groups).

It also lists coverage gaps: categories with no usable labels, fewer than 4
independent usable groups, censor rates above 20%, and single-level factors
(one model, one effort).

## Baseline diagnostics

`seshat-data evaluate-baseline --root <store> --output-dir <dir>` scores the
heuristic baseline on every real observation:

* point metrics (MAE, RMSE, median AE, WAPE, signed bias, median APE) on
  **uncensored** rows only, with 95% **group-bootstrap** CIs (resampling prompt
  families);
* interval coverage and width of the (uncalibrated) baseline interval;
* for **censored** rows, which are lower bounds: the share where the
  prediction is below the cap (known under-prediction) and where the interval
  upper bound is below the cap (known miss);
* slices by category, prompt-length bucket, effort and model, with cost per
  usable observation.

This is out-of-sample for the unfitted heuristic but subject to selection
bias (uncensored rows over-represent short outputs). It is a diagnostic, not
the protected-holdout evaluation of trained candidates.
