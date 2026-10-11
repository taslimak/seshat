# Accumulating real API observations and evaluating the predictor

The trained predictor implementation is merged, but the two-response pilot is
not sufficient to train it. This update adds a read-only readiness report and
baseline comparison to the gated persistent paid-collection job.

After a real paid run completes on the dedicated persistent runner, the job:
1. Normalizes real observations into the same **durable** data root.
2. Validates accumulated observations.
3. Scores stored requests with the existing baseline.
4. Records a comparison CSV at `exports/baseline_comparison.csv`.
5. Writes `reports/training_readiness.json`, showing counts of eligible
   verified, uncensored, controlled API observations and provider coverage.

The report is a *preliminary gate*. It does not guarantee enough distinct
leakage groups in each training, calibration and test split. Once enough real
data is available, separately run `prepare-training` and
`python -m seshat.train_predictor` and compare protected holdout performance.

**This PR makes no API calls and does not enable paid collection.** The paid job
still needs repository variable `SESHAT_ENABLE_SCHEDULED_PAID=true`, a
private persistent self-hosted runner, provider credentials stored only as
GitHub Actions Secrets, and provider-side spending restrictions.

GitHub-hosted smoke-test ZIP artifacts remain **ephemeral and separate** from
this durable store. They must be securely imported into persistent storage
before they contribute to cumulative training. Avoid using fixture labels as
real training targets or counting censored requests as complete output lengths.

Limitations: the initial baseline comparison is descriptive, not a held-out
evaluation. Test-set MAE and interval coverage are produced separately after
the training dataset is prepared.
