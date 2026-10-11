# Real-label predictor training

The current API has a working heuristic baseline. This PR adds an **optional**
trained ridge regression predictor for **real provider-observed output tokens**.

## Build a training dataset

After accumulating a sufficiently diverse, durable body of authorized real API
observations:

```bash
seshat-data validate --root /var/lib/seshat/data
seshat-data prepare-training --root /var/lib/seshat/data --version 0.2.0
python -m seshat.train_predictor \
  --dataset /var/lib/seshat/data/training/dataset_v0.2.0 \
  --output /var/lib/seshat/models/ridge_v0.2.0
```

The trainer refuses synthetic or non-controlled labels and filters censored,
partial, and unavailable usage. It requires **100 training, 20 calibration,
and 20 test** usable rows across leakage-disjoint groups. Split assignments
are inherited from the dataset preparation pipeline. Testing is performed on
the protected test partition only after model fitting and calibration. It
reports MAE, RMSE, and empirical 90% interval coverage.

## Serve an approved model

```bash
SESHAT_MODEL_ARTIFACT=/var/lib/seshat/models/ridge_v0.2.0/predictor.joblib \
  uvicorn seshat.api:app
```

Without `SESHAT_MODEL_ARTIFACT`, the API still uses the existing heuristic
baseline. Model artifacts use `joblib`, which must only be loaded from trusted,
operator-controlled paths. Do not load artifacts submitted by users.

## Honest limitations

- Our two-request live pilot is **not sufficient** to train a generalizable model.
- The model predicts *output tokens*, not future reasoning/total token costs.
- Input features are the prompt word count, API model ID, and reasoning effort,
  which exactly match the public API's current prediction inputs.
- Its 90% interval is derived from split-conformal absolute residuals, but the
  nominal coverage relies on calibration/test exchangeability. Do not present
  the number as a demonstrated reliability guarantee across new model families.
- Training is manually triggered; no paid API calls or automatic retraining are
  introduced by this PR.
