# Seshat

Seshat is a token-usage prediction service for AI prompts. The initial goal is to estimate likely output-token usage before a request is sent, return a confidence interval, and learn from prediction error after actual usage is observed.

## Initial scope

The starter implementation provides:

- a Python package under `src/seshat`
- deterministic prompt feature extraction
- a baseline prediction interface that can later be replaced by a trained model
- a FastAPI service with health and prediction endpoints
- tests for the feature and API layers
- a reproducible Python environment
- GitHub Actions for linting and tests

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn seshat.api:app --reload
```

Then open `http://127.0.0.1:8000/docs`.

## Test

```bash
pytest
ruff check .
```

## Project layout

```text
src/seshat/
  api.py          # HTTP interface
  features.py     # prompt feature extraction
  predictor.py    # baseline predictor contract/implementation
  schemas.py      # request/response models
tests/
  test_api.py
  test_features.py
```

## Near-term roadmap

1. Define and freeze the training-event schema.
2. Build the dataset ingestion pipeline.
3. Establish a simple statistical baseline.
4. Add model-family and effort-level calibration.
5. Train and compare candidate regressors.
6. Add residual correction and confidence calibration.
7. Version models and expose evaluation metrics.
