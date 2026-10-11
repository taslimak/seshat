"""FastAPI entrypoint for Seshat."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from seshat.predictor import BaselinePredictor, TrainedPredictor
from seshat.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Seshat",
    version="0.1.0",
    description="Predict likely output-token usage before an AI request is executed.",
)

# Trained loading is opt-in; the existing baseline remains the default.
_model_file = os.environ.get("SESHAT_MODEL_ARTIFACT")
predictor = TrainedPredictor(_model_file) if _model_file else BaselinePredictor()
_UI_PATH = Path(__file__).with_name("static") / "index.html"


@app.get("/", include_in_schema=False)
def demo_ui() -> FileResponse:
    return FileResponse(_UI_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    result = predictor.predict(
        prompt=request.prompt,
        model=request.model,
        effort=request.effort,
    )
    return PredictionResponse(
        predicted_output_tokens=result.predicted_output_tokens,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        confidence=result.confidence,
        predictor_version=result.predictor_version,
    )
