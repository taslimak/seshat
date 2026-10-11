"""FastAPI entrypoint for Seshat."""

import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from seshat.predictor import BaselinePredictor, CalibratedPredictor, TrainedPredictor
from seshat.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Seshat",
    version="0.1.0",
    description="Predict likely output-token usage before an AI request is executed.",
)

# Trained models are opt-in (operator-controlled paths); the baseline remains the default
# until a forecaster passes its promotion gates and is explicitly activated.
_forecaster_file = os.environ.get("SESHAT_FORECASTER_ARTIFACT")
_model_file = os.environ.get("SESHAT_MODEL_ARTIFACT")
predictor: BaselinePredictor | TrainedPredictor | CalibratedPredictor
if _forecaster_file:
    predictor = CalibratedPredictor(_forecaster_file)
elif _model_file:
    predictor = TrainedPredictor(_model_file)
else:
    predictor = BaselinePredictor()
_UI_PATH = Path(__file__).with_name("static") / "index.html"


@app.get("/", include_in_schema=False)
def demo_ui() -> FileResponse:
    return FileResponse(_UI_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if isinstance(predictor, CalibratedPredictor):
        result = predictor.predict(
            request.prompt,
            request.model,
            request.effort,
            max_output_tokens=request.max_output_tokens,
            system=request.system,
        )
    else:
        result = predictor.predict(
            prompt=request.prompt,
            model=request.model,
            effort=request.effort,
        )
    fields = asdict(result)
    return PredictionResponse(
        **fields,
        prediction_interval_lower=result.lower_bound,
        prediction_interval_upper=result.upper_bound,
    )
