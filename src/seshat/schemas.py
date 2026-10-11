"""API and model schemas."""

from typing import Literal

from pydantic import BaseModel, Field

EffortLevel = Literal["auto", "low", "medium", "high"]


class PredictionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = Field(min_length=1)
    effort: EffortLevel = "auto"
    max_output_tokens: int | None = Field(default=None, ge=1)
    system: str | None = None


class PredictionResponse(BaseModel):
    """Backward compatible: the first five fields are unchanged.

    ``confidence`` is kept for old clients; prefer ``confidence_level`` (nominal
    interval coverage, null when the interval is not calibrated) and
    ``confidence_margin_tokens`` / ``confidence_margin_pct`` (how far the actual may
    plausibly fall from the forecast at that level).
    """

    predicted_output_tokens: int = Field(ge=0)
    lower_bound: int = Field(ge=0)
    upper_bound: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    predictor_version: str
    prediction_interval_lower: int | None = Field(default=None, ge=0)
    prediction_interval_upper: int | None = Field(default=None, ge=0)
    interval_calibrated: bool = False
    confidence_level: float | None = None
    confidence_margin_tokens: int | None = Field(default=None, ge=0)
    confidence_margin_pct: float | None = None
    lower_margin_tokens: int | None = Field(default=None, ge=0)
    upper_margin_tokens: int | None = Field(default=None, ge=0)
    lower_margin_pct: float | None = None
    upper_margin_pct: float | None = None
    expected_absolute_error_tokens: float | None = None
    predicted_total_tokens: int | None = None
    predicted_cost_usd: float | None = None
    uncertainty_model_version: str | None = None
    calibration_version: str | None = None
    dataset_version: str | None = None
