"""API and model schemas."""

from typing import Literal

from pydantic import BaseModel, Field

EffortLevel = Literal["auto", "low", "medium", "high"]


class PredictionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = Field(min_length=1)
    effort: EffortLevel = "auto"


class PredictionResponse(BaseModel):
    predicted_output_tokens: int = Field(ge=0)
    lower_bound: int = Field(ge=0)
    upper_bound: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    predictor_version: str
