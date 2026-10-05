"""Prediction interface and starter baseline."""

from __future__ import annotations

from dataclasses import dataclass

from seshat.features import extract_prompt_features


@dataclass(frozen=True)
class Prediction:
    predicted_output_tokens: int
    lower_bound: int
    upper_bound: int
    confidence: float
    predictor_version: str


class BaselinePredictor:
    """A deliberately simple baseline to establish an end-to-end contract.

    This is not intended to be the production statistical model. It gives us a
    measurable reference implementation while dataset collection is built.
    """

    version = "baseline-0.1"

    def predict(self, prompt: str, model: str, effort: str) -> Prediction:
        features = extract_prompt_features(prompt)

        effort_multiplier = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.35,
            "auto": 1.0,
        }.get(effort, 1.0)

        # Conservative heuristic baseline. Replace after empirical calibration.
        estimate = max(32, round((48 + 0.55 * features.word_count) * effort_multiplier))
        interval = max(24, round(estimate * 0.65))

        return Prediction(
            predicted_output_tokens=estimate,
            lower_bound=max(0, estimate - interval),
            upper_bound=estimate + interval,
            confidence=0.35,
            predictor_version=self.version,
        )
