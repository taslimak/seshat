"""Prediction interface and starter baseline."""

from __future__ import annotations

from dataclasses import dataclass

from seshat.features import PromptFeatures, extract_prompt_features


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
        return self.predict_features(extract_prompt_features(prompt), model, effort)

    def predict_features(self, features: PromptFeatures, model: str, effort: str) -> Prediction:
        """Predict from precomputed prompt features (no prompt text needed)."""
        return self.predict_from_word_count(features.word_count, model, effort)

    def predict_from_word_count(
        self, word_count: int, model: str, effort: str | None
    ) -> Prediction:
        """Core heuristic. Used for stored, privacy-safe request records too.

        Note: the baseline does not use ``model`` and its interval is not calibrated.
        """
        del model  # The heuristic baseline does not yet use the model identifier.
        effort_multiplier = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.35,
            "auto": 1.0,
        }.get(effort or "auto", 1.0)

        # Conservative heuristic baseline. Replace after empirical calibration.
        estimate = max(32, round((48 + 0.55 * word_count) * effort_multiplier))
        interval = max(24, round(estimate * 0.65))

        return Prediction(
            predicted_output_tokens=estimate,
            lower_bound=max(0, estimate - interval),
            upper_bound=estimate + interval,
            confidence=0.35,
            predictor_version=self.version,
        )


class TrainedPredictor:
    """Opt-in real-label model; artifact paths must be operator-controlled."""

    def __init__(self, artifact_path: str) -> None:
        import joblib

        # Never load joblib/pickle artifacts from untrusted upload sources.
        bundle = joblib.load(artifact_path)
        self._model = bundle["model"]
        self._radius = float(bundle["radius"])
        self.version = str(bundle["version"])

    def predict(self, prompt: str, model: str, effort: str) -> Prediction:
        import math

        import pandas as pd

        from seshat.train_predictor import _features

        features = extract_prompt_features(prompt)
        frame = pd.DataFrame([{
            "prompt_word_count": features.word_count,
            "requested_model": model,
            "requested_effort": None if effort == "auto" else effort,
        }])
        prediction = float(self._model.predict(_features(frame))[0])
        point = max(0, round(math.expm1(min(max(prediction, 0), 16))))
        radius = math.ceil(self._radius)
        return Prediction(
            predicted_output_tokens=point,
            lower_bound=max(0, point - radius),
            upper_bound=point + radius,
            confidence=0.90,
            predictor_version=self.version,
        )
