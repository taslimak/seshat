"""Prediction interface and starter baseline."""

from __future__ import annotations

from dataclasses import dataclass

from seshat.features import PromptFeatures, extract_prompt_features


@dataclass(frozen=True)
class Prediction:
    """A pre-inference forecast.

    ``confidence_level`` is the nominal coverage the interval targets (e.g. 0.90);
    ``confidence_margin_*`` is how far reality may plausibly fall from the point
    forecast at that level (a calibrated forecast-error margin, not a probability
    that the forecast is right). Two forecasts can share a 90% level while one is
    +/-80 tokens and the other +/-500. ``interval_calibrated`` is False for the
    heuristic baseline, whose interval has no empirical support.
    """

    predicted_output_tokens: int
    lower_bound: int
    upper_bound: int
    confidence: float
    predictor_version: str
    interval_calibrated: bool = False
    confidence_level: float | None = None
    lower_margin_tokens: int | None = None
    upper_margin_tokens: int | None = None
    confidence_margin_tokens: int | None = None
    lower_margin_pct: float | None = None
    upper_margin_pct: float | None = None
    confidence_margin_pct: float | None = None
    expected_absolute_error_tokens: float | None = None
    predicted_total_tokens: int | None = None
    predicted_cost_usd: float | None = None
    uncertainty_model_version: str | None = None
    calibration_version: str | None = None
    dataset_version: str | None = None


def _margins(point: int, lower: int, upper: int) -> dict[str, float | int | None]:
    lower_margin, upper_margin = point - lower, upper - point
    margin = max(lower_margin, upper_margin)
    pct = (lambda v: round(100.0 * v / point, 1)) if point >= 1 else (lambda v: None)
    return {
        "lower_margin_tokens": lower_margin,
        "upper_margin_tokens": upper_margin,
        "confidence_margin_tokens": margin,
        "lower_margin_pct": pct(lower_margin),
        "upper_margin_pct": pct(upper_margin),
        "confidence_margin_pct": pct(margin),
    }


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

        lower, upper = max(0, estimate - interval), estimate + interval
        return Prediction(
            predicted_output_tokens=estimate,
            lower_bound=lower,
            upper_bound=upper,
            confidence=0.35,
            predictor_version=self.version,
            interval_calibrated=False,
            **_margins(estimate, lower, upper),  # type: ignore[arg-type]
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
        frame = pd.DataFrame(
            [
                {
                    "prompt_word_count": features.word_count,
                    "requested_model": model,
                    "requested_effort": None if effort == "auto" else effort,
                }
            ]
        )
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


class CalibratedPredictor:
    """Forecaster + bias correction + conformal calibration from a JSON artifact.

    The artifact (``forecaster.json`` from ``seshat-data train-forecaster``) is plain
    JSON -- no pickle -- and the features are computed by the same function used in
    training, from the same ``LLMRequest`` shape.
    """

    def __init__(self, artifact_path: str) -> None:
        import json
        from pathlib import Path

        from seshat.modeling.calibration import BiasCorrector, ConformalCalibrator
        from seshat.modeling.models import CensoredLogNormal

        data = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        if data.get("artifact_format") != "seshat-forecaster/1":
            raise ValueError("Unsupported forecaster artifact format")
        self.artifact = data
        self.version = str(data["predictor_version"])
        self.model = CensoredLogNormal.from_dict(data["model"])
        self.bias = (
            BiasCorrector.from_dict(data["bias_correction"]) if data["bias_correction"] else None
        )
        self.calibrator = ConformalCalibrator.from_dict(data["calibration"])

    def predict(
        self,
        prompt: str,
        model: str,
        effort: str,
        *,
        max_output_tokens: int | None = None,
        system: str | None = None,
    ) -> Prediction:
        import math

        import numpy as np
        import pandas as pd

        from seshat.data.pricing import PriceTable
        from seshat.modeling.features import model_features, request_from_prompt
        from seshat.modeling.pipeline import SEGMENT_COLUMNS
        from seshat.providers.base import approximate_preflight

        provider = _provider_for(model)
        request = request_from_prompt(
            prompt,
            provider=provider,
            model=model,
            effort=None if effort == "auto" else effort,
            system=system,
            max_output_tokens=max_output_tokens,
        )
        frame = pd.DataFrame([model_features(request)])
        mu, sigma = self.model.predict_log(frame)
        segments = {c: frame[c].astype(str).to_numpy() for c in SEGMENT_COLUMNS}
        if self.bias is not None:
            mu = mu + self.bias.delta(segments)
        q_lo, q_hi = self.calibrator.quantiles(segments, 1)
        cap = float(max_output_tokens) if max_output_tokens else math.inf
        point = int(round(min(max(math.expm1(float(mu[0])), 0.0), cap)))
        lower = int(round(min(max(math.expm1(float(mu[0] + q_lo[0] * sigma[0])), 0.0), point)))
        upper = int(round(min(max(math.expm1(float(mu[0] + q_hi[0] * sigma[0])), point), cap)))
        # E|Y - median| for a log-normal is not closed-form; Monte Carlo on the log scale.
        draws = np.expm1(mu[0] + sigma[0] * np.random.default_rng(0).standard_normal(4000))
        expected_abs = float(np.mean(np.abs(np.minimum(draws, cap) - point)))
        input_tokens = approximate_preflight(request, provider).input_tokens or 0
        total = input_tokens + point
        cost: float | None = None
        try:
            from datetime import date

            entry = PriceTable.load().lookup(provider, model, date.today())
            if entry is not None:
                cost = round(
                    (input_tokens * float(entry.rates.input) + point * float(entry.rates.output))
                    / 1e6,
                    6,
                )
        except (OSError, ValueError):
            cost = None
        level = float(self.artifact["confidence_level"])
        return Prediction(
            predicted_output_tokens=point,
            lower_bound=lower,
            upper_bound=upper,
            confidence=level,
            predictor_version=self.version,
            interval_calibrated=True,
            confidence_level=level,
            expected_absolute_error_tokens=round(expected_abs, 1),
            predicted_total_tokens=total,
            predicted_cost_usd=cost,
            uncertainty_model_version=str(self.artifact["uncertainty_model_version"]),
            calibration_version=str(self.artifact["calibration_version"]),
            dataset_version=str(self.artifact["dataset_version"]),
            **_margins(point, lower, upper),  # type: ignore[arg-type]
        )


def _provider_for(model: str) -> str:
    from seshat.providers.catalog import load_catalog

    catalog = load_catalog()
    for provider, spec in catalog.providers.items():
        if model in spec.get("models", {}):
            return str(provider)
    return "unknown"
