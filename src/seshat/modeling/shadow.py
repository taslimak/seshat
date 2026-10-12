"""Live shadow forecasting: an observation-only forecast made before each dispatch.

The collector calls :meth:`ShadowForecaster.forecast` after the preflight count and
before the provider request is sent, and records the result as a
``shadow_prediction`` event. Normalization turns those events into canonical
``PredictionRecord`` rows with ``prediction_mode = "live_shadow_preflight"`` and
``evaluation_role = "forward_validation"``; the quality loop joins them to the
observed usage like any other prediction.

Guarantees
----------
* **Pre-inference only.** Inputs are the planned request (a deep copy) and the
  preflight result. Nothing produced by the provider call exists yet.
* **One authoritative input count.** The collector's preflight count (provider-
  native in live runs) is passed to the forecaster; if native counting failed the
  approximation is used and labelled as a fallback -- never as native.
* **Observation-only.** The forecast does not change the request, cap, model,
  routing or budget. The shadow model is never activated for the API.
* **Integrity.** The artifact must match the SHA-256 and versions pinned in the
  reviewed batch config; otherwise no paid request is sent.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seshat.experiments.config import ShadowConfig

PREDICTION_MODE = "live_shadow_preflight"
EVALUATION_ROLE = "forward_validation"


class ShadowArtifactError(ValueError):
    """The shadow artifact is missing or does not match the reviewed config."""


def _code_revision() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class ShadowForecaster:
    def __init__(self, config: ShadowConfig) -> None:
        from seshat.predictor import CalibratedPredictor

        if not config.artifact_path:
            raise ShadowArtifactError(
                f"Shadow forecaster {config.predictor_version} required but no artifact was "
                f"provided (expected {config.artifact_ref} from the private store)"
            )
        path = Path(config.artifact_path)
        if not path.is_file():
            raise ShadowArtifactError(f"Shadow artifact not found at {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != config.artifact_sha256:
            raise ShadowArtifactError(
                f"Shadow artifact digest {digest} does not match the reviewed "
                f"{config.artifact_sha256}"
            )
        artifact = json.loads(path.read_text(encoding="utf-8"))
        for key in ("predictor_version", "calibration_version", "dataset_version"):
            if str(artifact.get(key)) != getattr(config, key):
                raise ShadowArtifactError(
                    f"Shadow artifact {key} {artifact.get(key)!r} != reviewed "
                    f"{getattr(config, key)!r}"
                )
        self.config = config
        self.predictor = CalibratedPredictor(str(path))
        self.artifact_sha256 = digest
        self.code_revision = _code_revision()

    def forecast(self, request: Any, preflight: dict[str, Any]) -> dict[str, Any]:
        """Event payload for one request. ``preflight`` is the recorded preflight result."""
        from dataclasses import asdict

        native = preflight.get("method") == "provider_native"
        count = preflight.get("input_tokens")
        if count is not None:
            tokens: int | None = int(count)
            source = "provider_native" if native else str(preflight.get("method"))
        else:
            tokens = None  # forecaster falls back to the approximation and says so
            source = (
                "approximate_fallback_after_native_failure"
                if native
                else "approximate_fallback_no_preflight"
            )
        prediction = self.predictor.predict_request(
            request.model_copy(deep=True),
            preflight_input_tokens=tokens,
            preflight_method=source,
        )
        payload = asdict(prediction)
        payload.update(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "prediction_mode": PREDICTION_MODE,
                "evaluation_role": EVALUATION_ROLE,
                "preflight_source": source,
                "preflight_error": preflight.get("error"),
                "artifact_sha256": self.artifact_sha256,
                "code_revision": self.code_revision,
            }
        )
        return payload
