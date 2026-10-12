"""Live shadow evaluation: native preflight -> forecast -> persist -> dispatch -> outcome.

No network: a mock adapter stands in for the live provider (as in test_collector).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from seshat.data.collectors.collector import Collector
from seshat.data.collectors.events import EventLog, EventType
from seshat.data.collectors.normalize import build_shadow_predictions, normalize_experiment
from seshat.data.generators import generate_prompts
from seshat.data.pricing import PriceTable
from seshat.data.privacy import RetentionPolicy
from seshat.data.schemas.records import ObservationRecord
from seshat.data.storage import StorageLayout
from seshat.experiments.config import CollectionConfig, ShadowConfig
from seshat.experiments.design import build_plan
from seshat.modeling.features import model_features
from seshat.modeling.pipeline import train_and_evaluate
from seshat.modeling.shadow import ShadowArtifactError, ShadowForecaster
from seshat.predictor import CalibratedPredictor
from seshat.providers.base import ErrorCategory, ProviderAdapter
from seshat.providers.catalog import load_catalog
from seshat.providers.mock import FailurePlan, MockOpenAIAdapter
from tests.real_shape import mock_collection, relabel_as_real


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("shadow_model")
    data = mock_collection(base / "run", count=200, seed=21)
    relabel_as_real(data)
    train_and_evaluate(data, base / "model", n_folds=3)
    return base / "model" / "forecaster.json"


def _shadow(artifact: Path, **overrides: Any) -> dict[str, Any]:
    data = json.loads(artifact.read_text())
    shadow = {
        "artifact_ref": "models/x/forecaster.json",
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "predictor_version": data["predictor_version"],
        "calibration_version": data["calibration_version"],
        "dataset_version": data["dataset_version"],
        "artifact_path": str(artifact),
    }
    shadow.update(overrides)
    return shadow


def _config(tmp_path: Path, artifact: Path | None, **overrides: Any) -> CollectionConfig:
    raw: dict[str, Any] = {
        "experiment_id": "shadow_exp",
        "seed": 4,
        "prompts": {"generate": {"count": 34, "seed": 4}},
        "targets": [
            {
                "provider": "openai",
                "model": "gpt-6.1-sol",
                "efforts": ["low", "high"],
                "max_output_tokens": [1024],
            }
        ],
        "sampling": {"configs_per_prompt": 1},
        "repetitions": {"fraction": 0.0, "count": 1},
        "budget": {"max_requests": 12, "max_cost_usd": "1.00"},
        "execution": {"preflight": "native", "backoff_base_seconds": 0, "concurrency": 2},
        "storage": {"root": str(tmp_path / "data")},
    }
    if artifact is not None:
        raw["shadow"] = _shadow(artifact)
    raw.update(overrides)
    return CollectionConfig.model_validate(raw)


class SpyAdapter(MockOpenAIAdapter):
    """Mock stand-in for the live OpenAI adapter that records call order."""

    log: list[tuple[str, str]] = []
    seen_requests: list[dict[str, Any]] = []

    def count_input_tokens(self, request: Any) -> Any:
        SpyAdapter.log.append(("count", request.model_dump_json()))
        return super().count_input_tokens(request)

    def execute_request(self, request: Any, *, timeout_seconds: float) -> Any:
        SpyAdapter.log.append(("execute", request.model_dump_json()))
        SpyAdapter.seen_requests.append(request.model_dump(mode="json"))
        return super().execute_request(request, timeout_seconds=timeout_seconds)


def _factory(failures: FailurePlan | None = None) -> Callable[[str], ProviderAdapter]:
    adapters: dict[str, ProviderAdapter] = {}

    def make(provider: str) -> ProviderAdapter:
        if provider not in adapters:
            adapters[provider] = SpyAdapter(failures)
        return adapters[provider]

    return make


def _run(config: CollectionConfig, **kwargs: Any) -> Any:
    SpyAdapter.log, SpyAdapter.seen_requests = [], []
    plan = build_plan(config, generate_prompts(34, config.seed), load_catalog())
    collector = Collector(
        config,
        plan,
        mode="live",
        execute_live=True,
        max_cost_usd=kwargs.pop("max_cost_usd", Decimal("1.00")),
        adapter_factory=kwargs.pop("adapter_factory", _factory()),
        **kwargs,
    )
    return plan, collector.run()


def _events(config: CollectionConfig) -> list[dict[str, Any]]:
    return EventLog(
        Path(config.storage.root) / "raw" / config.experiment_id / "events.jsonl"
    ).read()


def _normalize(config: CollectionConfig) -> None:
    normalize_experiment(
        config.storage.root,
        config.experiment_id,
        catalog=load_catalog(),
        prices=PriceTable.load(),
        policy=RetentionPolicy(),
    )


def test_native_preflight_in_live_runs_and_no_provider_calls_in_dry_runs(
    tmp_path: Path, artifact: Path
) -> None:
    config = _config(tmp_path, artifact)
    plan, result = _run(config)
    preflights = [e for e in _events(config) if e["event"] == EventType.PREFLIGHT]
    assert len(preflights) == result.dispatched_this_run > 0
    assert {e["result"]["method"] for e in preflights} == {"provider_native"}
    assert sum(1 for kind, _ in SpyAdapter.log if kind == "count") == len(preflights)

    def no_provider(_: str) -> ProviderAdapter:
        raise AssertionError("dry run must not create provider adapters")

    dry = _config(tmp_path / "dry", artifact)
    dry_plan = build_plan(dry, generate_prompts(34, dry.seed), load_catalog())
    out = Collector(dry, dry_plan, mode="dry_run", adapter_factory=no_provider).run()
    assert out.dispatched_this_run == 0
    estimates = Path(dry.storage.root) / "raw" / "shadow_exp" / "dry_run" / "estimates.jsonl"
    methods = {json.loads(line)["preflight_method"] for line in estimates.read_text().splitlines()}
    assert methods == {"statistical_approximation"}


def test_shadow_forecast_is_persisted_before_dispatch_and_response(
    tmp_path: Path, artifact: Path
) -> None:
    config = _config(tmp_path, artifact)
    _, result = _run(config)
    events = _events(config)
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("sample_id"):
            by_sample.setdefault(event["sample_id"], []).append(event)
    assert len(by_sample) == result.dispatched_this_run
    for sample_events in by_sample.values():
        kinds = [e["event"] for e in sample_events]
        assert kinds.index("preflight") < kinds.index("shadow_prediction")
        assert kinds.index("shadow_prediction") < kinds.index("request_dispatched")
        assert kinds.index("request_dispatched") < kinds.index("response_received")
        shadow = next(e for e in sample_events if e["event"] == "shadow_prediction")
        response = next(e for e in sample_events if e["event"] == "response_received")
        assert shadow["prediction"]["created_at_utc"] < response["response"]["completed_at_utc"]
    _normalize(config)
    layout = StorageLayout(config.storage.root)
    predictions = layout.read_frame("predictions")
    observations = layout.read_frame("observations")
    assert len(predictions) == result.dispatched_this_run
    assert set(predictions["prediction_mode"]) == {"live_shadow_preflight"}
    assert set(predictions["evaluation_role"]) == {"forward_validation"}
    joined = predictions.merge(observations, on="request_id")
    assert len(joined) == len(predictions)
    assert (joined["created_at_utc"] < joined["collected_at_utc"]).all()


def test_supplied_native_count_is_the_single_input_figure(tmp_path: Path, artifact: Path) -> None:
    config = _config(tmp_path, artifact)
    _run(config)
    events = _events(config)
    native = {
        e["sample_id"]: e["result"]["input_tokens"]
        for e in events
        if e["event"] == EventType.PREFLIGHT
    }
    entry = PriceTable.load().lookup("openai", "gpt-6.1-sol", __import__("datetime").date.today())
    assert entry is not None
    for event in (e for e in events if e["event"] == EventType.SHADOW_PREDICTION):
        p = event["prediction"]
        assert p["preflight_input_tokens"] == native[event["sample_id"]]
        assert p["preflight_source"] == "provider_native"
        assert (
            p["predicted_total_tokens"] == native[event["sample_id"]] + p["predicted_output_tokens"]
        )
        expected_cost = (
            native[event["sample_id"]] * float(entry.rates.input)
            + p["predicted_output_tokens"] * float(entry.rates.output)
        ) / 1e6
        assert p["predicted_cost_usd"] == pytest.approx(expected_cost, rel=1e-6)


def test_native_failure_is_labelled_as_fallback(artifact: Path) -> None:
    from seshat.modeling.features import request_from_prompt

    forecaster = ShadowForecaster(ShadowConfig.model_validate(_shadow(artifact)))
    request = request_from_prompt(
        "Summarize this.", provider="openai", model="gpt-6.1-sol", effort="low"
    )
    failed = {"method": "provider_native", "input_tokens": None, "error": "429"}
    payload = forecaster.forecast(request, failed)
    assert payload["preflight_source"] == "approximate_fallback_after_native_failure"
    assert payload["preflight_method"] == "approximate_fallback_after_native_failure"


def test_shadow_never_alters_the_provider_request(tmp_path: Path, artifact: Path) -> None:
    config = _config(tmp_path, artifact)
    plan, _ = _run(config)
    planned = {s.request_id: s.request.model_dump(mode="json") for s in plan.samples}
    for sent in SpyAdapter.seen_requests:
        assert sent in planned.values()
    # Same requests, same order of provider calls, with or without the shadow model.
    with_shadow = [req for kind, req in SpyAdapter.log if kind == "execute"]
    control = _config(tmp_path / "control", None)
    _run(control)
    without_shadow = [req for kind, req in SpyAdapter.log if kind == "execute"]
    assert sorted(with_shadow) == sorted(without_shadow)


def test_resume_does_not_duplicate_shadow_predictions(tmp_path: Path, artifact: Path) -> None:
    config = _config(tmp_path, artifact)
    _run(config)
    first = [e for e in _events(config) if e["event"] == EventType.SHADOW_PREDICTION]
    _, second = _run(config)
    again = [e for e in _events(config) if e["event"] == EventType.SHADOW_PREDICTION]
    assert second.dispatched_this_run == 0 and len(again) == len(first)


def test_failed_dispatch_keeps_the_pre_dispatch_prediction(tmp_path: Path, artifact: Path) -> None:
    config = _config(tmp_path, artifact)
    failures = FailurePlan(by_call={1: ErrorCategory.INVALID_REQUEST})
    _run(config, adapter_factory=_factory(failures))
    _normalize(config)
    predictions = StorageLayout(config.storage.root).read_frame("predictions")
    events = _events(config)
    failed = {e["sample_id"] for e in events if e["event"] == EventType.FAILED}
    shadowed = {e["sample_id"] for e in events if e["event"] == EventType.SHADOW_PREDICTION}
    assert failed and failed <= shadowed
    assert len(predictions) == len(shadowed)


def test_required_shadow_failure_sends_nothing(tmp_path: Path, artifact: Path) -> None:
    class Broken:
        def forecast(self, request: Any, preflight: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("model unavailable")

    config = _config(tmp_path, artifact)
    _, result = _run(config, shadow=Broken())
    assert result.dispatched_this_run == 0
    reasons = {e.get("reason") for e in _events(config) if e["event"] == EventType.SKIPPED}
    assert reasons == {"shadow_prediction_failed"}


def test_unverified_artifact_stops_the_run_before_any_request(
    tmp_path: Path, artifact: Path
) -> None:
    config = _config(tmp_path, None, shadow=_shadow(artifact, artifact_sha256="0" * 64))
    with pytest.raises(ShadowArtifactError, match="digest"):
        _run(config)
    missing = _config(tmp_path, None, shadow=_shadow(artifact, artifact_path=None))
    with pytest.raises(ShadowArtifactError, match="no artifact"):
        _run(missing)
    assert SpyAdapter.log == []


def test_budget_cap_still_enforced_with_shadow(tmp_path: Path, artifact: Path) -> None:
    config = _config(tmp_path, artifact, budget={"max_requests": 12, "max_cost_usd": "0.02"})
    _, result = _run(config, max_cost_usd=Decimal("0.02"))
    assert result.stopped_reason == "max_cost_reached"
    assert result.spent_or_reserved_usd <= Decimal("0.02")


def test_late_forecasts_are_never_recorded_as_live_shadow(tmp_path: Path, artifact: Path) -> None:
    config = _config(tmp_path, artifact)
    plan, _ = _run(config)
    events = _events(config)
    tampered = []
    for event in events:
        if event["event"] == EventType.SHADOW_PREDICTION:
            event = {
                **event,
                "prediction": {
                    **event["prediction"],
                    "created_at_utc": "2999-01-01T00:00:00+00:00",
                },
            }
        tampered.append(event)
    assert build_shadow_predictions(plan.samples, tampered) == []


def test_forecaster_inputs_are_pre_inference_only() -> None:
    params = set(inspect.signature(CalibratedPredictor.predict_request).parameters)
    assert params == {"self", "request", "preflight_input_tokens", "preflight_method"}
    from seshat.modeling.features import request_from_prompt

    features = model_features(
        request_from_prompt("Hello", provider="openai", model="gpt-6.1-sol", effort="high")
    )
    post_inference = set(ObservationRecord.model_fields) - {"provider", "requested_model"}
    assert not set(features) & post_inference


def test_shadow_run_does_not_activate_the_challenger(tmp_path: Path, artifact: Path) -> None:
    _run(_config(tmp_path, artifact))
    assert "SESHAT_FORECASTER_ARTIFACT" not in os.environ
    import importlib

    import seshat.api

    api = importlib.reload(seshat.api)
    assert type(api.predictor).__name__ == "BaselinePredictor"


def test_forward_report_separates_true_shadow_from_backtests(
    tmp_path: Path, artifact: Path
) -> None:
    from seshat.data.cli import main as cli_main
    from seshat.quality.outcomes import read_outcomes
    from seshat.quality.refresh import refresh_quality

    config = _config(tmp_path, artifact)
    _, result = _run(config)
    _normalize(config)
    relabel_as_real(Path(config.storage.root))  # tests only: mark the mock run as real
    assert cli_main(["validate", "--root", config.storage.root]) == 0
    summary = refresh_quality(config.storage.root)
    outcomes = read_outcomes(config.storage.root)
    shadow = outcomes[outcomes["prediction_mode"] == "live_shadow_preflight"]
    assert len(shadow) == result.dispatched_this_run
    assert set(shadow["evaluation_regime"]) == {"true_forward_shadow"}
    assert shadow["valid_out_of_sample"].all() and shadow["prediction_precedes_observation"].all()
    version = json.loads(artifact.read_text())["predictor_version"]
    report = json.loads(
        (
            Path(config.storage.root) / "quality" / "forward_validation" / f"{version}.json"
        ).read_text()
    )
    assert report["live_shadow_predictions"] == report["matched_live_shadow_observations"]
    assert report["regimes"]["out_of_fold_backtest"] == {"n_observations": 0}
    assert summary["forward_validation"][version]["matched"] == result.dispatched_this_run
