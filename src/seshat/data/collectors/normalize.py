"""Raw collection events -> canonical normalized records.

A pure function of ``(plan, events, catalog, prices, policy)``: re-running it always
produces the same normalized layer, so fixing a normalizer and re-normalizing never
needs a new paid request.

Labelling rules:

* Mock providers -> ``source_type=test_fixture``, ``label_quality=synthetic``.
* Real providers -> ``controlled_experiment``; label quality from the normalizer
  (verified / partial / censored / inconsistent / unavailable).
* Failed and uncertain samples get an observation with ``label_quality=unavailable``
  and no usage, so they are counted but never trained on.
* Contradictory or negative provider values are not repaired: the field becomes
  null, the original stays in ``raw_usage``, and the label is ``inconsistent``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from seshat.data.collectors.events import SampleState, SampleStatus, replay
from seshat.data.features.extract import extract_request_features
from seshat.data.io import write_json
from seshat.data.pricing import PriceTable, compute_cost
from seshat.data.privacy import RetentionPolicy
from seshat.data.schemas.enums import (
    CollectionStatus,
    ConsentStatus,
    CostMethod,
    LabelQuality,
    PreflightMethod,
    ResponseStatus,
    SourceType,
)
from seshat.data.schemas.records import (
    ObservationRecord,
    PredictionRecord,
    PreflightRecord,
    RequestRecord,
)
from seshat.experiments.design import PlannedSample
from seshat.providers.base import PreflightResult, ProviderResponse
from seshat.providers.catalog import Catalog
from seshat.providers.normalization import NORMALIZERS, NormalizedUsage
from seshat.providers.registry import adapter_class, source_type_for

_USAGE_FIELDS = (
    "actual_input_tokens",
    "actual_output_tokens",
    "actual_reasoning_tokens",
    "actual_total_tokens",
    "actual_cached_input_tokens",
    "actual_cache_creation_tokens",
    "actual_uncached_input_tokens",
    "actual_visible_output_tokens",
    "actual_tool_input_tokens",
    "actual_tool_output_tokens",
    "provider_total_tokens",
    "normalized_total_tokens",
)


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _sanitize_usage(usage: NormalizedUsage) -> dict[str, Any]:
    """Usage fields for the record; negative values become null (issue kept)."""
    values = usage.model_dump(include=set(_USAGE_FIELDS))
    for key, value in values.items():
        if isinstance(value, int) and value < 0:
            values[key] = None
    return values


def store_prompt_content(root: Path, sample: PlannedSample, prompt_hash: str) -> str:
    """Write the full request content (only called when privacy permits)."""
    path = root / "raw" / "prompt_content" / f"{prompt_hash}.json"
    if not path.exists():
        payload = sample.request.model_dump(mode="json", include={"system", "messages", "tools"})
        write_json(path, payload)
    return str(path.relative_to(root))


def build_records(
    samples: list[PlannedSample],
    events: list[dict[str, Any]],
    *,
    catalog: Catalog,
    prices: PriceTable,
    policy: RetentionPolicy,
    storage_root: Path,
    run_ids: dict[str, str] | None = None,
) -> tuple[list[RequestRecord], list[PreflightRecord], list[ObservationRecord]]:
    states = replay(events)
    requests: list[RequestRecord] = []
    preflights: list[PreflightRecord] = []
    observations: list[ObservationRecord] = []
    run_started = [e for e in events if e.get("event") == "run_started"]
    default_run = (
        run_started[0]["run_id"] if run_started else "00000000-0000-0000-0000-000000000000"
    )
    for sample in samples:
        sid = str(sample.sample_id)
        state = states.get(sid)
        if state is None or (
            state.attempts == 0 and state.status == SampleStatus.PENDING and state.preflight is None
        ):
            continue
        adapter_cls = adapter_class(sample.provider)
        family = adapter_cls.family
        source_type = source_type_for(sample.provider)
        caps = catalog.capabilities(sample.provider, sample.model)
        features = extract_request_features(
            sample.request, context_window_limit=caps.context_window
        )
        preflight = PreflightResult.model_validate(state.preflight) if state.preflight else None
        timestamp = _ts(state.first_dispatched_at) or (
            preflight.timestamp_utc if preflight else datetime.now(UTC)
        )
        run_id = uuid.UUID((state.run_ids[0] if state.run_ids else None) or default_run)
        collection_status = {
            SampleStatus.COMPLETED: CollectionStatus.COMPLETED,
            SampleStatus.FAILED: CollectionStatus.FAILED,
            SampleStatus.UNCERTAIN: CollectionStatus.UNCERTAIN,
            SampleStatus.SKIPPED: CollectionStatus.SKIPPED,
            SampleStatus.PENDING: CollectionStatus.PLANNED,
        }[state.status]
        storage_ref = (
            store_prompt_content(storage_root, sample, features.prompt_hash)
            if policy.store_prompt_text
            else None
        )
        requests.append(
            RequestRecord(
                sample_id=sample.sample_id,
                request_id=sample.request_id,
                run_id=run_id,
                timestamp_utc=timestamp,
                provider=sample.provider,
                api_surface=sample.request.api_surface or adapter_cls.api_surface,
                requested_model=sample.model,
                sdk_version=None,
                source_type=source_type,
                dataset_origin=f"experiment:{sample.experiment_id}",
                data_license=sample.data_license,
                consent_status=ConsentStatus.NOT_REQUIRED,
                collection_status=collection_status,
                experiment_id=sample.experiment_id,
                design_cell_id=sample.design_cell_id,
                variation_id=sample.variation_id,
                repetition_index=sample.repetition_index,
                generation_seed=sample.generation_seed,
                prompt_id=sample.prompt_id,
                prompt_template_id=sample.prompt_template_id,
                prompt_family_id=sample.prompt_family_id,
                prompt_category=sample.prompt_category,
                prompt_subcategory=sample.prompt_subcategory,
                prompt_language=sample.prompt_language,
                full_prompt_storage_ref=storage_ref,
                context_preflight_tokens=preflight.input_tokens if preflight else None,
                retain_until=policy.retain_until(source_type, timestamp),
                **features.model_dump(),
            )
        )
        if preflight is not None:
            preflights.append(
                PreflightRecord(
                    preflight_id=f"pre_{sample.sample_id.hex[:24]}_{preflight.method.value}",
                    sample_id=sample.sample_id,
                    request_id=sample.request_id,
                    provider=sample.provider,
                    requested_model=sample.model,
                    preflight_input_tokens=preflight.input_tokens,
                    preflight_method=preflight.method,
                    preflight_provider=preflight.preflight_provider,
                    tokenizer_name=preflight.tokenizer_name,
                    tokenizer_version=preflight.tokenizer_version,
                    preflight_timestamp_utc=preflight.timestamp_utc,
                    preflight_error=preflight.error
                    if preflight.input_tokens is not None
                    or preflight.method == PreflightMethod.UNAVAILABLE
                    else (preflight.error or "count unavailable"),
                    preflight_latency_ms=preflight.latency_ms,
                    preflight_includes=preflight.includes,
                    preflight_complete=preflight.complete,
                )
            )
        if state.status in {SampleStatus.COMPLETED, SampleStatus.FAILED, SampleStatus.UNCERTAIN}:
            observations.append(
                _observation(sample, state, family, source_type, run_id, prices, timestamp)
            )
    return requests, preflights, observations


def _observation(
    sample: PlannedSample,
    state: SampleState,
    family: str,
    source_type: SourceType,
    run_id: uuid.UUID,
    prices: PriceTable,
    request_time: datetime,
) -> ObservationRecord:
    common: dict[str, Any] = {
        "observation_id": f"obs_{sample.sample_id.hex[:24]}",
        "sample_id": sample.sample_id,
        "request_id": sample.request_id,
        "run_id": run_id,
        "provider": sample.provider,
        "requested_model": sample.model,
        "source_type": source_type,
        "dataset_origin": f"experiment:{sample.experiment_id}",
        "consent_status": ConsentStatus.NOT_REQUIRED,
        "retry_count": max(state.attempts - 1, 0),
        "rate_limit_encountered": any(f.get("category") == "rate_limit" for f in state.failures),
    }
    if state.status != SampleStatus.COMPLETED or state.response is None:
        last = state.failures[-1] if state.failures else {}
        empty = NORMALIZERS[family](None)
        return ObservationRecord(
            **common,
            input_usage_semantics=empty.input_usage_semantics,
            output_usage_semantics=empty.output_usage_semantics,
            label_quality=LabelQuality.UNAVAILABLE,
            usage_observation_source="none",
            response_status=ResponseStatus.UNCERTAIN
            if state.status == SampleStatus.UNCERTAIN
            else ResponseStatus.FAILED,
            error_type=last.get("category")
            or (
                "dispatched_without_recorded_outcome"
                if state.status == SampleStatus.UNCERTAIN
                else None
            ),
            error_code=str(last.get("status_code") or last.get("code") or "") or None,
            collected_at_utc=_ts(last.get("at")) or request_time,
        )
    response = ProviderResponse.model_validate(state.response)
    usage = NORMALIZERS[family](response.raw_usage)
    completed = response.status == ResponseStatus.COMPLETED
    quality = usage.label_quality(is_truncated=response.is_truncated, completed=completed)
    if source_type == SourceType.TEST_FIXTURE and quality != LabelQuality.UNAVAILABLE:
        quality = LabelQuality.SYNTHETIC
    entry = prices.lookup(
        sample.provider, sample.model, (response.completed_at_utc or request_time).date()
    )
    cost = compute_cost(family, entry, usage)
    dimensions: dict[str, Any] = {**usage.billing_dimensions, **cost.dimensions}
    if usage.issues:
        dimensions["accounting_issues"] = usage.issues
    if entry is not None and entry.is_fixture:
        dimensions["pricing_is_test_fixture"] = True
    rates = cost.rates
    write_rate = None
    if rates is not None:
        write_rate = (
            rates.cache_write_1h
            if dimensions.get("cache_write_1h_tokens")
            else rates.cache_write_5m
        )
    return ObservationRecord(
        **common,
        resolved_model=response.resolved_model,
        model_version=response.model_version,
        **_sanitize_usage(usage),
        input_usage_semantics=usage.input_usage_semantics,
        output_usage_semantics=usage.output_usage_semantics,
        reasoning_included_in_output=usage.reasoning_included_in_output,
        usage_iteration_count=usage.usage_iteration_count,
        label_quality=quality,
        usage_observation_source=usage.usage_observation_source,
        raw_usage=response.raw_usage,
        response_id=response.response_id,
        response_status=response.status,
        stop_reason=response.stop_reason,
        is_truncated=response.is_truncated,
        first_token_latency_ms=response.first_token_latency_ms,
        total_latency_ms=response.latency_ms,
        client_tool_call_count=response.client_tool_call_count,
        server_tool_call_count=response.server_tool_call_count,
        completed_at_utc=response.completed_at_utc,
        pricing_version=entry.pricing_version if entry else None,
        pricing_effective_date=entry.effective_from if entry else None,
        pricing_source=entry.source_url if entry else None,
        input_rate_usd_per_million=rates.input if rates else None,
        output_rate_usd_per_million=rates.output if rates else None,
        cache_read_rate_usd_per_million=rates.cache_read if rates else None,
        cache_write_rate_usd_per_million=write_rate,
        calculated_cost_usd=cost.calculated_cost_usd,
        provider_reported_cost_usd=Decimal(response.provider_reported_cost_usd)
        if response.provider_reported_cost_usd
        else None,
        cost_method=CostMethod.PRICE_TABLE_ESTIMATE
        if cost.calculated_cost_usd is not None
        else CostMethod.UNAVAILABLE,
        billing_dimensions=dimensions,
        collected_at_utc=response.completed_at_utc or request_time,
    )


def normalize_experiment(
    layout_root: str | Path,
    experiment_id: str,
    *,
    catalog: Catalog,
    prices: PriceTable,
    policy: RetentionPolicy,
) -> dict[str, int]:
    """Rebuild the normalized tables for one experiment from its raw layer."""
    from seshat.data.collectors.events import EventLog
    from seshat.data.io import load_jsonl
    from seshat.data.storage import StorageLayout

    layout = StorageLayout(layout_root)
    exp_dir = layout.experiment_dir(experiment_id)
    samples = load_jsonl(exp_dir / "plan.jsonl", PlannedSample)
    events = EventLog(exp_dir / "events.jsonl").read()
    requests, preflights, observations = build_records(
        samples, events, catalog=catalog, prices=prices, policy=policy, storage_root=layout.root
    )
    predictions = build_shadow_predictions(samples, events)
    source = f"experiment-{experiment_id}"
    layout.replace_source("requests", source, requests)
    layout.replace_source("preflight", source, preflights)
    layout.replace_source("observations", source, observations)
    layout.replace_source("predictions", source, predictions)
    counts = {
        "requests": len(requests),
        "preflight": len(preflights),
        "observations": len(observations),
        "shadow_predictions": len(predictions),
    }
    write_json(
        exp_dir / "normalization.json",
        {
            **counts,
            "normalized_at": datetime.now(UTC).isoformat(),
            "date": date.today().isoformat(),
        },
    )
    return counts


def build_shadow_predictions(
    samples: list[PlannedSample], events: list[dict[str, Any]]
) -> list[PredictionRecord]:
    """Canonical prediction rows from pre-dispatch ``shadow_prediction`` events.

    Only events whose forecast timestamp precedes the sample's first dispatch are
    accepted as live shadow predictions; anything else would not be pre-inference.
    """
    from seshat.data.collectors.events import EventType

    request_ids = {str(s.sample_id): s.request_id for s in samples}
    first_dispatch: dict[str, str] = {}
    shadows: dict[str, dict[str, Any]] = {}
    for event in events:
        sid = event.get("sample_id")
        if not sid:
            continue
        if event["event"] == EventType.DISPATCHED and sid not in first_dispatch:
            first_dispatch[sid] = str(event["at"])
        elif event["event"] == EventType.SHADOW_PREDICTION and sid not in shadows:
            shadows[sid] = event
    records: list[PredictionRecord] = []
    for sid, event in sorted(shadows.items()):
        if sid not in request_ids:
            continue
        p = event["prediction"]
        created = datetime.fromisoformat(str(p["created_at_utc"]))
        dispatched = first_dispatch.get(sid)
        if dispatched is not None and created >= datetime.fromisoformat(dispatched):
            continue  # not a pre-dispatch forecast: never recorded as live shadow
        request_id = request_ids[sid]
        cost = p.get("predicted_cost_usd")
        records.append(
            PredictionRecord(
                prediction_id=(
                    f"pred_{request_id}_{p['predictor_version']}_{p['calibration_version']}_shadow"
                ),
                request_id=request_id,
                created_at_utc=created,
                predictor_version=str(p["predictor_version"]),
                predicted_output_tokens=int(p["predicted_output_tokens"]),
                lower_bound=int(p["lower_bound"]),
                upper_bound=int(p["upper_bound"]),
                confidence=float(p["confidence"]),
                interval_calibrated=bool(p["interval_calibrated"]),
                prediction_mode=str(p["prediction_mode"]),
                evaluation_role=str(p["evaluation_role"]),
                base_predicted_output_tokens=p.get("base_predicted_output_tokens"),
                confidence_level=p.get("confidence_level"),
                lower_margin_tokens=p.get("lower_margin_tokens"),
                upper_margin_tokens=p.get("upper_margin_tokens"),
                confidence_margin_tokens=p.get("confidence_margin_tokens"),
                lower_margin_pct=p.get("lower_margin_pct"),
                upper_margin_pct=p.get("upper_margin_pct"),
                confidence_margin_pct=p.get("confidence_margin_pct"),
                predicted_total_tokens=p.get("predicted_total_tokens"),
                predicted_cost_usd=None if cost is None else Decimal(str(cost)),
                uncertainty_model_version=p.get("uncertainty_model_version"),
                calibration_version=p.get("calibration_version"),
                dataset_version=p.get("dataset_version"),
                feature_set_version=p.get("feature_set_version"),
                code_revision=p.get("code_revision"),
                expected_absolute_error_tokens=p.get("expected_absolute_error_tokens"),
                preflight_input_tokens=p.get("preflight_input_tokens"),
                preflight_method=p.get("preflight_source") or p.get("preflight_method"),
                pricing_version=p.get("pricing_version"),
                artifact_sha256=p.get("artifact_sha256"),
            )
        )
    return records
