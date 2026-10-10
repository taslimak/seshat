"""Canonical records from an *already executed* call (log imports and telemetry).

Shared by ``log_import`` (authorized log imports) and ``telemetry`` (consented API
telemetry), so both apply the same feature extraction, provider normalizers,
pricing, label rules and privacy policy as the collector.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from seshat.data.features.extract import extract_request_features
from seshat.data.io import write_json
from seshat.data.pricing import PriceTable, compute_cost
from seshat.data.privacy import RetentionPolicy
from seshat.data.schemas.enums import (
    REAL_USAGE_SOURCES,
    CollectionStatus,
    ConsentStatus,
    CostMethod,
    ResponseStatus,
    SourceType,
)
from seshat.data.schemas.records import ObservationRecord, RequestRecord, stable_uuid
from seshat.data.schemas.request import LLMRequest
from seshat.providers.catalog import Catalog
from seshat.providers.payloads import PARSERS
from seshat.providers.registry import adapter_class


def observed_sample_id(source_type: SourceType, provider: str, key: str) -> uuid.UUID:
    return stable_uuid("observed", source_type.value, provider, key)


def build_observed_records(
    *,
    request: LLMRequest,
    response_payload: dict[str, Any],
    request_id: str,
    timestamp_utc: datetime,
    source_type: SourceType,
    consent_status: ConsentStatus,
    dataset_origin: str,
    run_id: uuid.UUID,
    catalog: Catalog,
    prices: PriceTable,
    policy: RetentionPolicy,
    storage_root: Path,
    latency_ms: float | None = None,
    conversation_id: str | None = None,
    parent_request_id: str | None = None,
    prompt_category: str = "uncategorized",
    data_license: str | None = None,
    sdk_version: str | None = None,
) -> tuple[RequestRecord, ObservationRecord]:
    if source_type not in REAL_USAGE_SOURCES:
        raise ValueError(f"{source_type.value} cannot carry observed usage")
    if consent_status in {ConsentStatus.DENIED}:
        raise ValueError("consent denied: the call must not be recorded")
    family = adapter_class(request.provider).family
    try:
        caps = catalog.capabilities(request.provider, request.model)
        context_window = caps.context_window
    except ValueError:
        context_window = None  # unknown model: record, but without catalog limits
    features = extract_request_features(request, context_window_limit=context_window)
    key = str(response_payload.get("id") or response_payload.get("response_id") or request_id)
    sample_id = observed_sample_id(source_type, request.provider, key)
    storage_ref = None
    if policy.store_prompt_text and consent_status == ConsentStatus.GRANTED:
        path = storage_root / "raw" / "prompt_content" / f"{features.prompt_hash}.json"
        if not path.exists():
            write_json(
                path, request.model_dump(mode="json", include={"system", "messages", "tools"})
            )
        storage_ref = str(path.relative_to(storage_root))
    retain_until = policy.retain_until(source_type, timestamp_utc)
    request_record = RequestRecord(
        sample_id=sample_id,
        request_id=request_id,
        run_id=run_id,
        conversation_id=conversation_id,
        parent_request_id=parent_request_id,
        timestamp_utc=timestamp_utc,
        provider=request.provider,
        api_surface=request.api_surface or adapter_class(request.provider).api_surface,
        requested_model=request.model,
        sdk_version=sdk_version,
        source_type=source_type,
        dataset_origin=dataset_origin,
        data_license=data_license,
        consent_status=consent_status,
        collection_status=CollectionStatus.IMPORTED,
        prompt_id=f"obs_prompt_{features.prompt_hash[:16]}",
        prompt_family_id=None,
        prompt_category=prompt_category,
        full_prompt_storage_ref=storage_ref,
        retain_until=retain_until,
        **features.model_dump(),
    )
    response = PARSERS[family](response_payload, latency_ms=latency_ms)
    from seshat.providers.normalization import NORMALIZERS

    usage = NORMALIZERS[family](response.raw_usage)
    quality = usage.label_quality(
        is_truncated=response.is_truncated, completed=response.status == ResponseStatus.COMPLETED
    )
    entry = prices.lookup(request.provider, request.model, timestamp_utc.date())
    cost = compute_cost(family, entry, usage)
    dims: dict[str, Any] = {**usage.billing_dimensions, **cost.dimensions}
    if usage.issues:
        dims["accounting_issues"] = usage.issues
    values = usage.model_dump(
        include={
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
        }
    )
    values = {k: (None if isinstance(v, int) and v < 0 else v) for k, v in values.items()}
    rates = cost.rates
    observation = ObservationRecord(
        observation_id=f"obs_{sample_id.hex[:24]}",
        sample_id=sample_id,
        request_id=request_id,
        run_id=run_id,
        parent_request_id=parent_request_id,
        provider=request.provider,
        requested_model=request.model,
        resolved_model=response.resolved_model,
        model_version=response.model_version,
        source_type=source_type,
        dataset_origin=dataset_origin,
        consent_status=consent_status,
        **values,
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
        total_latency_ms=latency_ms,
        client_tool_call_count=response.client_tool_call_count,
        server_tool_call_count=response.server_tool_call_count,
        completed_at_utc=timestamp_utc,
        pricing_version=entry.pricing_version if entry else None,
        pricing_effective_date=entry.effective_from if entry else None,
        pricing_source=entry.source_url if entry else None,
        input_rate_usd_per_million=rates.input if rates else None,
        output_rate_usd_per_million=rates.output if rates else None,
        cache_read_rate_usd_per_million=rates.cache_read if rates else None,
        cache_write_rate_usd_per_million=rates.cache_write_5m if rates else None,
        calculated_cost_usd=cost.calculated_cost_usd,
        provider_reported_cost_usd=Decimal(str(response_payload["cost_usd"]))
        if response_payload.get("cost_usd") is not None
        else None,
        cost_method=CostMethod.PRICE_TABLE_ESTIMATE
        if cost.calculated_cost_usd is not None
        else CostMethod.UNAVAILABLE,
        billing_dimensions=dims,
        collected_at_utc=timestamp_utc,
        retain_until=retain_until,
    )
    return request_record, observation
