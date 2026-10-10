"""Consented API telemetry (Method 2): opt-in usage capture around real calls.

Disabled by default. When enabled, ``start(request)`` captures request-time features
*before* execution; ``finish(...)`` records the observed usage afterwards. Raw prompt
text is never kept unless ``capture_prompt_text`` is set *and* the caller passes
``consent_status=granted``. Records are buffered and written as a normalized batch
by ``flush()``.

Usage::

    recorder = TelemetryRecorder(root="data", enabled=True)
    handle = recorder.start(request, request_id="r1", consent_status=ConsentStatus.GRANTED)
    response = client.messages.create(...)
    recorder.finish(handle, response.model_dump(mode="json"), latency_ms=820.0)
    recorder.flush()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from seshat.data.collectors.observed import build_observed_records
from seshat.data.pricing import PriceTable
from seshat.data.privacy import RetentionPolicy
from seshat.data.schemas.enums import ConsentStatus, SourceType
from seshat.data.schemas.records import ObservationRecord, RequestRecord
from seshat.data.schemas.request import LLMRequest
from seshat.data.storage import StorageLayout
from seshat.providers.catalog import Catalog, load_catalog


@dataclass
class TelemetryHandle:
    request: LLMRequest
    request_id: str
    started_at: datetime
    consent_status: ConsentStatus
    prompt_category: str
    conversation_id: str | None
    parent_request_id: str | None


class TelemetryRecorder:
    def __init__(
        self,
        root: str | Path = "data",
        *,
        enabled: bool = False,
        capture_prompt_text: bool = False,
        retention_days: int = 90,
        dataset_origin: str = "seshat-api",
        catalog: Catalog | None = None,
        prices: PriceTable | None = None,
    ) -> None:
        self.enabled = enabled
        self.layout = StorageLayout(root)
        self.policy = RetentionPolicy(
            telemetry_days=retention_days, store_prompt_text=capture_prompt_text
        )
        self.dataset_origin = dataset_origin
        self.catalog = catalog or load_catalog()
        self.prices = prices or PriceTable.load()
        self.run_id = uuid.uuid4()
        self._requests: list[RequestRecord] = []
        self._observations: list[ObservationRecord] = []

    def start(
        self,
        request: LLMRequest,
        *,
        request_id: str,
        consent_status: ConsentStatus = ConsentStatus.UNKNOWN,
        prompt_category: str = "uncategorized",
        conversation_id: str | None = None,
        parent_request_id: str | None = None,
    ) -> TelemetryHandle | None:
        if not self.enabled or consent_status == ConsentStatus.DENIED:
            return None
        return TelemetryHandle(
            request,
            request_id,
            datetime.now(UTC),
            consent_status,
            prompt_category,
            conversation_id,
            parent_request_id,
        )

    def finish(
        self,
        handle: TelemetryHandle | None,
        response_payload: dict[str, Any],
        *,
        latency_ms: float | None = None,
    ) -> None:
        if handle is None:
            return
        request_record, observation = build_observed_records(
            request=handle.request,
            response_payload=response_payload,
            request_id=handle.request_id,
            timestamp_utc=handle.started_at,
            source_type=SourceType.CONSENTED_API_TELEMETRY,
            consent_status=handle.consent_status,
            dataset_origin=self.dataset_origin,
            run_id=self.run_id,
            catalog=self.catalog,
            prices=self.prices,
            policy=self.policy,
            storage_root=self.layout.root,
            latency_ms=latency_ms,
            conversation_id=handle.conversation_id,
            parent_request_id=handle.parent_request_id,
            prompt_category=handle.prompt_category,
        )
        self._requests.append(request_record)
        self._observations.append(observation)

    def flush(self) -> int:
        if not self._requests:
            return 0
        source = f"telemetry-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.layout.replace_source("requests", source, self._requests)
        self.layout.replace_source("observations", source, self._observations)
        count = len(self._requests)
        self._requests, self._observations = [], []
        return count
