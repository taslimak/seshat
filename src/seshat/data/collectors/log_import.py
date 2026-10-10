"""Authorized log imports (Method 3): usage logs -> canonical records.

Input: JSONL, one executed call per line::

    {"request_id": "...", "timestamp_utc": "2026-10-01T12:00:00Z",
     "request": {<LLMRequest: provider, model, system, messages, tools, params>},
     "response": {<provider SDK payload, e.g. message/response object incl. usage>},
     "latency_ms": 1234.5, "conversation_id": null, "parent_request_id": null,
     "prompt_category": "support", "consent_status": "granted"}

Rules: an ``authorization_ref`` (who approved the import) is mandatory; rows are
validated and deduplicated deterministically (sample_id = UUIDv5 of source, provider
and response id); rejected rows go to a redacted rejection report. Prompt text is
used to compute features and then dropped unless the privacy policy allows storage
*and* the row carries ``consent_status=granted``. Public text datasets without usage
belong in ``seshat-data prompts import`` instead: they stay unlabeled until executed.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from seshat.data.collectors.observed import build_observed_records, observed_sample_id
from seshat.data.io import write_json, write_jsonl
from seshat.data.pricing import PriceTable
from seshat.data.privacy import RetentionPolicy, redact_text
from seshat.data.schemas.enums import REAL_USAGE_SOURCES, ConsentStatus, SourceType
from seshat.data.schemas.request import LLMRequest
from seshat.data.storage import StorageLayout
from seshat.providers.catalog import Catalog


@dataclass
class LogImportReport:
    batch_id: str
    read: int = 0
    imported: int = 0
    duplicates: int = 0
    rejected: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    rejection_report: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "read": self.read,
            "imported": self.imported,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "reasons": dict(sorted(self.reasons.items())),
            "rejection_report": self.rejection_report,
        }


def _sanitize(row: dict[str, Any]) -> dict[str, Any]:
    """Row without prompt/response content, safe for the raw import record."""
    copy: dict[str, Any] = json.loads(json.dumps(row, default=str))
    request = copy.get("request") or {}
    for key in ("system", "messages"):
        request.pop(key, None)
    response = copy.get("response") or {}
    for key in ("content", "output", "candidates"):
        response.pop(key, None)
    return copy


def import_usage_log(
    path: str | Path,
    *,
    root: str | Path,
    source_type: SourceType,
    authorization_ref: str,
    dataset_origin: str,
    catalog: Catalog,
    prices: PriceTable,
    policy: RetentionPolicy,
    data_license: str | None = None,
) -> LogImportReport:
    if source_type not in REAL_USAGE_SOURCES:
        raise ValueError(
            f"usage-log import requires a real-usage source type, not {source_type.value}"
        )
    if not authorization_ref.strip():
        raise ValueError("authorization_ref is required for log imports")
    layout = StorageLayout(root)
    batch_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_id = uuid.uuid4()
    batch_dir = layout.raw / "imports" / batch_id
    report = LogImportReport(batch_id=batch_id)
    existing = (
        set(layout.read_frame("observations")["sample_id"])
        if layout.parts("observations")
        else set()
    )
    seen: set[str] = set()
    requests, observations, raw_rows, rejections = [], [], [], []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        report.read += 1

        def reject(reason: str, detail: str, payload: Any = None, line: int = number) -> None:
            report.rejected += 1
            report.reasons[reason] += 1
            rejections.append(
                {
                    "line": line,
                    "reason": reason,
                    "detail": redact_text(detail, max_length=1000),
                    "payload": _sanitize(payload) if isinstance(payload, dict) else None,
                }
            )

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            reject("invalid_json", str(exc))
            continue
        if not isinstance(row, dict) or not isinstance(row.get("response"), dict):
            reject(
                "invalid_record",
                "row must be an object with a 'response' payload",
                row if isinstance(row, dict) else None,
            )
            continue
        try:
            request = LLMRequest.model_validate(row.get("request") or {})
            timestamp = datetime.fromisoformat(str(row["timestamp_utc"]).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("timestamp_utc must include a timezone")
            consent = ConsentStatus(row.get("consent_status", "unknown"))
        except (ValidationError, KeyError, ValueError) as exc:
            reject("schema_error", str(exc), row)
            continue
        key = str(
            row["response"].get("id") or row["response"].get("response_id") or row.get("request_id")
        )
        sample_id = str(observed_sample_id(source_type, request.provider, key))
        if sample_id in existing or sample_id in seen:
            report.duplicates += 1
            continue
        try:
            request_record, observation = build_observed_records(
                request=request,
                response_payload=row["response"],
                request_id=str(row.get("request_id") or key),
                timestamp_utc=timestamp,
                source_type=source_type,
                consent_status=consent,
                dataset_origin=dataset_origin,
                run_id=run_id,
                catalog=catalog,
                prices=prices,
                policy=policy,
                storage_root=layout.root,
                latency_ms=row.get("latency_ms"),
                conversation_id=row.get("conversation_id"),
                parent_request_id=row.get("parent_request_id"),
                prompt_category=row.get("prompt_category") or "uncategorized",
                data_license=data_license,
            )
        except (ValidationError, ValueError) as exc:
            reject("normalization_error", str(exc), row)
            continue
        seen.add(sample_id)
        requests.append(request_record)
        observations.append(observation)
        raw_rows.append(_sanitize(row))
    if requests:
        source = f"import-{batch_id}"
        layout.replace_source("requests", source, requests)
        layout.replace_source("observations", source, observations)
        write_jsonl(batch_dir / "rows.sanitized.jsonl", raw_rows)
    report.imported = len(requests)
    if rejections:
        write_jsonl(batch_dir / "rejections.jsonl", rejections)
        report.rejection_report = str(batch_dir / "rejections.jsonl")
    write_json(
        batch_dir / "import_report.json",
        {
            **report.to_dict(),
            "authorization_ref": authorization_ref,
            "source_type": source_type.value,
            "input": str(path),
        },
    )
    return report
