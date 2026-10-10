"""Ingestion of versioned records into the ``DatasetStore``.

For each incoming record, in order:

1. **Provenance**: required on request-feature and observed-usage records. A
   caller-supplied default may fill it in; nothing is ever assumed to be real.
2. **Privacy**: if prompt text is present, the hash and request-time features are
   computed from it, then the text is dropped unless the retention policy allows
   keeping it.
3. **Schema validation** against the versioned Pydantic model.
4. **Retention**: ``retain_until`` is set from the policy when absent.
5. **Deduplication**: an identical record ID + content is skipped (idempotent
   re-ingestion); the same ID with different content, or a provider request ID
   already seen, is quarantined.
6. **Quality rules** (observed usage only): ``seshat.data.validate``; QUARANTINE and
   REJECT decisions are quarantined, PASS/WARNING are stored with their flags.

Nothing is silently dropped: every skipped record is either counted as an exact
duplicate or written, sanitized, to the quarantine file with its reason.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from seshat.data.io import load_jsonl, write_jsonl
from seshat.data.models import PromptRecord, QualityDecision, RawObservation
from seshat.data.normalize import observation_provenance, select_final_attempts
from seshat.data.privacy import RetentionPolicy, hash_prompt, redact_text
from seshat.data.provenance import PIPELINE_VERSION, Provenance
from seshat.data.records import (
    ID_COLUMNS,
    RECORD_MODELS,
    ObservedUsageRecord,
    PromptFeatureValues,
    RecordType,
    RequestFeatureRecord,
    record_hash,
    to_row,
)
from seshat.data.store import DatasetStore
from seshat.data.validate import OBSERVED_USAGE_REQUIRED_FIELDS, validate_dataframe

_STORED_DECISIONS = {QualityDecision.PASS.value, QualityDecision.WARNING.value}


@dataclass
class IngestReport:
    record_type: str
    batch_id: str
    source: str
    read: int = 0
    stored: int = 0
    duplicates: int = 0
    quarantined: int = 0
    prompt_text_dropped: int = 0
    quarantine_reasons: Counter[str] = field(default_factory=Counter)
    quality_decisions: Counter[str] = field(default_factory=Counter)
    batch_path: str | None = None
    quarantine_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "batch_id": self.batch_id,
            "source": self.source,
            "read": self.read,
            "stored": self.stored,
            "duplicates": self.duplicates,
            "quarantined": self.quarantined,
            "prompt_text_dropped": self.prompt_text_dropped,
            "quarantine_reasons": dict(sorted(self.quarantine_reasons.items())),
            "quality_decisions": dict(sorted(self.quality_decisions.items())),
            "batch_path": self.batch_path,
            "quarantine_path": self.quarantine_path,
        }


@dataclass(frozen=True)
class _InvalidPayload:
    message: str


def _sanitize(value: Any, *, keep_prompt_text: bool) -> Any:
    """Copy of a payload safe to write to quarantine (no prompt text, redacted)."""
    if isinstance(value, dict):
        return {
            key: _sanitize(item, keep_prompt_text=keep_prompt_text)
            for key, item in value.items()
            if keep_prompt_text or key not in {"prompt_text", "text", "prompt"}
        }
    if isinstance(value, list):
        return [_sanitize(item, keep_prompt_text=keep_prompt_text) for item in value]
    if isinstance(value, str):
        return redact_text(value, max_length=2000)
    return value


def _batch_id(source_name: str, now: datetime) -> str:
    digest = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:8]
    return f"{now.strftime('%Y%m%dT%H%M%S%f')}-{digest}"


def _reference_time(record: BaseModel) -> datetime:
    value = getattr(record, "created_at", None) or getattr(record, "observed_at", None)
    assert isinstance(value, datetime)
    return value


def ingest_records(
    store: DatasetStore,
    record_type: RecordType,
    payloads: Iterable[tuple[str, Any]],
    *,
    source_name: str,
    default_provenance: Provenance | None = None,
    policy: RetentionPolicy | None = None,
    quarantine_dir: str | Path | None = None,
    hash_key: str | None = None,
    now: datetime | None = None,
) -> IngestReport:
    """Validate, deduplicate and store ``(location, payload)`` pairs as one batch."""
    policy = policy or RetentionPolicy()
    now = now or datetime.now(UTC)
    model = RECORD_MODELS[record_type]
    id_column = ID_COLUMNS[record_type]
    report = IngestReport(record_type.value, _batch_id(source_name, now), source_name)
    quarantine: list[dict[str, Any]] = []

    def reject(location: str, reason: str, payload: Any, **extra: Any) -> None:
        report.quarantined += 1
        report.quarantine_reasons[reason] += 1
        quarantine.append(
            {
                "batch_id": report.batch_id,
                "location": location,
                "reason": reason,
                **extra,
                "payload": _sanitize(payload, keep_prompt_text=policy.store_prompt_text),
            }
        )

    stored_hashes = store.record_hashes(record_type)
    seen_hashes: dict[str, str] = {}
    stored_request_ids = (
        store.provider_request_ids() if record_type == RecordType.OBSERVED_USAGE else {}
    )
    seen_request_ids: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    row_payloads: list[tuple[str, Any]] = []

    for location, raw in payloads:
        report.read += 1
        if isinstance(raw, _InvalidPayload):
            reject(location, "invalid_json", None, detail=raw.message)
            continue
        if not isinstance(raw, dict):
            reject(location, "invalid_record", raw, detail="record is not a JSON object")
            continue
        payload = dict(raw)
        payload.setdefault("record_type", record_type.value)
        if payload["record_type"] != record_type.value:
            reject(location, "wrong_record_type", payload)
            continue
        if record_type != RecordType.PREDICTION and payload.get("provenance") is None:
            if default_provenance is None:
                reject(location, "missing_provenance", payload)
                continue
            payload["provenance"] = default_provenance.model_dump(mode="json")
        if record_type == RecordType.REQUEST_FEATURES and payload.get("prompt_text"):
            text = str(payload["prompt_text"])
            if not payload.get("prompt_hash"):
                digest, scheme = hash_prompt(text, hash_key)
                payload["prompt_hash"] = digest
                payload["prompt_hash_scheme"] = scheme.value
            if not payload.get("features"):
                payload["features"] = PromptFeatureValues.from_prompt(text).model_dump()
            if not policy.store_prompt_text:
                payload["prompt_text"] = None
                report.prompt_text_dropped += 1
        try:
            record = model.model_validate(payload)
        except ValidationError as exc:
            errors = [
                {"loc": ".".join(str(part) for part in error["loc"]), "msg": error["msg"]}
                for error in exc.errors(include_input=False, include_url=False)
            ]
            reject(location, "schema_error", payload, errors=errors)
            continue
        provenance = getattr(record, "provenance", None)
        if provenance is not None and getattr(record, "retain_until", None) is None:
            record = record.model_copy(
                update={"retain_until": policy.retain_until(provenance, _reference_time(record))}
            )
        record_id = str(getattr(record, id_column))
        content_hash = record_hash(record)
        previous = stored_hashes.get(record_id) or seen_hashes.get(record_id)
        if previous is not None:
            if previous == content_hash:
                report.duplicates += 1
            else:
                reject(location, "conflicting_duplicate_id", payload, record_id=record_id)
            continue
        if isinstance(record, ObservedUsageRecord) and record.provider_request_id:
            owner = stored_request_ids.get(record.provider_request_id) or seen_request_ids.get(
                record.provider_request_id
            )
            if owner is not None and owner != record_id:
                reject(location, "duplicate_provider_request_id", payload, record_id=record_id)
                continue
            seen_request_ids[record.provider_request_id] = record_id
        seen_hashes[record_id] = content_hash
        rows.append(to_row(record))
        row_payloads.append((location, payload))

    if record_type == RecordType.OBSERVED_USAGE and rows:
        checked = validate_dataframe(
            pd.DataFrame(rows), required_fields=OBSERVED_USAGE_REQUIRED_FIELDS
        )
        kept: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            decision = str(checked.at[index, "quality_decision"])
            flags = str(checked.at[index, "quality_flags"])
            if decision in _STORED_DECISIONS:
                kept.append({**row, "quality_decision": decision, "quality_flags": flags})
                report.quality_decisions[decision] += 1
            else:
                location, payload = row_payloads[index]
                reject(
                    location,
                    f"quality_{decision.lower()}",
                    payload,
                    record_id=row["observation_id"],
                    flags=json.loads(flags),
                )
        rows = kept

    batch_path = store.write_batch(record_type, rows, report.batch_id)
    report.stored = len(rows)
    report.batch_path = str(batch_path) if batch_path else None
    if quarantine:
        target_dir = Path(quarantine_dir) if quarantine_dir else store.root.parent / "quarantine"
        quarantine_path = target_dir / f"{report.batch_id}-{record_type.value}.jsonl"
        write_jsonl(quarantine_path, quarantine)
        report.quarantine_path = str(quarantine_path)
    store.log(
        {
            "event": "ingest",
            **report.to_dict(),
            "pipeline_version": PIPELINE_VERSION,
            "logged_at": now.isoformat(),
        }
    )
    return report


def _iter_jsonl(path: Path) -> Iterator[tuple[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path.name}:{line_number}"
            try:
                yield location, json.loads(line)
            except json.JSONDecodeError as exc:
                yield location, _InvalidPayload(str(exc))


def ingest_jsonl(
    store: DatasetStore,
    record_type: RecordType,
    path: str | Path,
    **kwargs: Any,
) -> IngestReport:
    """Ingest a JSONL file of versioned records. See ``ingest_records`` for options."""
    source = Path(path)
    return ingest_records(
        store, record_type, _iter_jsonl(source), source_name=str(source), **kwargs
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def experiment_records(
    raw_path: str | Path,
    prompts: Iterable[PromptRecord],
    *,
    policy: RetentionPolicy | None = None,
    hash_key: str | None = None,
) -> tuple[list[RequestFeatureRecord], list[ObservedUsageRecord]]:
    """Convert a controlled-experiment raw JSONL file into versioned records.

    Retried runs collapse to their final attempt. ``request_id`` is the run ID, and
    provenance comes from the raw record (or the adapter, for legacy files), so mock
    runs are always labelled synthetic.
    """
    prompt_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    requests: list[RequestFeatureRecord] = []
    observations: list[ObservedUsageRecord] = []
    for raw, _ in select_final_attempts(load_jsonl(raw_path, RawObservation)):
        prompt = prompt_by_id.get(raw.prompt_id)
        if prompt is None:
            raise ValueError(f"Raw observation references unknown prompt_id {raw.prompt_id}")
        provenance = observation_provenance(raw)
        timestamp = _aware(raw.timestamp)
        requests.append(
            RequestFeatureRecord.from_prompt(
                request_id=raw.run_id,
                prompt=prompt.text,
                provider=raw.provider,
                model=raw.model_requested,
                provenance=provenance,
                created_at=timestamp,
                policy=policy,
                hash_key=hash_key,
                effort=raw.effort,
                max_output_tokens=raw.max_output_tokens,
                temperature=raw.temperature,
                category=raw.category,
                prompt_family_id=raw.prompt_family_id,
            )
        )
        observations.append(
            ObservedUsageRecord(
                observation_id=raw.observation_id,
                request_id=raw.run_id,
                observed_at=timestamp,
                provider=raw.provider,
                model_requested=raw.model_requested,
                model_resolved=raw.model_resolved,
                provider_request_id=raw.provider_request_id,
                status=raw.status,
                usage=raw.normalized_usage,
                finish_reason=raw.finish_reason,
                is_truncated=raw.is_truncated,
                is_refusal=raw.is_refusal,
                latency_ms=raw.latency_ms,
                response_char_count=raw.response_char_count,
                attempt_count=raw.attempt_count,
                error_type=raw.error_type,
                error_message=raw.error_message,
                provenance=provenance,
                metadata={
                    "experiment_id": raw.experiment_id,
                    "condition_id": raw.condition_id,
                    "prompt_id": raw.prompt_id,
                    "repetition": raw.repetition,
                },
            )
        )
    return requests, observations


def ingest_experiment(
    store: DatasetStore,
    raw_path: str | Path,
    prompts: Iterable[PromptRecord],
    *,
    policy: RetentionPolicy | None = None,
    hash_key: str | None = None,
    quarantine_dir: str | Path | None = None,
    now: datetime | None = None,
) -> tuple[IngestReport, IngestReport]:
    """Load a controlled-experiment raw file into the store (requests, then usage)."""
    requests, observations = experiment_records(
        raw_path, prompts, policy=policy, hash_key=hash_key
    )
    source = str(raw_path)
    request_report = ingest_records(
        store,
        RecordType.REQUEST_FEATURES,
        ((f"{source}#request:{index}", item.model_dump(mode="json"))
         for index, item in enumerate(requests)),
        source_name=source,
        policy=policy,
        quarantine_dir=quarantine_dir,
        hash_key=hash_key,
        now=now,
    )
    usage_report = ingest_records(
        store,
        RecordType.OBSERVED_USAGE,
        ((f"{source}#usage:{index}", item.model_dump(mode="json"))
         for index, item in enumerate(observations)),
        source_name=source,
        policy=policy,
        quarantine_dir=quarantine_dir,
        hash_key=hash_key,
        now=now,
    )
    return request_report, usage_report
