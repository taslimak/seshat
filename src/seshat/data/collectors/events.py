"""Append-only collection event log (the raw layer's source of truth).

Every paid dispatch is logged (and fsynced) **before** the request is sent, and its
outcome afterwards. On restart, replaying the log tells exactly which samples are
done, which failed, and which were dispatched without a recorded outcome. The last
group is marked ``uncertain`` and is never re-sent automatically, because the
provider may already have executed and billed it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

log = logging.getLogger("seshat.collector")


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    PREFLIGHT = "preflight"
    DISPATCHED = "request_dispatched"
    RESPONSE = "response_received"
    FAILED = "request_failed"
    UNCERTAIN = "sample_uncertain"
    SKIPPED = "sample_skipped"
    RUN_FINISHED = "run_finished"


class SampleStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    SKIPPED = "skipped"


class EventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event_type: EventType, **fields: Any) -> dict[str, Any]:
        event = {"event": event_type.value, "at": datetime.now(UTC).isoformat(), **fields}
        line = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if number == len(lines):
                    log.warning("ignoring truncated final event line %s in %s", number, self.path)
                    continue
                raise ValueError(f"Corrupt event log {self.path}:{number}") from None
        return events


@dataclass
class SampleState:
    sample_id: str
    status: SampleStatus = SampleStatus.PENDING
    attempts: int = 0
    open_attempt: int | None = None
    preflight: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    reserved_cost_usd: float = 0.0
    actual_cost_usd: float | None = None
    first_dispatched_at: str | None = None
    run_ids: list[str] = field(default_factory=list)
    skip_reason: str | None = None


def replay(events: list[dict[str, Any]]) -> dict[str, SampleState]:
    """Fold events into per-sample state (later events win)."""
    states: dict[str, SampleState] = defaultdict(lambda: SampleState(sample_id=""))
    for event in events:
        sample_id = event.get("sample_id")
        if not sample_id:
            continue
        state = states[sample_id]
        state.sample_id = sample_id
        kind = event["event"]
        if kind == EventType.PREFLIGHT:
            state.preflight = event["result"]
        elif kind == EventType.DISPATCHED:
            state.attempts = max(state.attempts, int(event["attempt"]))
            state.open_attempt = int(event["attempt"])
            state.reserved_cost_usd = float(event.get("reserved_cost_usd") or 0.0)
            state.first_dispatched_at = state.first_dispatched_at or event["at"]
            if event.get("run_id") and event["run_id"] not in state.run_ids:
                state.run_ids.append(event["run_id"])
        elif kind == EventType.RESPONSE:
            state.open_attempt = None
            state.status = SampleStatus.COMPLETED
            state.response = event["response"]
            state.actual_cost_usd = event.get("calculated_cost_usd")
        elif kind == EventType.FAILED:
            state.open_attempt = None
            state.failures.append(event)
            if event.get("final"):
                state.status = (
                    SampleStatus.UNCERTAIN
                    if event.get("outcome_uncertain")
                    else SampleStatus.FAILED
                )
        elif kind == EventType.UNCERTAIN:
            state.open_attempt = None
            state.status = SampleStatus.UNCERTAIN
        elif kind == EventType.SKIPPED:
            if state.status == SampleStatus.PENDING:
                state.status = SampleStatus.SKIPPED
                state.skip_reason = event.get("reason")
    return dict(states)
