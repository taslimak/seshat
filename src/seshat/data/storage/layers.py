"""Physical storage layers.

::

    <root>/raw/<experiment_id>/events.jsonl        append-only collection events (+ raw usage)
    <root>/raw/<experiment_id>/plan.jsonl          planned samples (requests) for the experiment
    <root>/raw/<experiment_id>/design_report.json
    <root>/raw/<experiment_id>/runs/<run_id>.log.jsonl   structured collector log
    <root>/raw/imports/<batch_id>/...              raw import inputs + rejection reports
    <root>/raw/prompt_content/<hash>.json          prompt text, ONLY if privacy allows
    <root>/normalized/<table>/date=YYYY-MM-DD/provider=<p>/part-<source>.parquet
    <root>/normalized/_validation/                 validation decisions + quality reports
    <root>/training/<dataset_version>/             point-in-time feature/label datasets
    <root>/exports/                                CSV exports + data dictionary

Tables: ``requests`` (RequestRecord), ``preflight`` (PreflightRecord),
``observations`` (ObservationRecord), ``predictions`` (PredictionRecord).
Partition files are named by their *source* (an experiment or an import batch).
Re-normalizing an experiment atomically replaces that experiment's parts, so the
normalized layer is always a deterministic function of the raw layer.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from seshat.data.schemas.arrow import arrow_schema, from_table, to_table
from seshat.data.schemas.records import (
    ObservationRecord,
    PredictionRecord,
    PreflightRecord,
    RequestRecord,
)

T = TypeVar("T", bound=BaseModel)

TABLES: dict[str, type[BaseModel]] = {
    "requests": RequestRecord,
    "preflight": PreflightRecord,
    "observations": ObservationRecord,
    "predictions": PredictionRecord,
}
_TIME_FIELD = {
    "requests": "timestamp_utc",
    "preflight": "preflight_timestamp_utc",
    "observations": "collected_at_utc",
    "predictions": "created_at_utc",
}


class StorageLayout:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def normalized(self) -> Path:
        return self.root / "normalized"

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def validation(self) -> Path:
        return self.normalized / "_validation"

    def experiment_dir(self, experiment_id: str) -> Path:
        return self.raw / experiment_id

    def table_dir(self, table: str) -> Path:
        if table not in TABLES:
            raise ValueError(f"Unknown table {table!r}; known: {sorted(TABLES)}")
        return self.normalized / table

    # -- writing ------------------------------------------------------------------
    def replace_source(self, table: str, source: str, records: Sequence[BaseModel]) -> list[Path]:
        """Atomically replace every partition file written by ``source`` for ``table``."""
        model = TABLES[table]
        name = f"part-{_safe(source)}.parquet"
        stale = set(self.table_dir(table).glob(f"date=*/provider=*/{name}"))
        groups: dict[tuple[str, str], list[BaseModel]] = defaultdict(list)
        for record in records:
            if type(record) is not model:
                raise TypeError(f"{table} expects {model.__name__}")
            stamp = getattr(record, _TIME_FIELD[table], None)
            day = stamp.date().isoformat() if isinstance(stamp, datetime) else "unknown"
            provider = getattr(record, "provider", None) or "unknown"
            groups[(day, str(provider))].append(record)
        written: list[Path] = []
        for (day, provider), items in sorted(groups.items()):
            path = self.table_dir(table) / f"date={day}" / f"provider={_safe(provider)}" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            pq.write_table(to_table(items, model), tmp, compression="zstd")
            os.replace(tmp, path)
            written.append(path)
            stale.discard(path)
        for path in stale:
            path.unlink()
        return written

    # -- reading ------------------------------------------------------------------
    def parts(self, table: str) -> list[Path]:
        return sorted(self.table_dir(table).glob("date=*/provider=*/part-*.parquet"))

    def read_table(self, table: str) -> pa.Table:
        """All partitions of a table as one Arrow table with the canonical schema."""
        schema = arrow_schema(TABLES[table])
        tables = [pq.read_table(path, schema=schema) for path in self.parts(table)]
        return pa.concat_tables(tables) if tables else schema.empty_table()

    def read_records(self, table: str, model: type[T]) -> list[T]:
        return from_table(self.read_table(table), model)

    def read_frame(self, table: str) -> Any:
        return self.read_table(table).to_pandas()

    # -- retention ----------------------------------------------------------------
    def purge_expired(self, as_of: datetime) -> dict[str, int]:
        """Delete request/observation rows past ``retain_until`` (+ their preflight and
        predictions), rewriting affected parts atomically. Returns rows removed."""
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        cutoff = pa.scalar(as_of, type=pa.timestamp("us", tz="UTC"))
        removed: dict[str, int] = {}
        expired_samples: set[str] = set()
        expired_requests: set[str] = set()
        import pyarrow.compute as pc

        for table in ("requests", "observations"):
            schema = arrow_schema(TABLES[table])
            count = 0
            for path in self.parts(table):
                data = pq.read_table(path, schema=schema)
                mask = pc.fill_null(pc.less_equal(data["retain_until"], cutoff), False)
                expired = int(pc.sum(mask).as_py() or 0)
                if not expired:
                    continue
                gone = data.filter(mask)
                expired_samples.update(gone["sample_id"].to_pylist())
                expired_requests.update(gone["request_id"].to_pylist())
                self._rewrite(path, data.filter(pc.invert(mask)))
                count += expired
            removed[table] = count
        for table, column, ids in (
            ("preflight", "sample_id", expired_samples),
            ("predictions", "request_id", expired_requests),
        ):
            schema = arrow_schema(TABLES[table])
            count = 0
            for path in self.parts(table):
                data = pq.read_table(path, schema=schema)
                mask = pc.is_in(data[column], value_set=pa.array(sorted(ids), type=pa.string()))
                hits = int(pc.sum(mask).as_py() or 0)
                if hits:
                    self._rewrite(path, data.filter(pc.invert(mask)))
                    count += hits
            removed[table] = count
        return removed

    @staticmethod
    def _rewrite(path: Path, data: pa.Table) -> None:
        if data.num_rows == 0:
            path.unlink()
            return
        tmp = path.with_name(f".{path.name}.tmp")
        pq.write_table(data, tmp, compression="zstd")
        os.replace(tmp, path)


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def iter_tables() -> Iterable[str]:
    return iter(TABLES)
