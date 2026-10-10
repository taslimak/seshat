"""File-based Parquet store for versioned request-feature, prediction and usage records.

Layout::

    <root>/
      request_features/part-<batch_id>.parquet
      prediction/part-<batch_id>.parquet
      observed_usage/part-<batch_id>.parquet
      _ingestion_log.jsonl          # one line per batch / purge (counts only, no content)

Batches are written atomically and never modified, except by ``purge_expired``,
which rewrites a part without its expired rows to honour retention. Rows are the
flattened form produced by ``seshat.data.records.to_row``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.io import append_jsonl, write_parquet
from seshat.data.records import ID_COLUMNS, RecordType

DATETIME_COLUMNS = ("created_at", "observed_at", "retain_until")
LOG_FILE = "_ingestion_log.jsonl"


class DatasetStore:
    """Append-only store of record batches, one directory per record type."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def table_dir(self, record_type: RecordType) -> Path:
        return self.root / record_type.value

    def parts(self, record_type: RecordType) -> list[Path]:
        directory = self.table_dir(record_type)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("part-*.parquet"))

    def read(self, record_type: RecordType) -> pd.DataFrame:
        """All stored rows of one record type (empty frame when none)."""
        frames = [pd.read_parquet(path) for path in self.parts(record_type)]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True)
        for column in DATETIME_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    def record_hashes(self, record_type: RecordType) -> dict[str, str]:
        """``{record_id: record_hash}`` for deduplication."""
        frame = self.read(record_type)
        if frame.empty:
            return {}
        id_column = ID_COLUMNS[record_type]
        ids = frame[id_column].astype(str)
        return dict(zip(ids, frame["record_hash"].astype(str), strict=True))

    def provider_request_ids(self) -> dict[str, str]:
        """``{provider_request_id: observation_id}`` across stored observations."""
        frame = self.read(RecordType.OBSERVED_USAGE)
        if frame.empty:
            return {}
        frame = frame[frame["provider_request_id"].notna()]
        return dict(
            zip(
                frame["provider_request_id"].astype(str),
                frame["observation_id"].astype(str),
                strict=True,
            )
        )

    def write_batch(
        self, record_type: RecordType, rows: list[dict[str, Any]], batch_id: str
    ) -> Path | None:
        """Write one immutable batch. Returns ``None`` when there are no rows."""
        if not rows:
            return None
        path = self.table_dir(record_type) / f"part-{batch_id}.parquet"
        if path.exists():
            raise FileExistsError(f"Batch already exists: {path}")
        write_parquet(path, pd.DataFrame(rows))
        return path

    def log(self, entry: dict[str, Any]) -> None:
        append_jsonl(self.root / LOG_FILE, entry)

    def purge_expired(self, as_of: datetime | None = None) -> dict[str, int]:
        """Delete rows whose ``retain_until`` has passed, plus their predictions.

        Returns the number of rows removed per record type. Parts are rewritten
        atomically; a part left empty is deleted.
        """
        cutoff = pd.Timestamp(as_of or datetime.now(UTC))
        if cutoff.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        removed: dict[str, int] = {}
        expired_requests: set[str] = set()
        for record_type in (RecordType.REQUEST_FEATURES, RecordType.OBSERVED_USAGE):
            count = 0
            for path in self.parts(record_type):
                frame = pd.read_parquet(path)
                if "retain_until" not in frame.columns or frame.empty:
                    continue
                retain_until = pd.to_datetime(frame["retain_until"], utc=True)
                expired = retain_until.notna() & (retain_until <= cutoff)
                if not expired.any():
                    continue
                expired_requests.update(frame.loc[expired, "request_id"].astype(str))
                count += int(expired.sum())
                self._rewrite(path, frame[~expired])
            removed[record_type.value] = count
        count = 0
        for path in self.parts(RecordType.PREDICTION):
            frame = pd.read_parquet(path)
            orphaned = frame["request_id"].astype(str).isin(expired_requests)
            if orphaned.any():
                count += int(orphaned.sum())
                self._rewrite(path, frame[~orphaned])
        removed[RecordType.PREDICTION.value] = count
        self.log(
            {
                "event": "purge_expired",
                "as_of": cutoff.isoformat(),
                "removed": removed,
                "logged_at": datetime.now(UTC).isoformat(),
            }
        )
        return removed

    @staticmethod
    def _rewrite(path: Path, remaining: pd.DataFrame) -> None:
        if remaining.empty:
            path.unlink()
        else:
            write_parquet(path, remaining.reset_index(drop=True))
