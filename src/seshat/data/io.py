"""Small, deterministic IO helpers for the file-based v0 data factory."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, TypeVar

import pandas as pd
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def load_jsonl(path: str | Path, model: type[T]) -> list[T]:
    records: list[T] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def append_jsonl(path: str | Path, record: BaseModel | dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(record, BaseModel):
        payload = record.model_dump(mode="json")
    else:
        payload = record
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "
")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True, ensure_ascii=False)
        tmp.write("
")
        tmp_path = Path(tmp.name)
    tmp_path.replace(target)


def write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def dataframe_from_models(records: Iterable[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([record.model_dump(mode="json") for record in records])
