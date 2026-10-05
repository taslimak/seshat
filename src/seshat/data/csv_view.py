"""Human-readable CSV export for inspecting records without changing canonical Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> Path:
    source = Path(input_path)
    if source.suffix == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix == ".jsonl":
        frame = pd.read_json(source, lines=True)
    else:
        raise ValueError("CSV export supports .parquet or .jsonl input")
    if limit is not None:
        frame = frame.head(limit)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target
