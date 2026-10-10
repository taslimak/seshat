"""Human-readable CSV export. CSV is a view for inspection; Parquet/JSONL stay canonical."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _read_source(source: Path) -> pd.DataFrame:
    if source.is_dir():
        parts = sorted(source.glob("*.parquet"))
        if not parts:
            raise ValueError(f"No Parquet files in {source}")
        return pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix == ".jsonl":
        return pd.read_json(source, lines=True)
    raise ValueError(
        "CSV export supports a .parquet file, a .jsonl file or a store table directory"
    )


def _neutralize_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefix text cells that spreadsheets would execute as formulas with a quote."""
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: f"'{value}"
                if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES)
                else value
            )
    return result


def export_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    columns: list[str] | None = None,
    sanitize_formulas: bool = True,
) -> Path:
    """Export a Parquet/JSONL file or a store table directory to CSV."""
    frame = _read_source(Path(input_path))
    if columns:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Unknown columns: {missing}")
        frame = frame[columns]
    if limit is not None:
        frame = frame.head(limit)
    if sanitize_formulas:
        frame = _neutralize_formulas(frame)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target
