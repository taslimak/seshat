"""Readable CSV exports of the normalized layer, with a data dictionary.

The CSV joins request features, the preflight count and observed usage into one row
per observation, with plain column names. Nested/JSON fields are deliberately left
out; they are documented in ``data_dictionary.md`` and remain in Parquet.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd

from seshat.data.csv_view import neutralize_formulas
from seshat.data.io import read_parquet
from seshat.data.storage import StorageLayout

COLUMNS: dict[str, tuple[str, str]] = {
    # name: (source column, definition)
    "provider": ("provider", "Provider adapter name (mock-* are offline test fixtures)."),
    "model": ("requested_model", "Model ID requested."),
    "resolved_model": ("resolved_model", "Model the provider reports it used (post-inference)."),
    "prompt_category": ("prompt_category", "Task category of the prompt."),
    "prompt_char_count": ("prompt_char_count", "Characters in system prompt + all messages."),
    "message_count": ("message_count", "Messages in the request (excluding the system prompt)."),
    "effort": ("requested_effort", "Requested reasoning/thinking effort (blank = not set)."),
    "max_output_tokens": ("max_output_tokens", "Output token limit sent with the request."),
    "tool_count": ("tool_count", "Number of tools offered."),
    "caching_enabled": ("caching_enabled", "Prompt caching requested."),
    "preflight_input_tokens": (
        "preflight_input_tokens",
        "Input tokens counted before dispatch (not authoritative).",
    ),
    "preflight_method": (
        "preflight_method",
        "provider_native | local_tokenizer | statistical_approximation | unavailable.",
    ),
    "actual_input_tokens": (
        "actual_input_tokens",
        "Observed total input tokens (cache reads/writes included; "
        "provider-specific rules in DATA_SCHEMA.md).",
    ),
    "actual_cached_input_tokens": (
        "actual_cached_input_tokens",
        "Observed input tokens served from cache.",
    ),
    "actual_output_tokens": (
        "actual_output_tokens",
        "Observed billed output tokens, including reasoning/thinking.",
    ),
    "actual_reasoning_tokens": (
        "actual_reasoning_tokens",
        "Observed reasoning/thinking tokens (blank = not reported).",
    ),
    "actual_total_tokens": ("actual_total_tokens", "actual_input_tokens + actual_output_tokens."),
    "cost_usd": (
        "calculated_cost_usd",
        "Price-table estimate in USD (not an invoice; fixture prices for mock providers).",
    ),
    "cost_method": ("cost_method", "price_table_estimate | provider_reported | unavailable."),
    "latency_ms": ("total_latency_ms", "Wall-clock request latency in milliseconds."),
    "response_status": (
        "response_status",
        "completed | incomplete | refused | failed | uncertain.",
    ),
    "is_truncated": ("is_truncated", "Output hit the token limit (target is right-censored)."),
    "label_quality": (
        "label_quality",
        "verified | partial | censored | inconsistent | unavailable | synthetic.",
    ),
    "source_type": (
        "source_type",
        "controlled_experiment | consented_api_telemetry | authorized_log_import "
        "| test_fixture ...",
    ),
    "validation_decision": ("decision", "PASS | WARNING | QUARANTINE | REJECT from `validate`."),
    "collected_at_utc": ("collected_at_utc", "When the observation was collected (UTC)."),
    "sample_id": ("sample_id", "Stable sample identifier (joins all tables)."),
    "request_id": ("request_id", "Request identifier."),
}


def _cell(text: str) -> str:
    """Escape pipes so enumerations stay inside one Markdown table cell."""
    return text.replace("|", r"\|")


def export_csv(
    root: str | Path, output: str | Path | None = None, *, limit: int | None = None
) -> Path:
    layout = StorageLayout(root)
    requests = layout.read_frame("requests")
    observations = layout.read_frame("observations")
    preflight = layout.read_frame("preflight")
    if observations.empty:
        raise ValueError("No observations to export")
    frame = observations.merge(
        requests[
            [
                "sample_id",
                "prompt_category",
                "prompt_char_count",
                "message_count",
                "requested_effort",
                "max_output_tokens",
                "tool_count",
                "caching_enabled",
            ]
        ],
        on="sample_id",
        how="left",
    )
    if not preflight.empty:
        pre = preflight.sort_values("preflight_method").drop_duplicates("sample_id")
        frame = frame.merge(
            pre[["sample_id", "preflight_input_tokens", "preflight_method"]],
            on="sample_id",
            how="left",
        )
    else:
        frame["preflight_input_tokens"] = None
        frame["preflight_method"] = None
    decisions_path = layout.validation / "decisions.parquet"
    if decisions_path.exists():
        frame = frame.merge(
            read_parquet(decisions_path)[["observation_id", "decision"]],
            on="observation_id",
            how="left",
        )
    else:
        frame["decision"] = None
    out = pd.DataFrame({name: frame[source] for name, (source, _) in COLUMNS.items()})
    for column in out.columns:
        if column.endswith("_tokens") or column in {
            "prompt_char_count",
            "message_count",
            "tool_count",
        }:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
    out["cost_usd"] = out["cost_usd"].map(
        lambda v: None if v is None or pd.isna(v) else f"{Decimal(str(v)):.6f}"
    )
    out = out.sort_values(["collected_at_utc", "sample_id"]).reset_index(drop=True)
    if limit is not None:
        out = out.head(limit)
    target = Path(output) if output else layout.exports / "observations.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    neutralize_formulas(out).to_csv(target, index=False)
    write_data_dictionary(target.with_name("data_dictionary.md"))
    return target


def write_data_dictionary(path: Path) -> None:
    lines = [
        "# CSV data dictionary",
        "",
        "One row per observation (one billable model invocation). Blank = not reported / not",
        "applicable; 0 = observed zero. Full schema and nested fields: docs/DATA_SCHEMA.md.",
        "",
        "| Column | Source field | Definition |",
        "|---|---|---|",
        *[
            f"| {name} | `{source}` | {_cell(definition)} |"
            for name, (source, definition) in COLUMNS.items()
        ],
        "",
        "Not exported (see Parquet): raw_usage_json (original provider usage payload),",
        "billing_dimensions_json (cache TTL breakdown, accounting issues, tiers),",
        "provider_settings_json,",
        "attachment/image metadata lists and all hashes.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
