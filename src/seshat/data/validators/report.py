"""Dataset quality report (JSON + Markdown) and the ``validate`` entry point."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.io import write_json, write_parquet
from seshat.data.storage import StorageLayout
from seshat.data.validators.checks import flag_counts, validate_layer


def _dist(series: pd.Series) -> dict[str, float] | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "count": int(values.size),
        "mean": round(float(values.mean()), 2),
        "p10": float(values.quantile(0.10)),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "max": float(values.max()),
    }


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.fillna("null").value_counts().sort_index().items()}


def build_quality_report(
    requests: pd.DataFrame,
    preflight: pd.DataFrame,
    observations: pd.DataFrame,
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    total = int(len(observations))
    eligible = decisions[decisions["eligible_for_training"]] if total else decisions
    merged = (
        observations.merge(
            requests[
                [
                    "sample_id",
                    "prompt_category",
                    "requested_effort",
                    "prompt_hash",
                    "prompt_family_id",
                ]
            ],
            on="sample_id",
            how="left",
        )
        if total and not requests.empty
        else observations
    )
    labeled = merged[merged["actual_output_tokens"].notna()] if total else merged
    cost = pd.to_numeric(
        observations.get("calculated_cost_usd", pd.Series(dtype=float)).map(
            lambda v: None if v is None else float(Decimal(str(v)))
        ),
        errors="coerce",
    )
    spend = (
        {
            str(source): round(float(cost[observations["source_type"] == source].sum()), 6)
            for source in sorted(observations["source_type"].unique())
        }
        if total
        else {}
    )
    cache_share = None
    if not labeled.empty:
        with_cache = labeled["actual_cached_input_tokens"].fillna(0) > 0
        fraction = labeled["actual_cached_input_tokens"].fillna(0) / labeled[
            "actual_input_tokens"
        ].where(labeled["actual_input_tokens"] > 0)
        cache_share = {
            "observations_with_cache_reads": int(with_cache.sum()),
            "share_with_cache_reads": round(float(with_cache.mean()), 4),
            "cached_fraction_of_input": _dist(fraction[with_cache]),
            "cache_write_observations": int(
                (labeled["actual_cache_creation_tokens"].fillna(0) > 0).sum()
            ),
        }
    duplicates = {
        "duplicate_observation_ids": int(observations["observation_id"].duplicated().sum())
        if total
        else 0,
        "duplicate_prompt_content_rate": round(
            float(requests["prompt_hash"].duplicated().mean()), 4
        )
        if not requests.empty
        else 0.0,
        "prompt_hashes_in_multiple_families": int(
            (requests.groupby("prompt_hash")["prompt_family_id"].nunique() > 1).sum()
        )
        if not requests.empty
        else 0,
        "note": (
            "Repeated observations of the same prompt (by design) count as duplicate "
            "content; group-aware splitting keeps them together. Near-duplicate prompt "
            "text is detected at the prompt-bank stage (seshat-data prompts check)."
        ),
    }
    coverage = (
        merged.groupby(["prompt_category"])["provider"].nunique().to_dict()
        if total and "prompt_category" in merged
        else {}
    )
    native = (
        preflight[
            (preflight["preflight_method"] == "provider_native")
            & preflight["preflight_input_tokens"].notna()
        ]
        if not preflight.empty
        else preflight
    )
    discrepancy = None
    if not native.empty and total:
        joined = native.merge(observations[["sample_id", "actual_input_tokens"]], on="sample_id")
        joined = joined[joined["actual_input_tokens"].notna()]
        rel = (joined["actual_input_tokens"] - joined["preflight_input_tokens"]) / joined[
            "actual_input_tokens"
        ].clip(lower=1)
        discrepancy = _dist(rel)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "totals": {
            "request_records": int(len(requests)),
            "observations": total,
            "eligible_labeled": int(len(eligible)),
            "eligible_real_labels": int((eligible["data_origin"] == "real").sum())
            if len(eligible)
            else 0,
            "eligible_synthetic_labels": int((eligible["data_origin"] == "synthetic").sum())
            if len(eligible)
            else 0,
            "quarantined": int((decisions["decision"] == "QUARANTINE").sum()) if total else 0,
            "rejected": int((decisions["decision"] == "REJECT").sum()) if total else 0,
            "warnings": int((decisions["decision"] == "WARNING").sum()) if total else 0,
            "missing_label_pct": round(
                100 * float(observations["actual_output_tokens"].isna().mean()), 2
            )
            if total
            else 0.0,
        },
        "by_decision": _counts(decisions["decision"]) if total else {},
        "by_flag": flag_counts(decisions) if total else {},
        "by_label_quality": _counts(observations["label_quality"]) if total else {},
        "by_source_type": _counts(observations["source_type"]) if total else {},
        "by_provider_model": _counts(
            observations["provider"] + "/" + observations["requested_model"]
        )
        if total
        else {},
        "by_prompt_category": _counts(merged["prompt_category"])
        if total and "prompt_category" in merged
        else {},
        "by_effort": _counts(merged["requested_effort"])
        if total and "requested_effort" in merged
        else {},
        "by_response_status": _counts(observations["response_status"]) if total else {},
        "token_distributions": {
            name: _dist(labeled[name])
            for name in (
                "actual_input_tokens",
                "actual_output_tokens",
                "actual_reasoning_tokens",
                "actual_total_tokens",
            )
        }
        if total
        else {},
        "latency_ms": _dist(observations["total_latency_ms"]) if total else None,
        "cache_usage": cache_share,
        "approximate_spend_usd_by_source_type": spend,
        "spend_note": (
            "Price-table estimates. test_fixture spend uses fixture prices and is not real money."
        ),
        "duplicates": duplicates,
        "coverage_providers_per_category": {str(k): int(v) for k, v in coverage.items()},
        "preflight_relative_discrepancy": discrepancy,
        "token_accounting_discrepancies": {
            code: count
            for code, count in flag_counts(decisions).items()
            if code.endswith("accounting") or code == "preflight_discrepancy"
        }
        if total
        else {},
    }


def render_markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "# Seshat dataset quality report",
        "",
        f"Generated {report['generated_at']}.",
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *[f"| {k.replace('_', ' ')} | {v} |" for k, v in totals.items()],
        "",
    ]

    def table(title: str, data: dict[str, Any]) -> None:
        if not data:
            return
        lines.extend([f"## {title}", "", "| Value | Count |", "|---|---:|"])
        lines.extend(f"| {k} | {v} |" for k, v in data.items())
        lines.append("")

    table("Decisions", report["by_decision"])
    table("Validation flags", report["by_flag"])
    table("Label quality", report["by_label_quality"])
    table("Source type", report["by_source_type"])
    table("Provider / model", report["by_provider_model"])
    table("Prompt category", report["by_prompt_category"])
    table("Effort", report["by_effort"])
    table("Approximate spend (USD) by source type", report["approximate_spend_usd_by_source_type"])
    lines.extend(
        [
            "_" + report["spend_note"] + "_",
            "",
            "## Token distributions",
            "",
            "| Field | count | mean | p10 | p50 | p90 | max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, dist in report["token_distributions"].items():
        if dist:
            lines.append(
                f"| {name} | {dist['count']} | {dist['mean']} | {dist['p10']} "
                f"| {dist['p50']} | {dist['p90']} | {dist['max']} |"
            )
    lines.append("")
    if report.get("cache_usage"):
        lines.extend(["## Cache usage", "", f"```json\n{_json(report['cache_usage'])}\n```", ""])
    lines.extend(["## Duplicates", "", f"```json\n{_json(report['duplicates'])}\n```", ""])
    if report.get("preflight_relative_discrepancy"):
        lines.extend(
            [
                "## Preflight vs observed input (relative)",
                "",
                f"```json\n{_json(report['preflight_relative_discrepancy'])}\n```",
                "",
            ]
        )
    return "\n".join(lines)


def _json(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)


def validate_storage(root: str | Path, *, preflight_tolerance: float = 0.1) -> dict[str, Any]:
    """Validate the normalized layer and write decisions + quality reports."""
    layout = StorageLayout(root)
    requests = layout.read_frame("requests")
    preflight = layout.read_frame("preflight")
    observations = layout.read_frame("observations")
    decisions = validate_layer(
        requests, preflight, observations, preflight_tolerance=preflight_tolerance
    )
    report = build_quality_report(requests, preflight, observations, decisions)
    out = layout.validation
    write_parquet(out / "decisions.parquet", decisions)
    write_json(out / "quality_report.json", report)
    (out / "quality_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report
