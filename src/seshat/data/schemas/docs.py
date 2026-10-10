"""Field documentation for the canonical schema; renders ``docs/DATA_SCHEMA.md``.

Regenerate with ``python -m seshat.data.schemas.docs > docs/DATA_SCHEMA.md``.
A test fails if any schema field lacks documentation or the file is stale.
"""

from __future__ import annotations

import sys

from pydantic import BaseModel

from seshat.data.features.extract import FEATURE_DEFINITIONS
from seshat.data.schemas.arrow import arrow_schema, column_name
from seshat.data.schemas.enums import SCHEMA_VERSION
from seshat.data.schemas.records import (
    ObservationRecord,
    PredictionRecord,
    PreflightRecord,
    RequestRecord,
)

FIELD_DOCS: dict[str, str] = {
    # common / identity / provenance (A)
    "schema_version": "Canonical schema version (`1.0.0`).",
    "sample_id": "Stable UUID joining all tables. UUIDv5 of (experiment, prompt, configuration, repetition) for experiments; of (source, provider, response id) for imports/telemetry. Idempotency key.",
    "request_id": "Identifier of the provider request (one billable invocation).",
    "prediction_id": "Identifier of a prediction (deterministic from request and predictor version).",
    "run_id": "UUID of the collection run / import batch / telemetry session that produced the record.",
    "conversation_id": "Caller's conversation identifier, when known (grouped for leakage-safe splits).",
    "parent_request_id": "Parent invocation in a multi-call agent workflow.",
    "timestamp_utc": "When the request was dispatched (first attempt), or the logged request time for imports.",
    "provider": "Provider adapter key (`anthropic`, `openai`, `gemini`, `mock-*`).",
    "api_surface": "API used (e.g. `anthropic.messages`, `openai.responses`, `gemini.generate_content`).",
    "requested_model": "Model ID requested.",
    "sdk_version": "Client SDK version, when recorded.",
    "source_type": "Provenance: controlled_experiment | consented_api_telemetry | authorized_log_import | public_permitted_dataset | synthetic_unexecuted | test_fixture. Only the first three can carry real labels.",
    "dataset_origin": "Human-readable origin (e.g. `experiment:<id>`, the importing system).",
    "data_license": "License of the prompt content, when known.",
    "consent_status": "not_required | granted | denied | unknown. Raw text is stored only with `granted` (or project-authored content) and when privacy config allows.",
    "collection_status": "planned | dry_run | dispatched | completed | failed | uncertain | imported | skipped.",
    "experiment_id": "Experiment configuration that planned the sample.",
    "design_cell_id": "`<length stratum>|<category>|<config id>` design cell.",
    "variation_id": "Configuration (factor-level combination) identifier.",
    "repetition_index": "0 for the first observation of a cell, 1.. for repeated identical requests.",
    "generation_seed": "Seed of the prompt generator, for generated prompts.",
    # prompt identity (B)
    "prompt_id": "Prompt-bank identifier (or a content-hash id for imported calls).",
    "prompt_template_id": "Generator template id, when generated.",
    "prompt_family_id": "Group of near-identical prompts (template+topic, near-duplicate clusters). Never split across train/test.",
    "prompt_category": "Task category.",
    "prompt_subcategory": "Finer task label.",
    "prompt_language": "BCP-47-like language tag of the prompt.",
    "full_prompt_storage_ref": "Path (under the storage root) of the stored full request content; null unless privacy allows storage.",
    # request config (C)
    "max_output_tokens": "Output token limit sent (includes reasoning for OpenAI/Anthropic/Gemini).",
    "requested_effort": "Effort requested (Anthropic output_config.effort, OpenAI reasoning.effort, Gemini thinking_level).",
    "resolved_effort": "Effort actually sent after capability checks (same as requested unless mapped).",
    "thinking_mode": "adaptive | enabled | disabled (Anthropic) or null.",
    "thinking_budget_tokens": "Thinking budget, when set.",
    "temperature": "Sampling temperature; null when not set or unsupported (never a fabricated default).",
    "top_p": "Nucleus sampling parameter; null when unset/unsupported.",
    "top_k": "Top-k sampling parameter; null when unset/unsupported.",
    "seed": "Sampling seed, when supported and set.",
    "response_format": "text | json | json_schema, when requested.",
    "verbosity": "OpenAI text verbosity, when set.",
    "tool_choice": "auto | any | none | <tool name>.",
    "parallel_tool_calls": "Whether parallel tool calls were allowed (null when not set).",
    "streaming_enabled": "Streaming requested.",
    "requested_cache_ttl": "Requested prompt-cache TTL.",
    "service_tier": "Requested service tier.",
    "region": "Requested region / data residency.",
    "routing_mode": "Automatic routing mode requested (distinguishes routed requests from the resolved model).",
    "provider_settings": "Provider-specific options sent verbatim (structured JSON).",
    "context_window_limit": "Model context window from the catalog; null when undocumented.",
    "image_metadata": "Per-image metadata (mime type, bytes, dimensions); images themselves are never stored.",
    "attachment_metadata": "Per-attachment metadata (type, mime type, bytes, pages, duration).",
    "retain_until": "Expiry for privacy retention; null = kept. `purge-expired` deletes rows past it.",
    # preflight (E)
    "preflight_id": "Identifier of the measurement (sample + method).",
    "preflight_input_tokens": "Input tokens counted before dispatch. Never authoritative usage.",
    "preflight_method": "provider_native | local_tokenizer | statistical_approximation | unavailable.",
    "preflight_provider": "Who produced the count (provider endpoint or local estimator).",
    "tokenizer_name": "Counting endpoint or tokenizer name.",
    "tokenizer_version": "Tokenizer version / model whose tokenizer was used.",
    "preflight_timestamp_utc": "When the count was taken (always before dispatch).",
    "preflight_error": "Why no count is available.",
    "preflight_latency_ms": "Time taken by the preflight call.",
    "preflight_includes": "Request parts covered by the count (system, messages, tools, thinking ...).",
    "preflight_complete": "True when the count covers all billable input content.",
    # observation (F)
    "observation_id": "Identifier of the observation (one per billable invocation).",
    "observation_kind": "invocation (one API call; provider iterations summed) | workflow_aggregate.",
    "invocation_index": "Index of the invocation within a workflow.",
    "resolved_model": "Model the provider reports it used (post-inference).",
    "model_version": "Provider model version string, when reported.",
    "actual_input_tokens": "Total input tokens billed, reconstructed per provider rules (cache reads/writes included).",
    "actual_output_tokens": "Total generated tokens billed, including reasoning/thinking for every provider.",
    "actual_reasoning_tokens": "Reasoning/thinking tokens when reported; null when not reported.",
    "actual_total_tokens": "actual_input_tokens + actual_output_tokens.",
    "actual_cached_input_tokens": "Input tokens read from cache.",
    "actual_cache_creation_tokens": "Input tokens written to cache (Anthropic); null where not reported.",
    "actual_uncached_input_tokens": "Input tokens neither read from nor written to cache.",
    "actual_visible_output_tokens": "Output tokens excluding reasoning (Gemini: candidates).",
    "actual_tool_input_tokens": "Tool-use prompt tokens reported separately (Gemini tool_use_prompt_token_count).",
    "actual_tool_output_tokens": "Tool output tokens reported separately, when available.",
    "input_usage_semantics": "Provider rule used to derive input fields.",
    "output_usage_semantics": "Provider rule used to derive output fields.",
    "reasoning_included_in_output": "Whether the provider's own output field includes reasoning (OpenAI/Anthropic true, Gemini false).",
    "provider_total_tokens": "Total reported by the provider (null for Anthropic).",
    "normalized_total_tokens": "Seshat's total from documented definitions (cross-checked against provider_total_tokens).",
    "usage_iteration_count": "Sampling iterations in the call (e.g. Anthropic compaction).",
    "label_quality": "verified | partial | censored | inconsistent | unavailable | synthetic.",
    "usage_observation_source": "Which provider usage object the labels came from.",
    "raw_usage": "Original provider usage payload, unmodified (JSON).",
    # execution (G)
    "response_id": "Provider response id.",
    "response_status": "completed | incomplete | refused | failed | uncertain | not_executed.",
    "stop_reason": "Provider stop/finish reason.",
    "is_truncated": "Hit the output limit; the output target is right-censored.",
    "first_token_latency_ms": "Time to first token (streaming), when measured.",
    "total_latency_ms": "Wall-clock latency of the call.",
    "client_tool_call_count": "Client tool calls requested in the response.",
    "server_tool_call_count": "Server-side tool calls (e.g. web search) in the response.",
    "retry_count": "Retries before the final outcome (rate limits, overload).",
    "rate_limit_encountered": "Any attempt was rate limited.",
    "error_type": "Error category for failed/uncertain calls.",
    "error_code": "HTTP status or provider error code.",
    "completed_at_utc": "When the response completed.",
    # cost (H)
    "pricing_version": "Price-table version used (configs/pricing/*.json).",
    "pricing_effective_date": "Effective-from date of the price entry.",
    "pricing_source": "Official pricing page URL (or `fixture`).",
    "input_rate_usd_per_million": "Input rate applied (USD / 1M tokens).",
    "output_rate_usd_per_million": "Output rate applied (USD / 1M tokens).",
    "cache_read_rate_usd_per_million": "Cache-read rate applied.",
    "cache_write_rate_usd_per_million": "Cache-write rate applied (5m or 1h).",
    "calculated_cost_usd": "Price-table estimate of the call's cost. Not an invoice charge.",
    "provider_reported_cost_usd": "Cost reported by the provider, when available.",
    "currency": "Always USD.",
    "cost_method": "price_table_estimate | provider_reported | unavailable.",
    "billing_dimensions": "Extra billing facts: cache TTL breakdown, server tool use, tiers, accounting issues, fixture flags (JSON).",
    "collected_at_utc": "When the observation was recorded.",
    # predictions
    "created_at_utc": "When the prediction was made.",
    "predictor_version": "Predictor that produced the value.",
    "target": "Predicted quantity.",
    "predicted_output_tokens": "Point prediction.",
    "lower_bound": "Lower interval bound.",
    "upper_bound": "Upper interval bound.",
    "confidence": "Predictor-reported confidence (not calibrated unless interval_calibrated).",
    "interval_calibrated": "Whether the interval was empirically calibrated.",
}

TABLES: list[tuple[str, type[BaseModel], str]] = [
    (
        "requests",
        RequestRecord,
        "Identity & provenance (A), prompt & conversation features (B), execution configuration (C), context & multimodal (D). Request-time only.",
    ),
    (
        "preflight",
        PreflightRecord,
        "Preflight input-token measurements (E). Taken before dispatch; never authoritative.",
    ),
    (
        "observations",
        ObservationRecord,
        "Observed usage labels (F), execution results (G), cost & billing (H). Post-inference only.",
    ),
    ("predictions", PredictionRecord, "Seshat predictions. Never observations or labels."),
]


def describe(field: str) -> str:
    return FIELD_DOCS.get(field) or FEATURE_DEFINITIONS.get(field) or ""


def _cell(text: str) -> str:
    return text.replace("|", r"\|")


def undocumented_fields() -> list[str]:
    return sorted(
        {name for _, model, _ in TABLES for name in model.model_fields if not describe(name)}
    )


def render() -> str:
    lines = [
        "# Seshat canonical dataset schema",
        "",
        f'`schema_version = "{SCHEMA_VERSION}"`. Generated from `seshat.data.schemas` by',
        "`python -m seshat.data.schemas.docs > docs/DATA_SCHEMA.md`; do not edit by hand.",
        "",
        "Tables are joined by `sample_id` (and `request_id`). `null` means unavailable /",
        "not reported / not applicable; `0` means an observed zero. JSON-valued fields are",
        "stored in Parquet as string columns named `<field>_json`.",
        "",
    ]
    for table, model, description in TABLES:
        schema = arrow_schema(model)
        lines += [
            f"## `{table}` — {model.__name__}",
            "",
            description,
            "",
            "| Field | Parquet column | Arrow type | Definition |",
            "|---|---|---|---|",
        ]
        for name in model.model_fields:
            column = column_name(model, name)
            arrow_type = str(schema.field(column).type)
            lines.append(f"| `{name}` | `{column}` | `{arrow_type}` | {_cell(describe(name))} |")
        lines.append("")
    lines += [
        "## Normalized usage",
        "",
        "Provider rules (from official documentation, checked 2026-10-10). Totals are never",
        "computed by adding every available field.",
        "",
        "| Provider | actual_input_tokens | actual_output_tokens | actual_reasoning_tokens | totals |",
        "|---|---|---|---|---|",
        "| OpenAI (Responses) | `input_tokens` (cached_tokens is a subset) | `output_tokens` (includes reasoning) | `output_tokens_details.reasoning_tokens` | normalized = input + output; checked against `total_tokens` |",
        "| Anthropic (Messages) | `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens` (input_tokens = tokens after the last cache breakpoint) | `output_tokens` (includes thinking) | `output_tokens_details.thinking_tokens` | no provider total; with compaction, sum over `usage.iterations` (top level excludes compaction) |",
        "| Gemini (generate_content) | `prompt_token_count` (includes cached content) | `candidates_token_count` + `thoughts_token_count` | `thoughts_token_count` | total = prompt + candidates + thoughts; `tool_use_prompt_token_count` kept separately, never added |",
        "",
        "Contradictions (cached > input, reasoning > output, total mismatch, cache breakdown",
        "mismatch, negative values) are recorded in `billing_dimensions.accounting_issues`,",
        "the label becomes `inconsistent`, and validation quarantines the row. The original",
        "payload is always kept in `raw_usage`.",
        "",
        "## Label quality",
        "",
        "| Value | Meaning | Training |",
        "|---|---|---|",
        "| verified | complete usage, accounting consistent | eligible |",
        "| partial | primary labels present, some secondary fields unreported | eligible |",
        "| censored | truncated at the output limit (right-censored) | eligible, flagged `target_is_censored` |",
        "| inconsistent | contradictory provider accounting | quarantined |",
        "| unavailable | failed / uncertain / no usage | excluded |",
        "| synthetic | mock-provider test fixture | only with `--label-origin synthetic`, never with real labels |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    sys.stdout.write(render())


if __name__ == "__main__":
    main()
