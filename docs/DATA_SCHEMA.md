# Seshat canonical dataset schema

`schema_version = "1.0.0"`. Generated from `seshat.data.schemas` by
`python -m seshat.data.schemas.docs > docs/DATA_SCHEMA.md`; do not edit by hand.

Tables are joined by `sample_id` (and `request_id`). `null` means unavailable /
not reported / not applicable; `0` means an observed zero. JSON-valued fields are
stored in Parquet as string columns named `<field>_json`.

## `requests` — RequestRecord

Identity & provenance (A), prompt & conversation features (B), execution configuration (C), context & multimodal (D). Request-time only.

| Field | Parquet column | Arrow type | Definition |
|---|---|---|---|
| `schema_version` | `schema_version` | `string` | Canonical schema version (`1.0.0`). |
| `sample_id` | `sample_id` | `string` | Stable UUID joining all tables. UUIDv5 of (experiment, prompt, configuration, repetition) for experiments; of (source, provider, response id) for imports/telemetry. Idempotency key. |
| `request_id` | `request_id` | `string` | Identifier of the provider request (one billable invocation). |
| `run_id` | `run_id` | `string` | UUID of the collection run / import batch / telemetry session that produced the record. |
| `conversation_id` | `conversation_id` | `string` | Caller's conversation identifier, when known (grouped for leakage-safe splits). |
| `parent_request_id` | `parent_request_id` | `string` | Parent invocation in a multi-call agent workflow. |
| `timestamp_utc` | `timestamp_utc` | `timestamp[us, tz=UTC]` | When the request was dispatched (first attempt), or the logged request time for imports. |
| `provider` | `provider` | `string` | Provider adapter key (`anthropic`, `openai`, `gemini`, `mock-*`). |
| `api_surface` | `api_surface` | `string` | API used (e.g. `anthropic.messages`, `openai.responses`, `gemini.generate_content`). |
| `requested_model` | `requested_model` | `string` | Model ID requested. |
| `sdk_version` | `sdk_version` | `string` | Client SDK version, when recorded. |
| `source_type` | `source_type` | `string` | Provenance: controlled_experiment \| consented_api_telemetry \| authorized_log_import \| public_permitted_dataset \| synthetic_unexecuted \| test_fixture. Only the first three can carry real labels. |
| `dataset_origin` | `dataset_origin` | `string` | Human-readable origin (e.g. `experiment:<id>`, the importing system). |
| `data_license` | `data_license` | `string` | License of the prompt content, when known. |
| `consent_status` | `consent_status` | `string` | not_required \| granted \| denied \| unknown. Raw text is stored only with `granted` (or project-authored content) and when privacy config allows. |
| `collection_status` | `collection_status` | `string` | planned \| dry_run \| dispatched \| completed \| failed \| uncertain \| imported \| skipped. |
| `experiment_id` | `experiment_id` | `string` | Experiment configuration that planned the sample. |
| `design_cell_id` | `design_cell_id` | `string` | `<length stratum>\|<category>\|<config id>` design cell. |
| `variation_id` | `variation_id` | `string` | Configuration (factor-level combination) identifier. |
| `repetition_index` | `repetition_index` | `int64` | 0 for the first observation of a cell, 1.. for repeated identical requests. |
| `generation_seed` | `generation_seed` | `int64` | Seed of the prompt generator, for generated prompts. |
| `prompt_id` | `prompt_id` | `string` | Prompt-bank identifier (or a content-hash id for imported calls). |
| `prompt_template_id` | `prompt_template_id` | `string` | Generator template id, when generated. |
| `prompt_family_id` | `prompt_family_id` | `string` | Group of near-identical prompts (template+topic, near-duplicate clusters). Never split across train/test. |
| `prompt_category` | `prompt_category` | `string` | Task category. |
| `prompt_subcategory` | `prompt_subcategory` | `string` | Finer task label. |
| `prompt_language` | `prompt_language` | `string` | BCP-47-like language tag of the prompt. |
| `prompt_char_count` | `prompt_char_count` | `int64` | Unicode characters in all text sent: system prompt + every message, joined by blank lines (tool schemas excluded). |
| `prompt_word_count` | `prompt_word_count` | `int64` | Whitespace-separated tokens in the same text. |
| `prompt_line_count` | `prompt_line_count` | `int64` | Newline count + 1 in the same text. |
| `prompt_byte_count` | `prompt_byte_count` | `int64` | UTF-8 bytes of the same text. |
| `system_prompt_char_count` | `system_prompt_char_count` | `int64` | Characters in the system prompt (0 when absent). |
| `message_count` | `message_count` | `int64` | Messages in the request, excluding the system prompt. |
| `conversation_turn_count` | `conversation_turn_count` | `int64` | Number of user messages (turns initiated by the user). |
| `conversation_char_count` | `conversation_char_count` | `int64` | Characters across all messages, excluding the system prompt. |
| `current_user_message_char_count` | `current_user_message_char_count` | `int64` | Characters in the final (current) message. |
| `role_sequence` | `role_sequence` | `list<item: string>` | Ordered roles: 'system' first when present, then each message role. |
| `contains_code` | `contains_code` | `bool` | Regex: code fences, language keywords at line start, SQL statements, or statement terminators. |
| `contains_json` | `contains_json` | `bool` | A JSON object/array parses from the text, or a quoted-key pattern ("key":) appears. |
| `contains_markdown` | `contains_markdown` | `bool` | Regex: headings, list items, bold, links, tables or code fences. |
| `contains_mathematics` | `contains_mathematics` | `bool` | Regex: LaTeX, math symbols, arithmetic equations, exponents or modular notation. |
| `requested_output_format` | `requested_output_format` | `string` | response_format json/json_schema -> 'json'; else first matching keyword rule on the current message (json, yaml, csv, table, code, bullet_list, markdown); else 'unspecified'. |
| `explicit_output_length_hint` | `explicit_output_length_hint` | `string` | First matching length instruction in the current message, normalized (e.g. 'max_words:120', 'count_bullet_points:3', 'qualitative:brief'); null if none. |
| `prompt_hash` | `prompt_hash` | `string` | Hash (HMAC-SHA256 with SESHAT_PROMPT_HASH_KEY, else SHA-256) of the canonical JSON of system + messages (text and attachment metadata). |
| `conversation_hash` | `conversation_hash` | `string` | Hash of the prior context (system + all but the last message); null for single-message requests. |
| `full_prompt_storage_ref` | `full_prompt_storage_ref` | `string` | Path (under the storage root) of the stored full request content; null unless privacy allows storage. |
| `max_output_tokens` | `max_output_tokens` | `int64` | Output token limit sent (includes reasoning for OpenAI/Anthropic/Gemini). |
| `requested_effort` | `requested_effort` | `string` | Effort requested (Anthropic output_config.effort, OpenAI reasoning.effort, Gemini thinking_level). |
| `resolved_effort` | `resolved_effort` | `string` | Effort actually sent after capability checks (same as requested unless mapped). |
| `thinking_mode` | `thinking_mode` | `string` | adaptive \| enabled \| disabled (Anthropic) or null. |
| `thinking_budget_tokens` | `thinking_budget_tokens` | `int64` | Thinking budget, when set. |
| `temperature` | `temperature` | `double` | Sampling temperature; null when not set or unsupported (never a fabricated default). |
| `top_p` | `top_p` | `double` | Nucleus sampling parameter; null when unset/unsupported. |
| `top_k` | `top_k` | `int64` | Top-k sampling parameter; null when unset/unsupported. |
| `seed` | `seed` | `int64` | Sampling seed, when supported and set. |
| `stop_sequence_count` | `stop_sequence_count` | `int64` | Number of stop sequences configured. |
| `response_format` | `response_format` | `string` | text \| json \| json_schema, when requested. |
| `verbosity` | `verbosity` | `string` | OpenAI text verbosity, when set. |
| `tool_count` | `tool_count` | `int64` | Number of tools offered. |
| `tool_choice` | `tool_choice` | `string` | auto \| any \| none \| <tool name>. |
| `tool_schema_bytes` | `tool_schema_bytes` | `int64` | UTF-8 bytes of the canonical JSON of all tool definitions (0 without tools). |
| `tool_schema_hash` | `tool_schema_hash` | `string` | SHA-256 of that canonical JSON; null without tools. |
| `enabled_tool_types` | `enabled_tool_types` | `list<item: string>` | Sorted distinct tool types (function, web_search, ...). |
| `parallel_tool_calls` | `parallel_tool_calls` | `bool` | Whether parallel tool calls were allowed (null when not set). |
| `streaming_enabled` | `streaming_enabled` | `bool` | Streaming requested. |
| `caching_enabled` | `caching_enabled` | `bool` | Prompt caching requested. |
| `requested_cache_ttl` | `requested_cache_ttl` | `string` | Requested prompt-cache TTL. |
| `service_tier` | `service_tier` | `string` | Requested service tier. |
| `region` | `region` | `string` | Requested region / data residency. |
| `routing_mode` | `routing_mode` | `string` | Automatic routing mode requested (distinguishes routed requests from the resolved model). |
| `provider_settings` | `provider_settings_json` | `string` | Provider-specific options sent verbatim (structured JSON). |
| `context_window_limit` | `context_window_limit` | `int64` | Model context window from the catalog; null when undocumented. |
| `context_message_count` | `context_message_count` | `int64` | Messages before the current one. |
| `context_character_count` | `context_character_count` | `int64` | Characters in the system prompt plus all messages before the current one. |
| `context_preflight_tokens` | `context_preflight_tokens` | `int64` | Preflight input-token count for the whole request when available (from the preflight table). |
| `image_count` | `image_count` | `int64` | Image parts across all messages. |
| `image_total_bytes` | `image_total_bytes` | `int64` | Sum of known image byte sizes; null when no size is known. |
| `image_metadata` | `image_metadata` | `list<item: struct<mime_type: string, byte_size: int64, width: int64, height: int64>>` | Per-image metadata (mime type, bytes, dimensions); images themselves are never stored. |
| `document_count` | `document_count` | `int64` | Document parts across all messages. |
| `document_page_count` | `document_page_count` | `int64` | Sum of known document page counts; null when unknown. |
| `audio_duration_seconds` | `audio_duration_seconds` | `double` | Sum of known audio durations; null when no audio or unknown. |
| `video_duration_seconds` | `video_duration_seconds` | `double` | Sum of known video durations; null when no video or unknown. |
| `modality_types` | `modality_types` | `list<item: string>` | Sorted distinct content part types present (text, image, document, audio, video). |
| `attachment_metadata` | `attachment_metadata` | `list<item: struct<type: string, mime_type: string, byte_size: int64, page_count: int64, duration_seconds: double>>` | Per-attachment metadata (type, mime type, bytes, pages, duration). |
| `retain_until` | `retain_until` | `timestamp[us, tz=UTC]` | Expiry for privacy retention; null = kept. `purge-expired` deletes rows past it. |

## `preflight` — PreflightRecord

Preflight input-token measurements (E). Taken before dispatch; never authoritative.

| Field | Parquet column | Arrow type | Definition |
|---|---|---|---|
| `schema_version` | `schema_version` | `string` | Canonical schema version (`1.0.0`). |
| `preflight_id` | `preflight_id` | `string` | Identifier of the measurement (sample + method). |
| `sample_id` | `sample_id` | `string` | Stable UUID joining all tables. UUIDv5 of (experiment, prompt, configuration, repetition) for experiments; of (source, provider, response id) for imports/telemetry. Idempotency key. |
| `request_id` | `request_id` | `string` | Identifier of the provider request (one billable invocation). |
| `provider` | `provider` | `string` | Provider adapter key (`anthropic`, `openai`, `gemini`, `mock-*`). |
| `requested_model` | `requested_model` | `string` | Model ID requested. |
| `preflight_input_tokens` | `preflight_input_tokens` | `int64` | Input tokens counted before dispatch. Never authoritative usage. |
| `preflight_method` | `preflight_method` | `string` | provider_native \| local_tokenizer \| statistical_approximation \| unavailable. |
| `preflight_provider` | `preflight_provider` | `string` | Who produced the count (provider endpoint or local estimator). |
| `tokenizer_name` | `tokenizer_name` | `string` | Counting endpoint or tokenizer name. |
| `tokenizer_version` | `tokenizer_version` | `string` | Tokenizer version / model whose tokenizer was used. |
| `preflight_timestamp_utc` | `preflight_timestamp_utc` | `timestamp[us, tz=UTC]` | When the count was taken (always before dispatch). |
| `preflight_error` | `preflight_error` | `string` | Why no count is available. |
| `preflight_latency_ms` | `preflight_latency_ms` | `double` | Time taken by the preflight call. |
| `preflight_includes` | `preflight_includes` | `list<item: string>` | Request parts covered by the count (system, messages, tools, thinking ...). |
| `preflight_complete` | `preflight_complete` | `bool` | True when the count covers all billable input content. |

## `observations` — ObservationRecord

Observed usage labels (F), execution results (G), cost & billing (H). Post-inference only.

| Field | Parquet column | Arrow type | Definition |
|---|---|---|---|
| `schema_version` | `schema_version` | `string` | Canonical schema version (`1.0.0`). |
| `observation_id` | `observation_id` | `string` | Identifier of the observation (one per billable invocation). |
| `sample_id` | `sample_id` | `string` | Stable UUID joining all tables. UUIDv5 of (experiment, prompt, configuration, repetition) for experiments; of (source, provider, response id) for imports/telemetry. Idempotency key. |
| `request_id` | `request_id` | `string` | Identifier of the provider request (one billable invocation). |
| `run_id` | `run_id` | `string` | UUID of the collection run / import batch / telemetry session that produced the record. |
| `parent_request_id` | `parent_request_id` | `string` | Parent invocation in a multi-call agent workflow. |
| `observation_kind` | `observation_kind` | `string` | invocation (one API call; provider iterations summed) \| workflow_aggregate. |
| `invocation_index` | `invocation_index` | `int64` | Index of the invocation within a workflow. |
| `provider` | `provider` | `string` | Provider adapter key (`anthropic`, `openai`, `gemini`, `mock-*`). |
| `requested_model` | `requested_model` | `string` | Model ID requested. |
| `resolved_model` | `resolved_model` | `string` | Model the provider reports it used (post-inference). |
| `model_version` | `model_version` | `string` | Provider model version string, when reported. |
| `source_type` | `source_type` | `string` | Provenance: controlled_experiment \| consented_api_telemetry \| authorized_log_import \| public_permitted_dataset \| synthetic_unexecuted \| test_fixture. Only the first three can carry real labels. |
| `dataset_origin` | `dataset_origin` | `string` | Human-readable origin (e.g. `experiment:<id>`, the importing system). |
| `consent_status` | `consent_status` | `string` | not_required \| granted \| denied \| unknown. Raw text is stored only with `granted` (or project-authored content) and when privacy config allows. |
| `actual_input_tokens` | `actual_input_tokens` | `int64` | Total input tokens billed, reconstructed per provider rules (cache reads/writes included). |
| `actual_output_tokens` | `actual_output_tokens` | `int64` | Total generated tokens billed, including reasoning/thinking for every provider. |
| `actual_reasoning_tokens` | `actual_reasoning_tokens` | `int64` | Reasoning/thinking tokens when reported; null when not reported. |
| `actual_total_tokens` | `actual_total_tokens` | `int64` | actual_input_tokens + actual_output_tokens. |
| `actual_cached_input_tokens` | `actual_cached_input_tokens` | `int64` | Input tokens read from cache. |
| `actual_cache_creation_tokens` | `actual_cache_creation_tokens` | `int64` | Input tokens written to cache (Anthropic); null where not reported. |
| `actual_uncached_input_tokens` | `actual_uncached_input_tokens` | `int64` | Input tokens neither read from nor written to cache. |
| `actual_visible_output_tokens` | `actual_visible_output_tokens` | `int64` | Output tokens excluding reasoning (Gemini: candidates). |
| `actual_tool_input_tokens` | `actual_tool_input_tokens` | `int64` | Tool-use prompt tokens reported separately (Gemini tool_use_prompt_token_count). |
| `actual_tool_output_tokens` | `actual_tool_output_tokens` | `int64` | Tool output tokens reported separately, when available. |
| `input_usage_semantics` | `input_usage_semantics` | `string` | Provider rule used to derive input fields. |
| `output_usage_semantics` | `output_usage_semantics` | `string` | Provider rule used to derive output fields. |
| `reasoning_included_in_output` | `reasoning_included_in_output` | `bool` | Whether the provider's own output field includes reasoning (OpenAI/Anthropic true, Gemini false). |
| `provider_total_tokens` | `provider_total_tokens` | `int64` | Total reported by the provider (null for Anthropic). |
| `normalized_total_tokens` | `normalized_total_tokens` | `int64` | Seshat's total from documented definitions (cross-checked against provider_total_tokens). |
| `usage_iteration_count` | `usage_iteration_count` | `int64` | Sampling iterations in the call (e.g. Anthropic compaction). |
| `label_quality` | `label_quality` | `string` | verified \| partial \| censored \| inconsistent \| unavailable \| synthetic. |
| `usage_observation_source` | `usage_observation_source` | `string` | Which provider usage object the labels came from. |
| `raw_usage` | `raw_usage_json` | `string` | Original provider usage payload, unmodified (JSON). |
| `response_id` | `response_id` | `string` | Provider response id. |
| `response_status` | `response_status` | `string` | completed \| incomplete \| refused \| failed \| uncertain \| not_executed. |
| `stop_reason` | `stop_reason` | `string` | Provider stop/finish reason. |
| `is_truncated` | `is_truncated` | `bool` | Hit the output limit; the output target is right-censored. |
| `first_token_latency_ms` | `first_token_latency_ms` | `double` | Time to first token (streaming), when measured. |
| `total_latency_ms` | `total_latency_ms` | `double` | Wall-clock latency of the call. |
| `client_tool_call_count` | `client_tool_call_count` | `int64` | Client tool calls requested in the response. |
| `server_tool_call_count` | `server_tool_call_count` | `int64` | Server-side tool calls (e.g. web search) in the response. |
| `retry_count` | `retry_count` | `int64` | Retries before the final outcome (rate limits, overload). |
| `rate_limit_encountered` | `rate_limit_encountered` | `bool` | Any attempt was rate limited. |
| `error_type` | `error_type` | `string` | Error category for failed/uncertain calls. |
| `error_code` | `error_code` | `string` | HTTP status or provider error code. |
| `completed_at_utc` | `completed_at_utc` | `timestamp[us, tz=UTC]` | When the response completed. |
| `pricing_version` | `pricing_version` | `string` | Price-table version used (configs/pricing/*.json). |
| `pricing_effective_date` | `pricing_effective_date` | `date32[day]` | Effective-from date of the price entry. |
| `pricing_source` | `pricing_source` | `string` | Official pricing page URL (or `fixture`). |
| `input_rate_usd_per_million` | `input_rate_usd_per_million` | `decimal128(24, 12)` | Input rate applied (USD / 1M tokens). |
| `output_rate_usd_per_million` | `output_rate_usd_per_million` | `decimal128(24, 12)` | Output rate applied (USD / 1M tokens). |
| `cache_read_rate_usd_per_million` | `cache_read_rate_usd_per_million` | `decimal128(24, 12)` | Cache-read rate applied. |
| `cache_write_rate_usd_per_million` | `cache_write_rate_usd_per_million` | `decimal128(24, 12)` | Cache-write rate applied (5m or 1h). |
| `calculated_cost_usd` | `calculated_cost_usd` | `decimal128(24, 12)` | Price-table estimate of the call's cost. Not an invoice charge. |
| `provider_reported_cost_usd` | `provider_reported_cost_usd` | `decimal128(24, 12)` | Cost reported by the provider, when available. |
| `currency` | `currency` | `string` | Always USD. |
| `cost_method` | `cost_method` | `string` | price_table_estimate \| provider_reported \| unavailable. |
| `billing_dimensions` | `billing_dimensions_json` | `string` | Extra billing facts: cache TTL breakdown, server tool use, tiers, accounting issues, fixture flags (JSON). |
| `collected_at_utc` | `collected_at_utc` | `timestamp[us, tz=UTC]` | When the observation was recorded. |
| `retain_until` | `retain_until` | `timestamp[us, tz=UTC]` | Expiry for privacy retention; null = kept. `purge-expired` deletes rows past it. |

## `predictions` — PredictionRecord

Seshat predictions. Never observations or labels.

| Field | Parquet column | Arrow type | Definition |
|---|---|---|---|
| `schema_version` | `schema_version` | `string` | Canonical schema version (`1.0.0`). |
| `prediction_id` | `prediction_id` | `string` | Identifier of a prediction (deterministic from request and predictor version). |
| `request_id` | `request_id` | `string` | Identifier of the provider request (one billable invocation). |
| `created_at_utc` | `created_at_utc` | `timestamp[us, tz=UTC]` | When the prediction was made. |
| `predictor_version` | `predictor_version` | `string` | Predictor that produced the value. |
| `target` | `target` | `string` | Predicted quantity. |
| `predicted_output_tokens` | `predicted_output_tokens` | `int64` | Point prediction. |
| `lower_bound` | `lower_bound` | `int64` | Lower interval bound. |
| `upper_bound` | `upper_bound` | `int64` | Upper interval bound. |
| `confidence` | `confidence` | `double` | Predictor-reported confidence (not calibrated unless interval_calibrated). |
| `interval_calibrated` | `interval_calibrated` | `bool` | Whether the interval was empirically calibrated. |
| `prediction_mode` | `prediction_mode` | `string` | preflight (before dispatch) \| backtest_holdout \| backtest_out_of_fold \| retrospective (stored request re-scored; in-sample unless stated). |
| `evaluation_role` | `evaluation_role` | `string` | protected_test \| cross_validation \| none: how the row may be used in quality reporting. |
| `base_predicted_output_tokens` | `base_predicted_output_tokens` | `int64` | Point forecast before bias correction (stage A). |
| `confidence_level` | `confidence_level` | `double` | Nominal coverage targeted by the calibrated interval (e.g. 0.90). Not a probability that the point forecast is right. |
| `lower_margin_tokens` | `lower_margin_tokens` | `int64` | Point forecast minus interval lower bound. |
| `upper_margin_tokens` | `upper_margin_tokens` | `int64` | Interval upper bound minus point forecast. |
| `confidence_margin_tokens` | `confidence_margin_tokens` | `int64` | Display margin: max(lower, upper margin), so point +/- margin covers the calibrated interval. |
| `lower_margin_pct` | `lower_margin_pct` | `double` | lower_margin_tokens as % of the point forecast. |
| `upper_margin_pct` | `upper_margin_pct` | `double` | upper_margin_tokens as % of the point forecast. |
| `confidence_margin_pct` | `confidence_margin_pct` | `double` | confidence_margin_tokens as % of the point forecast. |
| `predicted_reasoning_tokens` | `predicted_reasoning_tokens` | `int64` | Forecast reasoning tokens, when the predictor models them. |
| `predicted_total_tokens` | `predicted_total_tokens` | `int64` | Forecast realized total tokens: preflight input + min(output forecast, max_output_tokens). |
| `predicted_cost_usd` | `predicted_cost_usd` | `decimal128(24, 12)` | Price-table cost of the forecast realized usage. Not an invoice. |
| `uncertainty_model_version` | `uncertainty_model_version` | `string` | Version of the residual/scale model behind the interval. |
| `calibration_version` | `calibration_version` | `string` | Version of the bias-correction + conformal calibration layer. |
| `dataset_version` | `dataset_version` | `string` | Dataset the predictor was trained on (hash of observation ids). |
| `feature_set_version` | `feature_set_version` | `string` | Model feature definition version. |
| `code_revision` | `code_revision` | `string` | Git commit that produced the prediction. |

## Normalized usage

Provider rules (from official documentation, checked 2026-10-10). Totals are never
computed by adding every available field.

| Provider | actual_input_tokens | actual_output_tokens | actual_reasoning_tokens | totals |
|---|---|---|---|---|
| OpenAI (Responses) | `input_tokens` (cached_tokens is a subset) | `output_tokens` (includes reasoning) | `output_tokens_details.reasoning_tokens` | normalized = input + output; checked against `total_tokens` |
| Anthropic (Messages) | `input_tokens` + `cache_read_input_tokens` + `cache_creation_input_tokens` (input_tokens = tokens after the last cache breakpoint) | `output_tokens` (includes thinking) | `output_tokens_details.thinking_tokens` | no provider total; with compaction, sum over `usage.iterations` (top level excludes compaction) |
| Gemini (generate_content) | `prompt_token_count` (includes cached content) | `candidates_token_count` + `thoughts_token_count` | `thoughts_token_count` | total = prompt + candidates + thoughts; `tool_use_prompt_token_count` kept separately, never added |

Contradictions (cached > input, reasoning > output, total mismatch, cache breakdown
mismatch, negative values) are recorded in `billing_dimensions.accounting_issues`,
the label becomes `inconsistent`, and validation quarantines the row. The original
payload is always kept in `raw_usage`.

## Label quality

| Value | Meaning | Training |
|---|---|---|
| verified | complete usage, accounting consistent | eligible |
| partial | primary labels present, some secondary fields unreported | eligible |
| censored | truncated at the output limit (right-censored) | eligible, flagged `target_is_censored` |
| inconsistent | contradictory provider accounting | quarantined |
| unavailable | failed / uncertain / no usage | excluded |
| synthetic | mock-provider test fixture | only with `--label-origin synthetic`, never with real labels |
