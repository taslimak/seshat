# CSV data dictionary

One row per observation (one billable model invocation). Blank = not reported / not
applicable; 0 = observed zero. Full schema and nested fields: docs/DATA_SCHEMA.md.

| Column | Source field | Definition |
|---|---|---|
| provider | `provider` | Provider adapter name (mock-* are offline test fixtures). |
| model | `requested_model` | Model ID requested. |
| resolved_model | `resolved_model` | Model the provider reports it used (post-inference). |
| prompt_category | `prompt_category` | Task category of the prompt. |
| prompt_char_count | `prompt_char_count` | Characters in system prompt + all messages. |
| message_count | `message_count` | Messages in the request (excluding the system prompt). |
| effort | `requested_effort` | Requested reasoning/thinking effort (blank = not set). |
| max_output_tokens | `max_output_tokens` | Output token limit sent with the request. |
| tool_count | `tool_count` | Number of tools offered. |
| caching_enabled | `caching_enabled` | Prompt caching requested. |
| preflight_input_tokens | `preflight_input_tokens` | Input tokens counted before dispatch (not authoritative). |
| preflight_method | `preflight_method` | provider_native \| local_tokenizer \| statistical_approximation \| unavailable. |
| actual_input_tokens | `actual_input_tokens` | Observed total input tokens (cache reads/writes included; provider-specific rules in DATA_SCHEMA.md). |
| actual_cached_input_tokens | `actual_cached_input_tokens` | Observed input tokens served from cache. |
| actual_output_tokens | `actual_output_tokens` | Observed billed output tokens, including reasoning/thinking. |
| actual_reasoning_tokens | `actual_reasoning_tokens` | Observed reasoning/thinking tokens (blank = not reported). |
| actual_total_tokens | `actual_total_tokens` | actual_input_tokens + actual_output_tokens. |
| cost_usd | `calculated_cost_usd` | Price-table estimate in USD (not an invoice; fixture prices for mock providers). |
| cost_method | `cost_method` | price_table_estimate \| provider_reported \| unavailable. |
| latency_ms | `total_latency_ms` | Wall-clock request latency in milliseconds. |
| response_status | `response_status` | completed \| incomplete \| refused \| failed \| uncertain. |
| is_truncated | `is_truncated` | Output hit the token limit (target is right-censored). |
| label_quality | `label_quality` | verified \| partial \| censored \| inconsistent \| unavailable \| synthetic. |
| source_type | `source_type` | controlled_experiment \| consented_api_telemetry \| authorized_log_import \| test_fixture ... |
| validation_decision | `decision` | PASS \| WARNING \| QUARANTINE \| REJECT from `validate`. |
| collected_at_utc | `collected_at_utc` | When the observation was collected (UTC). |
| sample_id | `sample_id` | Stable sample identifier (joins all tables). |
| request_id | `request_id` | Request identifier. |

Not exported (see Parquet): raw_usage_json (original provider usage payload),
billing_dimensions_json (cache TTL breakdown, accounting issues, tiers),
provider_settings_json,
attachment/image metadata lists and all hashes.
