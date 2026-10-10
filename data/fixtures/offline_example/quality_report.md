# Seshat dataset quality report

Generated 2026-10-10T21:35:38.986663+00:00.

## Totals

| Metric | Value |
|---|---:|
| request records | 420 |
| observations | 420 |
| eligible labeled | 420 |
| eligible real labels | 0 |
| eligible synthetic labels | 420 |
| quarantined | 0 |
| rejected | 0 |
| warnings | 63 |
| missing label pct | 0.0 |

## Decisions

| Value | Count |
|---|---:|
| PASS | 357 |
| WARNING | 63 |

## Validation flags

| Value | Count |
|---|---:|
| censored_output | 63 |

## Label quality

| Value | Count |
|---|---:|
| synthetic | 420 |

## Source type

| Value | Count |
|---|---:|
| test_fixture | 420 |

## Provider / model

| Value | Count |
|---|---:|
| mock-anthropic/mock-claude | 133 |
| mock-gemini/mock-gemini | 141 |
| mock-openai/mock-gpt | 146 |

## Prompt category

| Value | Count |
|---|---:|
| classification | 24 |
| code_generation | 21 |
| code_review | 27 |
| creative_writing | 27 |
| debugging | 23 |
| extraction | 27 |
| long_context_analysis | 25 |
| mathematics | 23 |
| multi_turn | 21 |
| question_answering | 23 |
| reasoning | 25 |
| short_answer | 25 |
| structured_json | 29 |
| summarization | 23 |
| tool_calling | 23 |
| translation | 23 |
| verbose_explanation | 31 |

## Effort

| Value | Count |
|---|---:|
| high | 138 |
| low | 139 |
| medium | 73 |
| minimal | 70 |

## Approximate spend (USD) by source type

| Value | Count |
|---|---:|
| test_fixture | 1.796525 |

_Price-table estimates. test_fixture spend uses fixture prices and is not real money._

## Token distributions

| Field | count | mean | p10 | p50 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| actual_input_tokens | 420 | 806.77 | 55.0 | 360.0 | 1927.900000000005 | 7518.0 |
| actual_output_tokens | 420 | 389.78 | 67.80000000000001 | 320.0 | 750.4000000000001 | 3660.0 |
| actual_reasoning_tokens | 420 | 159.76 | 0.0 | 79.5 | 512.0 | 2271.0 |
| actual_total_tokens | 420 | 1196.56 | 238.00000000000006 | 803.5 | 2497.7000000000007 | 8417.0 |

## Cache usage

```json
{
  "cache_write_observations": 16,
  "cached_fraction_of_input": {
    "count": 26,
    "max": 0.8,
    "mean": 0.68,
    "p10": 0.5263157894736842,
    "p50": 0.7498257839721254,
    "p90": 0.7998405103668261
  },
  "observations_with_cache_reads": 26,
  "share_with_cache_reads": 0.0619
}
```

## Duplicates

```json
{
  "duplicate_observation_ids": 0,
  "duplicate_prompt_content_rate": 0.7238,
  "note": "Repeated observations of the same prompt (by design) count as duplicate content; group-aware splitting keeps them together. Near-duplicate prompt text is detected at the prompt-bank stage (seshat-data prompts check).",
  "prompt_hashes_in_multiple_families": 0
}
```

## Preflight vs observed input (relative)

```json
{
  "count": 420,
  "max": 0.0,
  "mean": 0.0,
  "p10": 0.0,
  "p50": 0.0,
  "p90": 0.0
}
```
