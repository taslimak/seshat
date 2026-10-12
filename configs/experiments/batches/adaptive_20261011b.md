# Dry-run plan: `adaptive_20261011b`

**No requests have been sent.** Paid API calls. Requires explicit approval and a manual workflow dispatch with approve_paid_call=true. No request has been sent.

- Target: `openai/gpt-6.1-sol` (pricing 2026-10-10: $2.00 in / $10.00 out per 1M tokens)
- Efforts (balanced across prompts): low, medium, high
- Planned requests: **200** (182 base + repetitions of 10% of cells x2)
- Expected estimated cost: **$1.67** (hard cap $2.00; uncapped worst case $3.88)
- Expected new usable (uncensored) rows: **190** (~$0.0088 per usable row)
- Store observations used for the length models: 222

_The collector reserves the worst case per request and stops before the estimated spend could exceed $2.00; provider billing can differ._

| Category | Observed rows / usable / censor | Model median (src) | Cap | Pred. censor | Requests | Exp. usable | Exp. $/req | Why |
|---|---|---:|---:|---:|---:|---:|---:|---|
| classification | 13 / 13 / 0% | 52 (fit) | 512 | 2% | 22 | 21.6 | 0.0016 | forecast-quality priority x1.93 |
| code_generation | 13 / 0 / 100% | 2415 (fit) | 4096 (top of ladder) | 25% | 12 | 9.0 | 0.0274 | observed censor rate 100% (truncated at cap 512) -> cap 4096; no usable labels yet; forecast-quality priority x2.26 |
| code_review | 13 / 0 / 100% | 1997 (fit) | 4096 (top of ladder) | 16% | 14 | 11.7 | 0.0241 | observed censor rate 100% (truncated at cap 512) -> cap 4096; no usable labels yet; forecast-quality priority x2.26 |
| creative_writing | 13 / 4 / 69% | 727 (fit) | 4096 | 5% | 7 | 6.6 | 0.0129 | observed censor rate 69% (truncated at cap 512) -> cap 4096; forecast-quality priority x1.36 |
| debugging | 13 / 13 / 0% | 284 (fit) | 1024 | 0% | 7 | 7.0 | 0.0042 | forecast-quality priority x1.22 |
| extraction | 16 / 16 / 0% | 107 (fit) | 512 | 0% | 10 | 10.0 | 0.0029 | forecast-quality priority x1.38 |
| long_context_analysis | 9 / 8 / 11% | 308 (fit) | 2048 | 0% | 7 | 7.0 | 0.0111 | observed censor rate 11% (truncated at cap 512) -> cap 2048; forecast-quality priority x1.58 |
| mathematics | 15 / 7 / 53% | 665 (fit) | 4096 | 1% | 8 | 7.9 | 0.0099 | observed censor rate 53% (truncated at cap 512) -> cap 4096; forecast-quality priority x1.16 |
| multi_turn | 12 / 11 / 8% | 172 (fit) | 1024 | 0% | 12 | 11.9 | 0.0031 | forecast-quality priority x1.62 |
| question_answering | 15 / 13 / 13% | 230 (fit) | 1024 | 2% | 7 | 6.8 | 0.0049 | observed censor rate 13% (truncated at cap 512) -> cap 1024; forecast-quality priority x1.25 |
| reasoning | 12 / 9 / 25% | 497 (fit) | 1024 | 1% | 7 | 7.0 | 0.0068 | observed censor rate 25% (truncated at cap 512) -> cap 1024; forecast-quality priority x1.08 |
| short_answer | 13 / 13 / 0% | 20 (fit) | 512 | 0% | 24 | 24.0 | 0.0005 | forecast-quality priority x1.87 |
| structured_json | 16 / 16 / 0% | 98 (fit) | 512 | 0% | 6 | 6.0 | 0.0028 | forecast-quality priority x1.27 |
| summarization | 13 / 13 / 0% | 117 (fit) | 512 | 0% | 10 | 10.0 | 0.0034 | forecast-quality priority x1.37 |
| tool_calling | 9 / 9 / 0% | 63 (fit) | 512 | 0% | 23 | 23.0 | 0.0018 | forecast-quality priority x1.50 |
| translation | 15 / 9 / 40% | 497 (fit) | 4096 | 2% | 12 | 11.8 | 0.0097 | observed censor rate 40% (truncated at cap 512/2048) -> cap 4096; forecast-quality priority x2.26 |
| verbose_explanation | 12 / 0 / 100% | 2616 (fit) | 4096 (top of ladder) | 29% | 12 | 8.5 | 0.0291 | observed censor rate 100% (truncated at cap 512) -> cap 4096; no usable labels yet; forecast-quality priority x2.26 |

## Readiness projection

Usable rows now: 154; expected after this batch: ~344. `real_label` split ratios need about 167 usable rows (more if groups are uneven).

Model notes: per-category log-normal output length, right-censored at the cap (truncated responses are lower bounds), MAP with weak priors; effort multipliers are planning assumptions until effort levels are observed.
