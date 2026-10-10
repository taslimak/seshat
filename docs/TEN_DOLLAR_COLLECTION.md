# Seshat $10/month scheduled API collection

The scheduled collection runs at **9:17 AM and 9:17 PM America/New_York**
using the workflow in `.github/workflows/scheduled-collection.yml`.

## Spending and sampling

- Monthly shared estimated-cost cap: **$10.00**.
- Daily shared estimated-cost cap: **$0.30**.
- Per-run estimated-cost cap: **$0.15**.
- At most **4 attempted API requests per run**, with a maximum of
  **256 output tokens each** and **no automatic retries**.
- Generated offline template prompts are rotated by date and morning/evening
  slots. No LLM API calls are needed to create prompts.
- The collector records actual provider usage when live responses succeed.
- These are price-table estimates, **not billing guarantees**. Use provider-side
  budget controls and monitor invoices separately.

## Activation (currently OFF)

The job `paid-collection` executes only when repository variable
`SESHAT_ENABLE_SCHEDULED_PAID=true`. Do not enable it until ALL of the
following are in place:

1. A dedicated, trusted **self-hosted Linux Actions runner** with label
   `seshat-persistent`. It must have a private, backed-up persistent
   filesystem, and be restricted to trusted repository workflows.
2. Set repository variables `SESHAT_PERSISTENT_DATA_ROOT` to a real absolute
   mounted data directory, `SESHAT_SHARED_LEDGER_PATH` to a SQLite file
   within that directory, `SESHAT_COLLECTION_PROVIDER` to one of
   `openai`, `anthropic`, `gemini`, and `SESHAT_COLLECTION_MODEL` to a
   verified, supported, low-cost API model. Storage is shared between runs.
3. Verify restart persistence, backups, and directory permissions. Only after
   that, create the marker file
   `$SESHAT_PERSISTENT_DATA_ROOT/.seshat-persistent-ready`.
4. Set the matching GitHub Actions secret `OPENAI_API_KEY`,
   `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`, plus
   `SESHAT_PROMPT_HASH_KEY`. Never hardcode or commit secret values.
5. Set a provider-side spending limit/alert where supported, complete a
   supervised live pilot, and verify usage normalization and measured costs.
6. Finally set `SESHAT_ENABLE_SCHEDULED_PAID=true`.

Until enabled, the workflow runs **mock-only** experiments on hosted runners.
The paid job does not execute. There is no automatic provider API collection
yet. The previous separate manual pilot PR #9 may be used for a supervised test
after review and merge.

## Operational notes

- GitHub schedule timing is best-effort rather than exact.
- SQLite is appropriate only when all jobs access the same durable
  single-machine filesystem and share a single writer/transactional file.
  Do not locate the ledger in the ephemeral GitHub workspace or use
  network-shared SQLite without evaluating its locking guarantees.
- Run concurrency is serialized by the workflow's concurrency group.
- Provider subscriptions and API usage are billed independently.
- Do not merge the superseded old scheduled collector from PR #7.
