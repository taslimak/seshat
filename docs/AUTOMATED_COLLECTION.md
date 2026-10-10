# Automated collection and shared billing guards

The workflow `.github/workflows/scheduled-collection.yml` runs at 9:17 AM
and 9:17 PM America/New_York, using the **new schema 1.0 pipeline**.

## Safety defaults

- Scheduled runs use mock providers ONLY. No provider API keys are injected.
- Each scheduled mock run creates an Actions artifact for inspection, retained
  for 30 days. GitHub Actions runner storage is ephemeral.
- Mock requests do not produce ground-truth API labels.
- Live calls require explicit `--execute-live --max-cost-usd` authorization.
- Set `SESHAT_UNATTENDED_COLLECTION=1` for any unattended paid collector.
  In that mode, the collector refuses to run without
  `SESHAT_SHARED_LEDGER_PATH`.
- For a shared ledger, also set `SESHAT_DAILY_COST_CAP_USD` and
  `SESHAT_MONTHLY_COST_CAP_USD`. All experiments must mount the **same durable
  local SQLite database path**. A path within ephemeral GitHub Actions runner
  storage does **not** provide global spending limits.
- Each attempt consumes a request quota, including provider-rejected attempts.
- Network timeouts/uncertain outcomes retain conservative cost reservations
  and are not automatically retried.
- SQLite represents projected cost, not an authoritative provider charge.
  Reconcile invoices manually; provider billing caps are an additional defense.

## Production activation blocker

**Do not enable unattended paid collection on GitHub Actions yet.**
Provision a persistent single-writer execution environment or a transactional
shared database and durable dataset object storage first. Then configure
the daily/monthly caps, keys, allowed models, and limited experiments.
SQLite on a shared network filesystem is not automatically safe for
multi-host writers. GitHub artifact retention does not provide durable training
data storage.

PR #7 used the older experiment API. This workflow supersedes its integration,
and PR #7 must not be merged independently.
