# Scheduled data collection

This branch adds `.github/workflows/scheduled-collection.yml`.
The job runs every day at **9:17 AM and 9:17 PM America/New_York** after merging to the default branch. GitHub may delay or skip scheduled jobs; it is a scheduler, not an exact-time SLA.

## Important safeguards
- Default execution is **mock only** and **does not create real token-usage labels**.
- Maximum **2 API requests per scheduled batch**, i.e. normally at most 4 requests per day, with max output 256 tokens per request.
- The job makes no automatic retries on paid calls.
- Paid execution is explicitly gated behind variables: `SESHAT_ENABLE_PAID_COLLECTION=true` and `SESHAT_PROVIDER_BUDGET_CONFIRMED=yes`.
- Specify `SESHAT_COLLECTION_PROVIDER=openai` (or `anthropic`) and a real, currently supported model in `SESHAT_COLLECTION_MODEL`.
- Configure the corresponding API key using a repository Actions **secret**.
- Before enabling paid mode, establish spending controls with the provider. **This implementation does not calculate or enforce a dollar/month budget.** Request limits and output ceilings only bound nominal work, not billed dollars.
- Personal ChatGPT or Claude subscriptions are not the same as API credits. A provider API account with available billing is needed.
- Provider availability/model names must be confirmed before opting into live runs.

## How to start
1. Merge this branch to the default branch after CI review.
2. Go to **Actions → Scheduled Seshat data collection → Run workflow** and select `mock`.
3. Inspect the artifact and its `summary.json`, `records.csv`, and `validated.parquet`.
4. Confirm live API connectivity manually in `paid` mode before allowing scheduled paid runs.
5. Keep `SESHAT_ENABLE_PAID_COLLECTION` unset or false until that point.

## Retention and limitations
This initial implementation uploads each batch to GitHub Actions artifacts with **90 days' retention**. Artifacts are not an append-only durable dataset. The pipeline cannot accumulate a permanent history until an S3/GCS/R2 bucket or equivalent append-only data store is configured, with lifecycle/retention policies, deduplication across batches, and budget reconciliation. Do **not** assume multiple scheduled runs form one merged historical dataset.

Run locally:
```bash
pip install -e ".[data,providers]"
python scripts/scheduled_collect.py --mode mock --output-dir data/scheduled/manual
```

The collection script selects two prompts from the current seed bank each run. This is only a smoke-test collector; improving sample coverage and active-learning selection is future work.
