# Durable observation store

GitHub Actions artifacts are **not** storage: they expire after 30 days
(`seshat-calibration-38106728852` expires **2026-11-10**) and each run's
artifact is an independent `data/` tree. Real observations are therefore
consolidated into one long-lived **store**.

## Strategy

| Decision | Choice | Why |
|---|---|---|
| Medium | A **private** GitHub repository (e.g. `taslimak/seshat-data`) holding the store as plain files | Free, versioned (every import is a commit), access-controlled, no new paid infrastructure. A batch adds ~0.3–0.5 MB including the original ZIP. |
| Not the main repo | `taslimak/seshat` is **public**; `raw/*/plan.jsonl` contains the exact request messages | Prompt text must not be published. (Note: artifacts and job logs of a public repo are already visible to signed-in GitHub users; keep secrets and private prompts out of them.) |
| Unit of truth | The raw layer: `plan.jsonl`, `events.jsonl` (with raw provider usage), run logs, the original artifact ZIP | Normalized Parquet is rebuilt from raw on every import, so it is a deterministic function of the raw layer. |
| Writer | A workflow **inside the store repository** (`store_template/.github/workflows/sync-calibration-artifacts.yml`) | Reads the public `taslimak/seshat` runs and artifacts and commits to its own repo with the built-in `GITHUB_TOKEN`: no personal access token, no cross-repo secret, no provider calls, no spend (it uses ~2–3 private-repo Actions minutes per sync from the free allowance). |

## Setup

1. Create an empty **private** repository `taslimak/seshat-data` (with a README).
2. Copy `store_template/` into it (`README.md`, `SESHAT_REF`, the workflow).
   `SESHAT_REF` pins the Seshat branch/commit used for consolidation (`main`
   once this code is merged).
3. Pushing the workflow triggers the first sync. Afterwards it runs every
   6 hours, on pushes to `SESHAT_REF`/`SYNC_REQUEST`, and on manual dispatch
   (optionally with specific run IDs). Each sync imports every successful
   calibration run's artifact not yet in the ledger and warns about expired
   ones.

The workflow refuses to run if the store repository is public.

## What an import guarantees

`seshat-data consolidate --source <zip-or-dir> --store <root>`:

* **Idempotent.** An import is keyed by the SHA-256 of the ZIP (or of a
  directory's file list + hashes). Re-importing is a no-op.
* **Stable de-duplication.** Plan rows merge by `sample_id` (a deterministic
  UUID of experiment, prompt, configuration and repetition); a sample defined
  differently in store and source aborts the import. Events merge by the
  SHA-256 of their canonical JSON, so overlapping artifacts never double-count.
* **Provenance.** `_provenance/imports.jsonl` records source digest, GitHub
  run/artifact ids and digest (verified against the API), per-file hashes,
  per-experiment new/duplicate event counts, code revision and pricing
  versions; the original ZIP is kept under `_provenance/archives/` and the
  run's config/stats under `_provenance/source_files/`.
* **Consistency check.** Rebuilt normalized row counts are compared with the
  artifact's own normalized tables (`artifact_rows_match`).
* **Fixture isolation.** `STORE.json` fixes the store kind (`real` or
  `fixture`). Mock-provider experiments cannot enter a real store, real ones
  cannot enter a fixture store, and mixed experiments are refused. Validation
  additionally rejects mock providers labelled as anything but `test_fixture`.
* **Prompt protection.** `raw/prompt_content/` is never imported; reports and
  the consolidated CSV contain identifiers, counts and metadata only.
* **All-or-nothing, locked.** Every check runs before the first write; files
  are replaced atomically under an exclusive lock.

After each import, `reports/` is regenerated: `observations_consolidated.csv`
(one row per observation for human inspection), `quality_report.md`,
`collection_stats.json`, `spend_by_month.json` (price-table estimates, not
invoices) and `training_readiness.{json,md}`.

## Manual / local use

```bash
git clone https://github.com/taslimak/seshat-data store          # private
python scripts/sync_calibration_artifacts.py --store store        # needs gh auth
# or, from a downloaded ZIP:
seshat-data consolidate --store store \
  --source ~/Downloads/seshat-calibration-38106728852.zip \
  --meta github_run_id=38106728852 --meta artifact_id=11690398729
seshat-data readiness --root store --markdown
seshat-data evaluate-baseline --root store --output-dir store/reports
```

Training reads the store directly: `seshat-data prepare-training --root store
--version <v> --ratios real_label`.

## Restore and audit

The store is reproducible from `_provenance/archives/*.zip`: re-importing them
in ledger order into an empty store yields identical raw files and normalized
tables. Git history gives point-in-time snapshots for dataset lineage.
