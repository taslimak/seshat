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
| Writer | `.github/workflows/consolidate-artifacts.yml` (free; no provider calls) | Runs automatically after each successful *Manual calibration data batch* and on demand for older run IDs. |

## One-time setup (owner)

1. Create a **private** repository, e.g. `taslimak/seshat-data` (empty is fine).
2. Create a fine-grained personal access token restricted to that one
   repository with **Contents: Read and write** (Metadata: Read is implied).
3. In `taslimak/seshat` → Settings → Secrets and variables → Actions:
   * secret `SESHAT_DATA_STORE_TOKEN` = the token;
   * variable `SESHAT_DATA_STORE_REPO` = `taslimak/seshat-data`.
4. Run **Consolidate calibration artifacts** manually with
   `run_ids = 38106728852` **before 2026-11-10**.

Until steps 1–3 are done the workflow exits successfully with a notice and
imports nothing. The workflow refuses to write to a public store repository.

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
