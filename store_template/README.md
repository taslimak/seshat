# seshat-data (private)

Durable store of real Seshat observations. **Keep this repository private**:
`raw/*/plan.jsonl` contains the exact request messages sent to providers.

Managed by `.github/workflows/sync-calibration-artifacts.yml`, which imports
`seshat-calibration-*` artifacts from `taslimak/seshat` (every 6 hours, on
push of `SESHAT_REF`/`SYNC_REQUEST`, or manually with specific run IDs) using
only the built-in `GITHUB_TOKEN`. `SESHAT_REF` pins the Seshat code revision
used for consolidation.

| Path | Contents |
|---|---|
| `STORE.json` | store kind (`real`) |
| `raw/<experiment>/` | merged `plan.jsonl`, `events.jsonl` (raw provider usage), run logs |
| `normalized/` | Parquet tables rebuilt from `raw/` on every import |
| `_provenance/imports.jsonl` | one ledger entry per imported artifact |
| `_provenance/archives/` | original artifact ZIPs (digest-verified) |
| `reports/` | consolidated CSV, quality, readiness, spend-by-month |

See `docs/DATA_STORE.md` in the Seshat repository.
