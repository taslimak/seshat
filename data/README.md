# Data

Collected and generated datasets are **not** committed. Only this README and
`fixtures/` (small, mock-provider outputs safe to publish) are tracked; everything
else under `data/` is gitignored.

| Path | Written by | Contents |
|---|---|---|
| `raw/<experiment_id>/` | `seshat-data collect` | `events.jsonl` (append-only event log with raw usage), `plan.jsonl`, `design_report.json`, `runs/*.log.jsonl`, `dry_run/` |
| `raw/imports/<batch_id>/` | `seshat-data import --kind usage-log` | sanitized rows, `rejections.jsonl`, `import_report.json` |
| `raw/generated/` | `seshat-data generate` | generated prompt sets (unexecuted, unlabeled) |
| `raw/prompt_content/` | collector / importers | prompt text, **only** when the privacy config allows it |
| `normalized/<table>/date=…/provider=…/` | `collect`, `normalize`, `import` | canonical Parquet: `requests`, `preflight`, `observations`, `predictions` |
| `normalized/_validation/` | `seshat-data validate` | `decisions.parquet`, `quality_report.md`, `quality_report.json` |
| `training/dataset_v<version>[_synthetic]/` | `seshat-data prepare-training` | `train/validation/calibration/test.parquet`, `manifest.json`, `schema.json`, `split_stats.json` |
| `exports/` | `seshat-data export` | `observations.csv`, `data_dictionary.md` |
| `fixtures/offline_example/` | committed | results of the default offline run (see below) |

## `fixtures/offline_example/`

Produced by the default offline workflow (`configs/data_collection.yaml`, mock
providers only, `source_type=test_fixture`, `label_quality=synthetic`). It shows
the output formats; it is **not** real usage data and must not be used to judge
model accuracy.

```bash
seshat-data collect --config configs/data_collection.yaml
seshat-data validate
seshat-data export --format csv --limit 25 --output data/exports/sample.csv
seshat-data prepare-training --version 0.1.0 --label-origin synthetic
seshat-data stats > collection_stats.json
```

Files: `sample_export.csv` (first 25 observations), `data_dictionary.md`,
`quality_report.{md,json}`, `design_report.json`, `split_stats.json`,
`collection_stats.json`. Do not edit CSV rows by hand or use them as training truth.
See [docs/DATA_COLLECTION_GUIDE.md](../docs/DATA_COLLECTION_GUIDE.md).
