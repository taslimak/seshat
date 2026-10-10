# Data

Generated raw and model-ready datasets are gitignored. The repository commits schemas, experiment configs, a reviewed seed prompt bank, tests, and a small CSV preview.

- `raw/`: immutable JSONL observations
- `manifests/`: planned runs
- `normalized/`: provider-normalized Parquet
- `features/`: leakage-safe feature matrices + targets
- `splits/`: prompt-family split assignments
- `profiles/`: dataset health reports
- `releases/`: frozen versioned train/validation/calibration/test datasets
- `quarantine/`: anomalous or invalid records retained (sanitized) for inspection
- `store/`: versioned record store (`request_features/`, `observed_usage/`, `prediction/` Parquet parts + `_ingestion_log.jsonl`)
- `records.csv`: human-readable CSV view of the **synthetic** example observations

Regenerate `records.csv` (do not edit it by hand):

```bash
seshat-data ingest --store /tmp/example-store --record-type observed_usage --input examples/synthetic_v1/observed_usage.jsonl
seshat-data export-csv --input /tmp/example-store/observed_usage --output data/records.csv
```

Do not manually edit CSV rows and use them as training truth.
