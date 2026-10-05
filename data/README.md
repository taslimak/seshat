# Data

Generated raw and model-ready datasets are gitignored. The repository commits schemas, experiment configs, a reviewed seed prompt bank, tests, and a small CSV preview.

- `raw/`: immutable JSONL observations
- `manifests/`: planned runs
- `normalized/`: provider-normalized Parquet
- `features/`: leakage-safe feature matrices + targets
- `splits/`: prompt-family split assignments
- `profiles/`: dataset health reports
- `releases/`: frozen versioned train/validation/calibration/test datasets
- `quarantine/`: anomalous observations retained for inspection
- `records.csv`: human-readable preview generated from canonical records

Do not manually edit CSV rows and use them as training truth.
