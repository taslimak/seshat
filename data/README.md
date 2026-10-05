# Data

Dataset contents are intentionally not committed yet.

See the canonical implementation guide:

- [Dataset Creation Workflow](../docs/DATASET_CREATION_WORKFLOW.md)

Planned lifecycle:

- `raw/`: immutable source observations
- `manifests/`: explicit planned experiment runs
- `normalized/`: validated and provider-normalized observations
- `features/`: request-time feature matrices and targets
- `splits/`: train/validation/test/calibration assignments
- `releases/`: frozen versioned datasets
- `quarantine/`: anomalous or invalid observations retained for inspection

Raw and derived data directories are ignored by Git by default. Dataset schemas,
experiment configs, small synthetic fixtures, release manifests, and summary
statistics should be version-controlled separately.

The first milestone is to take ~50 prompts, execute ~400 controlled provider calls,
and reproducibly produce versioned train, validation, test, and calibration Parquet
files without manually editing any observation.
