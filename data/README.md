# Data

Dataset contents are intentionally not committed yet.

Planned lifecycle:

- `raw/`: immutable source observations
- `interim/`: validated and normalized events
- `processed/`: train/validation/test-ready datasets

Raw and derived data directories are ignored by Git by default. Dataset schemas and
small synthetic fixtures should be version-controlled separately.
