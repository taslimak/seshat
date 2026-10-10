# Example datasets

Everything here is **synthetic** and labelled as such in each record's `provenance`.
None of it is real provider usage, and the feature builder will not use it as training
labels unless `--label-origin synthetic` is passed explicitly.

| Path | What it is | How it was made |
|---|---|---|
| `synthetic_v1/request_features.jsonl` | 48 `RequestFeatureRecord`s (prompt hash + features, no prompt text) | `seshat-data synth --prompts prompts/hand_authored.jsonl --n 48 --seed 1729 --output-dir examples/synthetic_v1` |
| `synthetic_v1/observed_usage.jsonl` | 48 matching `ObservedUsageRecord`s produced by the mock adapter's usage model | same command |
| `edge_cases/observed_usage_edge_cases.jsonl` | 12 hand-written lines, one per ingestion guard (each record's `metadata.case` says what it tests) | hand-written |
| `edge_cases/expected_report.json` | The ingestion report those lines must produce | hand-written, checked by `tests/test_synthetic_examples.py` |

The synthetic files are regenerated and compared byte-for-byte in CI, so they cannot
drift from the generator.

Try it:

```bash
seshat-data ingest --store /tmp/seshat-store --record-type request_features --input examples/synthetic_v1/request_features.jsonl
seshat-data ingest --store /tmp/seshat-store --record-type observed_usage --input examples/synthetic_v1/observed_usage.jsonl
seshat-data predict --store /tmp/seshat-store
seshat-data compare --store /tmp/seshat-store
seshat-data ingest --store /tmp/seshat-store --record-type observed_usage --input examples/edge_cases/observed_usage_edge_cases.jsonl
```
