import json
from pathlib import Path

from seshat.cli import main
from seshat.data.ingest import ingest_jsonl
from seshat.data.models import PromptRecord
from seshat.data.records import ObservedUsageRecord, RecordType, RequestFeatureRecord
from seshat.data.store import DatasetStore
from seshat.data.synthetic import generate_synthetic_records

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "synthetic_v1"
EDGE_CASES = REPO_ROOT / "examples" / "edge_cases" / "observed_usage_edge_cases.jsonl"


def test_generation_is_deterministic_and_labelled_synthetic(prompts: list[PromptRecord]) -> None:
    first = generate_synthetic_records(prompts, n=10, seed=3)
    second = generate_synthetic_records(prompts, n=10, seed=3)
    assert [r.model_dump() for r in first[0]] == [r.model_dump() for r in second[0]]
    assert [o.model_dump() for o in first[1]] == [o.model_dump() for o in second[1]]
    for record in (*first[0], *first[1]):
        assert record.provenance.data_origin == "synthetic"
        assert record.provenance.source == "simulation"
    assert all(request.prompt_text is None for request in first[0])


def test_committed_example_dataset_is_reproducible(tmp_path: Path) -> None:
    argv = ["synth", "--prompts", str(REPO_ROOT / "prompts" / "hand_authored.jsonl")]
    assert main([*argv, "--n", "48", "--seed", "1729", "--output-dir", str(tmp_path)]) == 0
    for name in ("request_features.jsonl", "observed_usage.jsonl"):
        assert (tmp_path / name).read_bytes() == (EXAMPLE_DIR / name).read_bytes(), name


def test_committed_examples_validate_and_ingest_cleanly(tmp_path: Path) -> None:
    for line in (EXAMPLE_DIR / "request_features.jsonl").read_text().splitlines():
        RequestFeatureRecord.model_validate_json(line)
    for line in (EXAMPLE_DIR / "observed_usage.jsonl").read_text().splitlines():
        ObservedUsageRecord.model_validate_json(line)
    store = DatasetStore(tmp_path / "store")
    requests = ingest_jsonl(
        store, RecordType.REQUEST_FEATURES, EXAMPLE_DIR / "request_features.jsonl"
    )
    usage = ingest_jsonl(store, RecordType.OBSERVED_USAGE, EXAMPLE_DIR / "observed_usage.jsonl")
    assert (requests.stored, requests.quarantined) == (48, 0)
    assert (usage.stored, usage.quarantined) == (48, 0)


def test_edge_case_examples_exercise_every_guard(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "store")
    report = ingest_jsonl(store, RecordType.OBSERVED_USAGE, EDGE_CASES)
    expected = json.loads((EDGE_CASES.parent / "expected_report.json").read_text())
    assert report.to_dict()["quarantine_reasons"] == expected["quarantine_reasons"]
    assert report.stored == expected["stored"]
    assert report.duplicates == expected["duplicates"]
