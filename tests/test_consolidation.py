"""Durable consolidation of collection artifacts (offline, mock providers only)."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from seshat.data.consolidate import (
    LEDGER,
    ConsolidationError,
    DataStore,
    consolidate,
    find_data_root,
)
from seshat.data.storage import StorageLayout
from tests.real_shape import mock_collection


def _zip_dir(directory: Path, target: Path, *, nested: str | None = None) -> Path:
    with zipfile.ZipFile(target, "w") as bundle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                arc = path.relative_to(directory).as_posix()
                bundle.write(path, f"{nested}/{arc}" if nested else arc)
    return target


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    data = mock_collection(tmp_path / "run1", count=40)
    return _zip_dir(data, tmp_path / "artifact.zip")


def test_import_is_idempotent_and_preserves_provenance(tmp_path: Path, artifact: Path) -> None:
    store = tmp_path / "store"
    first = consolidate(
        artifact, store, store_kind="fixture", metadata={"github_run_id": "1"}, write_reports=False
    )
    assert first["status"] == "imported"
    exp = first["experiments"][0]
    assert exp["events_new"] == exp["events_source"] > 0
    assert exp["artifact_rows_match"] is True
    assert (store / first["archive"]).exists()
    assert first["metadata"] == {"github_run_id": "1"}
    observations = StorageLayout(store).read_frame("observations")
    assert len(observations) == 40

    again = consolidate(artifact, store, store_kind="fixture", write_reports=False)
    assert again["status"] == "already_imported"
    assert len(DataStore(store).ledger()) == 1
    assert len(StorageLayout(store).read_frame("observations")) == 40


def test_overlapping_sources_deduplicate_by_stable_ids(tmp_path: Path, artifact: Path) -> None:
    store = tmp_path / "store"
    consolidate(artifact, store, store_kind="fixture", write_reports=False)
    # Same records, different archive layout (nested data/ folder) => new import id.
    extracted = tmp_path / "x"
    with zipfile.ZipFile(artifact) as bundle:
        bundle.extractall(extracted)
    nested = _zip_dir(extracted, tmp_path / "nested.zip", nested="data")
    result = consolidate(nested, store, store_kind="fixture", write_reports=False)
    exp = result["experiments"][0]
    assert result["status"] == "imported"
    assert exp["events_new"] == 0 and exp["plan_rows_new"] == 0
    assert exp["events_duplicate"] == exp["events_source"]
    assert len(StorageLayout(store).read_frame("observations")) == 40


def test_two_experiments_merge(tmp_path: Path, artifact: Path) -> None:
    store = tmp_path / "store"
    consolidate(artifact, store, store_kind="fixture", write_reports=False)
    other = mock_collection(tmp_path / "run2", experiment_id="exp_b", count=20, seed=9)
    consolidate(other, store, store_kind="fixture", write_reports=False)
    observations = StorageLayout(store).read_frame("observations")
    assert len(observations) == 60
    assert observations["observation_id"].is_unique


def test_real_store_refuses_fixture_data_without_writing(tmp_path: Path, artifact: Path) -> None:
    store = tmp_path / "real"
    with pytest.raises(ConsolidationError, match="'fixture' data"):
        consolidate(artifact, store, store_kind="real", write_reports=False)
    assert not (store / "STORE.json").exists()
    assert not (store / "raw").exists()


def test_store_kind_is_sticky(tmp_path: Path, artifact: Path) -> None:
    store = tmp_path / "store"
    consolidate(artifact, store, store_kind="fixture", write_reports=False)
    with pytest.raises(ConsolidationError, match="is a 'fixture' store"):
        consolidate(artifact, store, store_kind="real", write_reports=False)


def test_conflicting_sample_definition_aborts(tmp_path: Path, artifact: Path) -> None:
    store = tmp_path / "store"
    consolidate(artifact, store, store_kind="fixture", write_reports=False)
    extracted = tmp_path / "tampered"
    with zipfile.ZipFile(artifact) as bundle:
        bundle.extractall(extracted)
    plan = next(extracted.glob("raw/*/plan.jsonl"))
    rows = [json.loads(line) for line in plan.read_text().splitlines()]
    rows[0]["prompt_category"] = "changed"
    plan.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    before = (store / LEDGER).read_text()
    with pytest.raises(ConsolidationError, match="defined differently"):
        consolidate(extracted, store, store_kind="fixture", write_reports=False)
    assert (store / LEDGER).read_text() == before


def test_unsafe_archive_paths_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as bundle:
        bundle.writestr("../escape.txt", "x")
    with pytest.raises(ConsolidationError, match="Unsafe path"):
        consolidate(bad, tmp_path / "store", store_kind="fixture", write_reports=False)


def test_prompt_content_is_never_imported(tmp_path: Path) -> None:
    data = mock_collection(tmp_path / "run", count=10)
    secret = data / "raw" / "prompt_content" / "abc.json"
    secret.parent.mkdir(parents=True)
    secret.write_text('{"text": "secret"}')
    store = tmp_path / "store"
    result = consolidate(data, store, store_kind="fixture", write_reports=False)
    assert result["skipped_prompt_content"] is True
    assert not list(store.rglob("prompt_content"))
    assert not any("prompt_content" in name for name in result["source_files"])


def test_reports_and_csv(tmp_path: Path, artifact: Path) -> None:
    store = tmp_path / "store"
    result = consolidate(artifact, store, store_kind="fixture")
    csv = pd.read_csv(result["reports"]["observations_csv"])
    assert len(csv) == 40
    assert {"prompt_family_id", "length_stratum", "is_truncated", "decision"} <= set(csv.columns)
    assert not any("text" in column or "content" in column for column in csv.columns)
    readiness = json.loads((store / "reports" / "training_readiness.json").read_text())
    assert readiness["status"] == "no_real_labels"  # fixtures never count as real labels


def test_find_data_root_handles_nesting(tmp_path: Path) -> None:
    (tmp_path / "data" / "raw").mkdir(parents=True)
    assert find_data_root(tmp_path) == tmp_path / "data"
    shutil.rmtree(tmp_path / "data")
    with pytest.raises(ConsolidationError):
        find_data_root(tmp_path)
