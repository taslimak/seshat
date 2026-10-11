"""Store-side artifact sync (GitHub API mocked; consolidation is real)."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts import sync_calibration_artifacts as sync_mod
from seshat.data.consolidate import DataStore
from tests.real_shape import mock_collection


def _zip(directory: Path, target: Path) -> Path:
    with zipfile.ZipFile(target, "w") as bundle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(directory).as_posix())
    return target


@pytest.fixture
def fake_github(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    archive = _zip(mock_collection(tmp_path / "run", count=20), tmp_path / "a.zip")
    digest = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
    state: dict[str, Any] = {"downloads": 0, "expired": False}

    def fake_json(path: str) -> Any:
        if "/workflows/" in path:
            return {"workflow_runs": [{"id": 111, "conclusion": "success"}]}
        return {
            "artifacts": [
                {
                    "id": 999,
                    "name": "seshat-calibration-111",
                    "digest": digest,
                    "expired": state["expired"],
                    "created_at": "2026-10-11T00:00:00Z",
                },
                {"id": 5, "name": "other-artifact", "expired": False},
            ]
        }

    def fake_download(repo: str, artifact_id: int, target: Path) -> str:
        state["downloads"] += 1
        shutil.copyfile(archive, target)
        return digest

    monkeypatch.setattr(sync_mod, "_gh_json", fake_json)
    monkeypatch.setattr(sync_mod, "download", fake_download)
    # Mock-provider data needs a fixture store; production syncs use kind "real".
    real_consolidate = sync_mod.consolidate
    monkeypatch.setattr(
        sync_mod,
        "consolidate",
        lambda *a, **k: real_consolidate(*a, **{**k, "store_kind": "fixture"}),
    )
    return state


def test_sync_imports_once_with_provenance(tmp_path: Path, fake_github: dict[str, Any]) -> None:
    store = tmp_path / "store"
    first = sync_mod.sync(store, "taslimak/seshat", None)
    assert [i["artifact_id"] for i in first["imported"]] == ["999"]
    ledger = DataStore(store).ledger()
    assert ledger[0]["metadata"]["github_run_id"] == "111"
    assert ledger[0]["metadata"]["artifact_name"] == "seshat-calibration-111"
    second = sync_mod.sync(store, "taslimak/seshat", [111])
    assert second["imported"] == [] and second["skipped"] == ["999"]
    assert fake_github["downloads"] == 1


def test_expired_artifacts_are_reported(tmp_path: Path, fake_github: dict[str, Any]) -> None:
    fake_github["expired"] = True
    summary = sync_mod.sync(tmp_path / "store", "taslimak/seshat", [111])
    assert summary["expired"][0]["artifact_id"] == "999"
    assert not (tmp_path / "store" / "STORE.json").exists()


def test_cancelled_runs_are_listed_and_empty_artifacts_recorded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = {
        "workflow_runs": [
            {"id": 1, "conclusion": "success"},
            {"id": 2, "conclusion": "cancelled"},
            {"id": 3, "conclusion": "failure"},
            {"id": 4, "conclusion": "skipped"},
        ]
    }
    monkeypatch.setattr(sync_mod, "_gh_json", lambda path: runs)
    assert sync_mod.completed_runs("o/r", 10) == [1, 2, 3]

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as bundle:
        bundle.writestr("calibration_stats.json", "{}")
    calls = {"n": 0}

    def fake_json(path: str) -> Any:
        return {"artifacts": [{"id": 7, "name": "seshat-calibration-3", "expired": False}]}

    def fake_download(repo: str, artifact_id: int, target: Path) -> str:
        calls["n"] += 1
        shutil.copyfile(empty, target)
        return "sha256:x"

    monkeypatch.setattr(sync_mod, "_gh_json", fake_json)
    monkeypatch.setattr(sync_mod, "download", fake_download)
    store = tmp_path / "store"
    first = sync_mod.sync(store, "o/r", [3])
    assert first["empty"][0]["artifact_id"] == "7" and first["imported"] == []
    second = sync_mod.sync(store, "o/r", [3])
    assert second["skipped"] == ["7"] and calls["n"] == 1
