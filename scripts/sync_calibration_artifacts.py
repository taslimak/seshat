"""Import every not-yet-consolidated calibration artifact into a store.

Run from a checkout of ``taslimak/seshat`` (configs are read relative to it) by
the store repository's own workflow (``store_template/``). Uses the ``gh`` CLI
with the job's built-in ``GITHUB_TOKEN`` to read the *public* source repository,
so no personal access token or cross-repository secret is needed.

* Lists successful runs of the calibration workflow (or the given run ids).
* Skips artifacts whose ``artifact_id`` already appears in the store ledger,
  and expired artifacts (reported as warnings).
* Downloads the original ZIP, verifies its SHA-256 against the API digest and
  calls :func:`seshat.data.consolidate.consolidate` with full provenance.

No provider API calls are made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from seshat.data.consolidate import DataStore, consolidate, refresh_reports

ARTIFACT_PREFIX = "seshat-calibration-"
WORKFLOW_FILE = "manual-calibration-batch.yml"


def _gh_json(path: str) -> Any:
    out = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github+json", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


# Paid requests may have been made before a run was cancelled, failed or timed out;
# their artifacts hold real observations and must be imported too.
IMPORTABLE_CONCLUSIONS = frozenset({"success", "cancelled", "failure", "timed_out"})
EMPTY_LEDGER = Path("_provenance") / "empty_artifacts.jsonl"


def completed_runs(repo: str, limit: int) -> list[int]:
    data = _gh_json(
        f"repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs?status=completed&per_page={limit}"
    )
    return [
        int(run["id"])
        for run in data.get("workflow_runs", [])
        if run.get("conclusion") in IMPORTABLE_CONCLUSIONS
    ]


def _record_empty(store: Path, record: dict[str, Any]) -> None:
    path = store / EMPTY_LEDGER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_artifacts(repo: str, run_id: int) -> list[dict[str, Any]]:
    data = _gh_json(f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100")
    return [a for a in data.get("artifacts", []) if str(a["name"]).startswith(ARTIFACT_PREFIX)]


def imported_artifact_ids(store: Path) -> set[str]:
    """Artifacts already imported, or known to contain no experiments."""
    done: set[str] = set()
    empty = store / EMPTY_LEDGER
    if empty.exists():
        done |= {
            str(json.loads(line)["artifact_id"])
            for line in empty.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if (store / "STORE.json").exists():
        done |= {
            str(entry.get("metadata", {}).get("artifact_id"))
            for entry in DataStore(store).ledger()
            if entry.get("metadata", {}).get("artifact_id")
        }
    return done


def _has_experiments(archive: Path) -> bool:
    """True if the artifact holds at least one raw/<experiment>/plan.jsonl."""
    with zipfile.ZipFile(archive) as bundle:
        return any(name.endswith("/plan.jsonl") and "raw/" in name for name in bundle.namelist())


def download(repo: str, artifact_id: int, target: Path) -> str:
    with target.open("wb") as handle:
        subprocess.run(
            ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact_id}/zip"],
            check=True,
            stdout=handle,
        )
    return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()


def sync(store: Path, repo: str, run_ids: list[int] | None, *, limit: int = 50) -> dict[str, Any]:
    runs = run_ids or completed_runs(repo, limit)
    done = imported_artifact_ids(store)
    summary: dict[str, Any] = {
        "imported": [],
        "skipped": [],
        "expired": [],
        "empty": [],
        "runs": runs,
    }
    with tempfile.TemporaryDirectory(prefix="seshat-sync-") as tmp:
        for run_id in sorted(runs):
            for artifact in run_artifacts(repo, run_id):
                artifact_id = str(artifact["id"])
                if artifact_id in done:
                    summary["skipped"].append(artifact_id)
                    continue
                if artifact.get("expired"):
                    summary["expired"].append(
                        {"run_id": run_id, "artifact_id": artifact_id, "name": artifact["name"]}
                    )
                    continue
                zip_path = Path(tmp) / f"{artifact_id}.zip"
                actual = download(repo, int(artifact_id), zip_path)
                expected = artifact.get("digest")
                if expected and expected != actual:
                    raise ValueError(f"Digest mismatch for artifact {artifact_id}")
                if not _has_experiments(zip_path):
                    record = {"run_id": run_id, "artifact_id": artifact_id, "digest": actual}
                    _record_empty(store, record)
                    summary["empty"].append(record)
                    done.add(artifact_id)
                    continue
                entry = consolidate(
                    zip_path,
                    store,
                    store_kind="real",
                    metadata={
                        "github_repository": repo,
                        "github_run_id": str(run_id),
                        "artifact_id": artifact_id,
                        "artifact_name": str(artifact["name"]),
                        "artifact_digest": actual,
                        "artifact_created_at": str(artifact.get("created_at")),
                        "imported_by": os.environ.get("GITHUB_SERVER_URL", "")
                        + "/"
                        + os.environ.get("GITHUB_REPOSITORY", "")
                        + "/actions/runs/"
                        + os.environ.get("GITHUB_RUN_ID", "local"),
                    },
                    write_reports=False,
                )
                done.add(artifact_id)
                summary["imported"].append(
                    {
                        "run_id": run_id,
                        "artifact_id": artifact_id,
                        "status": entry["status"],
                        "experiments": [
                            {
                                k: e[k]
                                for k in (
                                    "experiment_id",
                                    "events_new",
                                    "events_duplicate",
                                    "normalized_rows",
                                    "artifact_rows_match",
                                )
                            }
                            for e in entry.get("experiments", [])
                        ],
                    }
                )
    if (store / "STORE.json").exists():
        summary["reports"] = refresh_reports(store)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--repo", default="taslimak/seshat")
    parser.add_argument("--run-ids", nargs="*", type=int, help="Default: recent successful runs")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    summary = sync(args.store.resolve(), args.repo, args.run_ids or None, limit=args.limit)
    print(json.dumps(summary, indent=2, default=str))
    if summary["expired"]:
        print(
            f"warning: {len(summary['expired'])} artifact(s) expired before import", file=sys.stderr
        )


if __name__ == "__main__":
    main()
