"""Consolidate collection artifacts into a durable, canonical data store.

GitHub Actions artifacts expire (30 days by default) and every calibration run
uploads an independent ``data/`` tree. This module merges such trees -- a ZIP as
downloaded from the Actions UI/API or an already-extracted directory -- into one
long-lived *store* that uses the normal :class:`~seshat.data.storage.StorageLayout`
(so ``validate``, ``prepare-training`` etc. run on it unchanged).

Guarantees
----------
* **Raw-first.** Only the raw layer (``raw/<experiment>/plan.jsonl`` +
  ``events.jsonl`` + run logs) is merged. The normalized Parquet tables are then
  rebuilt from the merged raw layer, so they are a deterministic function of it.
  The artifact's own normalized tables are used only as a consistency check.
* **Provenance.** Every import appends one entry to
  ``_provenance/imports.jsonl`` (source digest, artifact/run identifiers, per-file
  hashes, per-experiment event counts, code revision, pricing version) and, by
  default, keeps the original archive under ``_provenance/archives/``.
* **Idempotent.** An import is identified by the SHA-256 of the archive (or of
  the sorted file list + file hashes for a directory). Re-importing the same
  source is a no-op. Overlapping sources merge by stable identifiers:
  plan rows by ``sample_id`` (conflicting definitions abort the import) and
  events by the SHA-256 of their canonical JSON (exact replays are dropped).
* **No fixture/real mixing.** A store is created with ``kind`` ``real`` or
  ``fixture`` (``STORE.json``). Experiments are classified from their planned
  providers: mock providers are fixtures, everything else is real. A real store
  refuses fixture experiments (and vice versa) and refuses mixed experiments.
* **Prompt text.** ``raw/prompt_content/`` is never imported. ``plan.jsonl`` does
  contain the exact (project-authored) request messages because normalization
  needs them; the store must therefore live in private storage.
* **All-or-nothing.** All checks run before anything is written; files are
  written via temporary files and atomic renames, under an exclusive store lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from seshat.data.io import write_json

StoreKind = Literal["real", "fixture"]
STORE_MARKER = "STORE.json"
STORE_FORMAT_VERSION = 1
LEDGER = Path("_provenance") / "imports.jsonl"
ARCHIVES = Path("_provenance") / "archives"
SOURCE_CONFIGS = Path("_provenance") / "source_files"
REPORTS = Path("reports")
# raw/ sub-directories that are not experiments.
NON_EXPERIMENT_DIRS = frozenset({"imports", "generated", "prompt_content"})
# Artifact files kept for provenance (copied, never interpreted).
PROVENANCE_GLOBS = ("*.yml", "*.yaml", "*_stats.json")
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024


class ConsolidationError(ValueError):
    """The source cannot be merged into the store without violating an invariant."""


@dataclass
class ExperimentMerge:
    experiment_id: str
    kind: StoreKind
    providers: list[str]
    plan_rows_source: int
    plan_rows_new: int
    events_source: int
    events_new: int
    events_duplicate: int
    files_copied: list[str] = field(default_factory=list)
    files_conflicting: list[str] = field(default_factory=list)
    merged_plan: list[str] = field(default_factory=list, repr=False)
    merged_events: list[str] = field(default_factory=list, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "kind": self.kind,
            "providers": self.providers,
            "plan_rows_source": self.plan_rows_source,
            "plan_rows_new": self.plan_rows_new,
            "events_source": self.events_source,
            "events_new": self.events_new,
            "events_duplicate": self.events_duplicate,
            "files_copied": self.files_copied,
            "files_conflicting_kept_existing": self.files_conflicting,
        }


# --------------------------------------------------------------------------- helpers
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(line: str) -> str:
    return json.dumps(json.loads(line), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _line_key(line: str) -> str:
    return hashlib.sha256(_canonical(line).encode("utf-8")).hexdigest()


def _nonblank_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(target)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP refusing absolute paths, ``..`` traversal, links and zip bombs."""
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ConsolidationError(f"{archive} exceeds {MAX_ARCHIVE_BYTES} bytes")
    with zipfile.ZipFile(archive) as bundle:
        total = 0
        for info in bundle.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
                raise ConsolidationError(f"Unsafe path in archive: {info.filename!r}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ConsolidationError(f"Symlink in archive: {info.filename!r}")
            if info.file_size > MAX_MEMBER_BYTES:
                raise ConsolidationError(f"Archive member too large: {info.filename!r}")
            total += info.file_size
            if total > MAX_ARCHIVE_BYTES * 4:
                raise ConsolidationError("Archive expands beyond the allowed size")
        bundle.extractall(destination)


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def find_data_root(extracted: Path) -> Path:
    """The directory holding ``raw/`` (the artifact root, or one ``data/`` level down)."""
    candidates = [extracted, extracted / "data"]
    candidates += [p for p in sorted(extracted.iterdir()) if p.is_dir()]
    for candidate in candidates:
        if (candidate / "raw").is_dir():
            return candidate
    raise ConsolidationError(f"No raw/ directory found in {extracted}")


def _classify(providers: set[str]) -> StoreKind:
    from seshat.providers.registry import adapter_class

    live = {p for p in providers if adapter_class(p).makes_external_calls}
    if live and live != providers:
        raise ConsolidationError(
            f"Experiment mixes real and mock providers ({sorted(providers)}); refusing to import"
        )
    return "real" if live else "fixture"


# --------------------------------------------------------------------------- store
class DataStore:
    """A durable store root: a StorageLayout plus marker, ledger and lock."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def marker(self) -> Path:
        return self.root / STORE_MARKER

    def kind(self) -> StoreKind | None:
        if not self.marker.exists():
            return None
        data = json.loads(self.marker.read_text(encoding="utf-8"))
        kind = data.get("kind")
        if kind not in ("real", "fixture"):
            raise ConsolidationError(f"Corrupt store marker {self.marker}")
        return kind  # type: ignore[no-any-return]

    def check(self, kind: StoreKind) -> None:
        """Fail early (without writing) if the store cannot accept ``kind`` data."""
        current = self.kind()
        if current is None and (self.root / "raw").exists():
            raise ConsolidationError(
                f"{self.root} already holds data but no {STORE_MARKER}; refusing to adopt it"
            )
        if current is not None and current != kind:
            raise ConsolidationError(f"Store {self.root} is a {current!r} store, not {kind!r}")

    def ensure(self, kind: StoreKind) -> None:
        self.check(kind)
        current = self.kind()
        if current is None:
            if self.root.exists() and (self.root / "raw").exists():
                raise ConsolidationError(
                    f"{self.root} already holds data but no {STORE_MARKER}; refusing to adopt it"
                )
            self.root.mkdir(parents=True, exist_ok=True)
            gitignore = self.root / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(".consolidate.lock\n", encoding="utf-8")
            write_json(
                self.marker,
                {
                    "kind": kind,
                    "format_version": STORE_FORMAT_VERSION,
                    "created_at": datetime.now(UTC).isoformat(),
                    "note": (
                        "Canonical Seshat observation store. Real stores contain exact "
                        "request messages in raw/*/plan.jsonl: keep them in private storage."
                    ),
                },
            )
        elif current != kind:
            raise ConsolidationError(f"Store {self.root} is a {current!r} store, not {kind!r}")

    def ledger(self) -> list[dict[str, Any]]:
        path = self.root / LEDGER
        return [json.loads(line) for line in _nonblank_lines(path)]

    def imported_ids(self) -> set[str]:
        return {str(entry["import_id"]) for entry in self.ledger()}

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".consolidate.lock").open("w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ConsolidationError(f"Store {self.root} is locked by another import") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- merging
def _plan_index(lines: list[str], where: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        try:
            sample_id = str(json.loads(line)["sample_id"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise ConsolidationError(f"Invalid plan row {where}:{number}") from exc
        if sample_id in index and _canonical(index[sample_id]) != _canonical(line):
            raise ConsolidationError(f"Conflicting duplicate sample {sample_id} in {where}")
        index.setdefault(sample_id, line)
    return index


def _event_time(line: str) -> str:
    value = json.loads(line).get("at")
    return str(value) if value is not None else ""


def plan_experiment_merge(source_dir: Path, store_dir: Path, experiment_id: str) -> ExperimentMerge:
    """Compute (without writing) the merged plan/events for one experiment."""
    plan_path, events_path = source_dir / "plan.jsonl", source_dir / "events.jsonl"
    if not plan_path.exists() or not events_path.exists():
        raise ConsolidationError(f"{source_dir} lacks plan.jsonl or events.jsonl")
    src_plan = _nonblank_lines(plan_path)
    src_events = _nonblank_lines(events_path)
    providers = {str(json.loads(line)["provider"]) for line in src_plan}
    experiments = {str(json.loads(line)["experiment_id"]) for line in src_plan}
    if experiments - {experiment_id}:
        raise ConsolidationError(
            f"{plan_path} contains samples of other experiments: {sorted(experiments)}"
        )
    kind = _classify(providers)

    store_plan = _nonblank_lines(store_dir / "plan.jsonl")
    existing = _plan_index(store_plan, str(store_dir / "plan.jsonl"))
    incoming = _plan_index(src_plan, str(plan_path))
    merged_plan = list(store_plan)
    new_rows = 0
    for sample_id, line in incoming.items():
        if sample_id in existing:
            if _canonical(existing[sample_id]) != _canonical(line):
                raise ConsolidationError(
                    f"Sample {sample_id} of {experiment_id} is defined differently in the "
                    "store and the source (design changed under the same experiment_id)"
                )
            continue
        merged_plan.append(line)
        new_rows += 1

    store_events = _nonblank_lines(store_dir / "events.jsonl")
    seen = {_line_key(line) for line in store_events}
    additions: list[str] = []
    duplicates = 0
    for line in src_events:
        try:
            key = _line_key(line)
        except json.JSONDecodeError as exc:
            raise ConsolidationError(f"Corrupt event line in {events_path}") from exc
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        additions.append(line)
    if store_events and additions:
        # Interleave by timestamp (stable: ties keep store-then-source order).
        ordered = sorted(
            enumerate(store_events + additions), key=lambda item: (_event_time(item[1]), item[0])
        )
        merged_events = [line for _, line in ordered]
    else:
        merged_events = store_events + additions
    return ExperimentMerge(
        experiment_id=experiment_id,
        kind=kind,
        providers=sorted(providers),
        plan_rows_source=len(incoming),
        plan_rows_new=new_rows,
        events_source=len(src_events),
        events_new=len(additions),
        events_duplicate=duplicates,
        merged_plan=merged_plan,
        merged_events=merged_events,
    )


def _source_counts(data_root: Path, experiment_id: str) -> dict[str, int] | None:
    """Row counts in the artifact's own normalized tables (consistency check only)."""
    import pyarrow.parquet as pq

    from seshat.data.storage.layers import _safe

    name = f"part-experiment-{_safe(experiment_id)}.parquet"
    counts: dict[str, int] = {}
    found = False
    for table in ("requests", "preflight", "observations"):
        parts = sorted((data_root / "normalized" / table).glob(f"date=*/provider=*/{name}"))
        found = found or bool(parts)
        counts[table] = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    return counts if found else None


# --------------------------------------------------------------------------- main API
def consolidate(
    source: str | Path,
    store_root: str | Path,
    *,
    store_kind: StoreKind,
    metadata: dict[str, str] | None = None,
    keep_archive: bool = True,
    pricing_dir: str | Path | None = None,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Merge one artifact (ZIP or directory) into ``store_root``. Returns the ledger entry."""
    from seshat.data.collectors.normalize import normalize_experiment
    from seshat.data.pricing import PriceTable
    from seshat.data.privacy import RetentionPolicy
    from seshat.data.training import _git_revision
    from seshat.providers.catalog import load_catalog

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    store = DataStore(store_root)
    is_zip = source_path.is_file()
    if is_zip and not zipfile.is_zipfile(source_path):
        raise ConsolidationError(f"{source_path} is neither a directory nor a ZIP archive")
    import_id = _sha256_file(source_path) if is_zip else _directory_digest(source_path)

    with store.lock(), tempfile.TemporaryDirectory(prefix="seshat-consolidate-") as tmp:
        store.check(store_kind)
        if import_id in store.imported_ids():
            return {"status": "already_imported", "import_id": import_id, "store": str(store.root)}
        if is_zip:
            extracted = Path(tmp) / "source"
            extracted.mkdir()
            _safe_extract(source_path, extracted)
        else:
            extracted = source_path
        data_root = find_data_root(extracted)
        experiment_dirs = sorted(
            p
            for p in (data_root / "raw").iterdir()
            if p.is_dir() and p.name not in NON_EXPERIMENT_DIRS
        )
        if not experiment_dirs:
            raise ConsolidationError(f"No experiments under {data_root / 'raw'}")

        # Phase 1: plan every merge and run every check before writing anything.
        merges: list[tuple[Path, ExperimentMerge]] = []
        for exp_dir in experiment_dirs:
            target = store.root / "raw" / exp_dir.name
            merge = plan_experiment_merge(exp_dir, target, exp_dir.name)
            if merge.kind != store_kind:
                raise ConsolidationError(
                    f"Experiment {exp_dir.name} is {merge.kind!r} data (providers "
                    f"{merge.providers}); this is a {store_kind!r} store"
                )
            for extra in sorted(p for p in exp_dir.rglob("*") if p.is_file()):
                rel = extra.relative_to(exp_dir)
                if rel.as_posix() in {"plan.jsonl", "events.jsonl", "normalization.json"}:
                    continue
                if rel.parts[0] == "dry_run":
                    continue
                destination = target / rel
                if not destination.exists():
                    merge.files_copied.append(rel.as_posix())
                elif _sha256_file(destination) != _sha256_file(extra):
                    merge.files_conflicting.append(rel.as_posix())
            merges.append((exp_dir, merge))
        skipped_prompt_content = (data_root / "raw" / "prompt_content").exists()
        file_hashes = {
            p.relative_to(data_root).as_posix(): _sha256_file(p)
            for p in sorted(data_root.rglob("*"))
            if p.is_file() and "prompt_content" not in p.relative_to(data_root).parts
        }

        # Phase 2: write raw layer, provenance copies, then rebuild normalized tables.
        store.ensure(store_kind)
        for exp_dir, merge in merges:
            target = store.root / "raw" / merge.experiment_id
            _atomic_write_text(target / "plan.jsonl", "\n".join(merge.merged_plan) + "\n")
            _atomic_write_text(target / "events.jsonl", "\n".join(merge.merged_events) + "\n")
            for rel_name in merge.files_copied:
                _atomic_copy(exp_dir / rel_name, target / rel_name)
        provenance_dir = store.root / SOURCE_CONFIGS / import_id[:16]
        for pattern in PROVENANCE_GLOBS:
            for extra in sorted(data_root.glob(pattern)):
                _atomic_copy(extra, provenance_dir / extra.name)
        if keep_archive and is_zip:
            _atomic_copy(source_path, store.root / ARCHIVES / f"{import_id}.zip")

        catalog = load_catalog()
        prices = PriceTable.load(pricing_dir)
        policy = RetentionPolicy(store_prompt_text=False)
        experiments: list[dict[str, Any]] = []
        for _, merge in merges:
            counts = normalize_experiment(
                store.root, merge.experiment_id, catalog=catalog, prices=prices, policy=policy
            )
            source_counts = _source_counts(data_root, merge.experiment_id)
            summary = merge.summary()
            summary["normalized_rows"] = counts
            summary["artifact_normalized_rows"] = source_counts
            summary["artifact_rows_match"] = (
                None
                if source_counts is None
                else all(counts[t] >= source_counts[t] for t in source_counts)
            )
            experiments.append(summary)

        entry: dict[str, Any] = {
            "import_id": import_id,
            "status": "imported",
            "imported_at": datetime.now(UTC).isoformat(),
            "source_name": source_path.name,
            "source_kind": "zip" if is_zip else "directory",
            "metadata": dict(sorted((metadata or {}).items())),
            "store_kind": store_kind,
            "experiments": experiments,
            "skipped_prompt_content": skipped_prompt_content,
            "source_files": file_hashes,
            "archive": (ARCHIVES / f"{import_id}.zip").as_posix()
            if keep_archive and is_zip
            else None,
            "code_revision": _git_revision(),
            "pricing_versions": sorted({e.pricing_version for e in prices.entries}),
        }
        ledger_path = store.root / LEDGER
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    if write_reports:
        entry["reports"] = refresh_reports(store.root)
    return entry


def refresh_reports(store_root: str | Path) -> dict[str, str]:
    """Re-validate the store and regenerate human-readable reports under ``reports/``."""
    from seshat.data.readiness import readiness_report, render_readiness_markdown
    from seshat.data.stats import collection_stats
    from seshat.data.validators import validate_storage

    root = Path(store_root)
    reports = root / REPORTS
    reports.mkdir(parents=True, exist_ok=True)
    validate_storage(root)
    csv_path = write_consolidated_csv(root, reports / "observations_consolidated.csv")
    write_json(reports / "collection_stats.json", collection_stats(root))
    write_json(reports / "spend_by_month.json", spend_by_month(root))
    readiness = readiness_report(root)
    write_json(reports / "training_readiness.json", readiness)
    (reports / "training_readiness.md").write_text(
        render_readiness_markdown(readiness), encoding="utf-8"
    )
    quality_md = root / "normalized" / "_validation" / "quality_report.md"
    if quality_md.exists():
        shutil.copyfile(quality_md, reports / "quality_report.md")
    return {
        "observations_csv": str(csv_path),
        "collection_stats": str(reports / "collection_stats.json"),
        "spend_by_month": str(reports / "spend_by_month.json"),
        "training_readiness": str(reports / "training_readiness.md"),
        "quality_report": str(reports / "quality_report.md"),
    }


def spend_by_month(store_root: str | Path) -> dict[str, Any]:
    """Estimated (price-table) spend of real observations per UTC month.

    These are internal estimates from the pricing table, not provider invoices.
    """
    from seshat.data.storage import StorageLayout

    observations = StorageLayout(store_root).read_frame("observations")
    months: dict[str, dict[str, Any]] = {}
    if not observations.empty:
        real = observations[observations["source_type"] != "test_fixture"]
        for (month, model), group in real.groupby(
            [real["collected_at_utc"].dt.strftime("%Y-%m"), "requested_model"]
        ):
            total = sum(
                (Decimal(str(v)) for v in group["calculated_cost_usd"].dropna()), Decimal(0)
            )
            entry = months.setdefault(str(month), {"estimated_usd": "0", "by_model": {}})
            entry["by_model"][str(model)] = {
                "requests": int(len(group)),
                "estimated_usd": str(total),
            }
            entry["estimated_usd"] = str(Decimal(entry["estimated_usd"]) + total)
    return {
        "basis": "pricing-table estimates of real observations (not provider invoices)",
        "months": months,
    }


def month_to_date_spend(store_root: str | Path, month: str) -> Decimal:
    entry = spend_by_month(store_root)["months"].get(month)
    return Decimal(entry["estimated_usd"]) if entry else Decimal(0)


CSV_COLUMNS = [
    "observation_id",
    "sample_id",
    "experiment_id",
    "collected_at_utc",
    "provider",
    "requested_model",
    "resolved_model",
    "requested_effort",
    "max_output_tokens",
    "prompt_category",
    "prompt_subcategory",
    "prompt_family_id",
    "prompt_template_id",
    "length_stratum",
    "repetition_index",
    "prompt_word_count",
    "message_count",
    "tool_count",
    "requested_output_format",
    "explicit_output_length_hint",
    "preflight_input_tokens",
    "actual_input_tokens",
    "actual_output_tokens",
    "actual_reasoning_tokens",
    "actual_visible_output_tokens",
    "actual_total_tokens",
    "reasoning_included_in_output",
    "response_status",
    "stop_reason",
    "is_truncated",
    "label_quality",
    "decision",
    "eligible_for_training",
    "data_origin",
    "calculated_cost_usd",
]


def write_consolidated_csv(store_root: str | Path, output: str | Path) -> Path:
    """One row per observation with request-time context, labels and validation decision.

    Contains identifiers, hashes-free metadata and counts only -- never prompt text.
    """
    import pandas as pd

    from seshat.data.io import read_parquet
    from seshat.data.storage import StorageLayout

    layout = StorageLayout(store_root)
    observations = layout.read_frame("observations")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if observations.empty:
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(target, index=False)
        return target
    requests = layout.read_frame("requests").drop_duplicates("sample_id")
    request_cols = [
        "sample_id",
        "experiment_id",
        "requested_effort",
        "max_output_tokens",
        "prompt_category",
        "prompt_subcategory",
        "prompt_family_id",
        "prompt_template_id",
        "design_cell_id",
        "repetition_index",
        "prompt_word_count",
        "message_count",
        "tool_count",
        "requested_output_format",
        "explicit_output_length_hint",
    ]
    frame = observations.merge(requests[request_cols], on="sample_id", how="left")
    frame["length_stratum"] = frame["design_cell_id"].map(
        lambda v: str(v).split("|", 1)[0] if isinstance(v, str) else None
    )
    preflight = layout.read_frame("preflight")
    if not preflight.empty:
        pre = preflight[preflight["preflight_input_tokens"].notna()].copy()
        pre["_rank"] = (pre["preflight_method"] != "provider_native").astype(int)
        pre = pre.sort_values(["sample_id", "_rank"]).drop_duplicates("sample_id")
        frame = frame.merge(
            pre[["sample_id", "preflight_input_tokens"]], on="sample_id", how="left"
        )
    decisions_path = layout.validation / "decisions.parquet"
    if decisions_path.exists():
        decisions = read_parquet(decisions_path)[
            ["observation_id", "decision", "eligible_for_training", "data_origin"]
        ]
        frame = frame.merge(decisions, on="observation_id", how="left")
    for column in CSV_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[CSV_COLUMNS].sort_values(["collected_at_utc", "observation_id"])
    frame.to_csv(target, index=False)
    return target
