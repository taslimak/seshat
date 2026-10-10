"""Import prompts from licensed / explicitly permitted datasets (JSONL or CSV).

Imported prompts become ``source_type=public_permitted_dataset`` bank entries. They
are **unlabeled** until a controlled experiment executes them, and they start as
``unreviewed`` (live collection requires approval). Each row is:

1. merged with CLI defaults (source, license, category, ...), never overriding
   values present in the row;
2. validated against ``PromptRecord`` and the collection policy (license allowlist,
   commercial-use terms, attribution, secrets/PII, length);
3. deduplicated deterministically: exact duplicates (normalized text + tools) are
   skipped; near-duplicates join the existing prompt family (leakage control);
4. written, sorted, to the target bank file; rejections go to a report under
   ``data/raw/imports/`` (gitignored, since rejected rows may contain PII).
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from seshat.data.io import write_jsonl
from seshat.data.models import PromptRecord
from seshat.data.privacy import redact_text
from seshat.prompts.bank import all_prompts, content_hash, load_bank, write_prompt_file
from seshat.prompts.checks import check_prompt
from seshat.prompts.dedup import SimilarityIndex
from seshat.prompts.policy import CollectionPolicy

_JSON_COLUMNS = {"messages", "tools", "tags", "metadata"}
_REVIEW_FIELDS = {"review_status", "reviewed_by", "reviewed_at", "review_notes"}


@dataclass
class ImportReport:
    source: str
    target: str
    read: int = 0
    imported: int = 0
    duplicates: int = 0
    near_duplicates_joined: int = 0
    rejected: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    imported_ids: list[str] = field(default_factory=list)
    rejection_report: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "read": self.read,
            "imported": self.imported,
            "duplicates": self.duplicates,
            "near_duplicates_joined": self.near_duplicates_joined,
            "rejected": self.rejected,
            "reasons": dict(sorted(self.reasons.items())),
            "rejection_report": self.rejection_report,
        }


def _rows(path: Path) -> Iterator[tuple[int, dict[str, Any] | str]]:
    if path.suffix == ".jsonl":
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield number, f"invalid_json: {exc}"
                    continue
                yield number, value if isinstance(value, dict) else "invalid_record: not an object"
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle), start=2):
                parsed: dict[str, Any] = {}
                for key, value in row.items():
                    if key is None or value in (None, ""):
                        continue
                    if key in _JSON_COLUMNS:
                        try:
                            parsed[key] = json.loads(value)
                        except json.JSONDecodeError:
                            parsed[key] = value
                    elif key in {"is_synthetic", "commercial_use_allowed"}:
                        parsed[key] = value.strip().lower() in {"1", "true", "yes"}
                    elif key == "generation_seed":
                        parsed[key] = int(value)
                    else:
                        parsed[key] = value
                yield number, parsed
    else:
        raise ValueError("Prompt import supports .jsonl and .csv")


def import_prompts(
    input_path: str | Path,
    *,
    bank_dir: str | Path,
    target_file: str | Path,
    policy: CollectionPolicy,
    defaults: dict[str, Any],
    report_dir: str | Path,
) -> ImportReport:
    source = Path(input_path)
    target = Path(target_file)
    bank = load_bank(bank_dir)
    existing = all_prompts(bank)
    target_existing = bank.get(target, [])
    if target.exists() and target not in bank:
        target_existing = list(_load_existing(target))
        existing.extend(target_existing)
    by_hash = {content_hash(p): p for p in existing}
    by_id = {p.prompt_id: p for p in existing}
    family_of = {p.prompt_id: p.prompt_family_id for p in existing}
    index = SimilarityIndex(policy.near_duplicate_threshold)
    for prompt in existing:
        index.add(prompt.prompt_id, prompt.text)
    report = ImportReport(source=str(source), target=str(target))
    rejections: list[dict[str, Any]] = []
    new_prompts: list[PromptRecord] = []

    def reject(line: int, reason: str, payload: Any, **extra: Any) -> None:
        report.rejected += 1
        report.reasons[reason] += 1
        rejections.append({"line": line, "reason": reason, **extra, "payload": _redact(payload)})

    for line, raw in _rows(source):
        report.read += 1
        if isinstance(raw, str):
            reject(line, raw.split(":")[0], None, detail=raw)
            continue
        row = {**{k: v for k, v in defaults.items() if v is not None}, **raw}
        ignored = _REVIEW_FIELDS & row.keys()
        for key in ignored:
            row.pop(key)
        row["review_status"] = "unreviewed"
        missing = [k for k in ("category", "complexity", "source", "license_id") if not row.get(k)]
        if missing:
            reject(line, "missing_metadata", row, fields=missing)
            continue
        row.setdefault("prompt_id", "pending")
        row.setdefault("prompt_family_id", "pending")
        try:
            prompt = PromptRecord.model_validate(row)
        except ValidationError as exc:
            errors = [
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                for e in exc.errors(include_input=False, include_url=False)
            ]
            reject(line, "schema_error", row, errors=errors)
            continue
        issues = check_prompt(prompt, policy)
        if issues:
            reject(line, "policy_violation", row, issues=[f"{i.code}: {i.message}" for i in issues])
            continue
        digest = content_hash(prompt)
        if digest in by_hash:
            report.duplicates += 1
            continue
        prompt_id = raw.get("prompt_id") or f"imp_{digest[:16]}"
        if prompt_id in by_id:
            reject(line, "prompt_id_conflict", row, prompt_id=prompt_id)
            continue
        family = raw.get("prompt_family_id") or f"fam_{digest[:16]}"
        matches = index.matches(prompt.text)
        metadata = dict(prompt.metadata)
        if matches:
            other, score = matches[0]
            family = family_of[other]
            metadata["near_duplicate_of"] = other
            metadata["near_duplicate_similarity"] = round(score, 4)
            report.near_duplicates_joined += 1
        prompt = prompt.model_copy(
            update={"prompt_id": prompt_id, "prompt_family_id": family, "metadata": metadata}
        )
        by_hash[digest] = prompt
        by_id[prompt_id] = prompt
        family_of[prompt_id] = family
        index.add(prompt_id, prompt.text)
        new_prompts.append(prompt)
        report.imported_ids.append(prompt_id)
    report.imported = len(new_prompts)
    if new_prompts:
        write_prompt_file(target, [*target_existing, *new_prompts])
    if rejections:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        path = Path(report_dir) / f"prompt-import-{stamp}-{source.stem}-rejections.jsonl"
        write_jsonl(path, rejections)
        report.rejection_report = str(path)
    return report


def _load_existing(path: Path) -> list[PromptRecord]:
    from seshat.data.io import load_jsonl

    return load_jsonl(path, PromptRecord)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value, max_length=2000)
    return value
