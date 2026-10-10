"""Prompt bank files: ``prompts/*.jsonl`` (one PromptRecord per line, sorted by ID).

The bank is source-controlled and reviewed through pull requests. Unlike raw
observations it is editable, but only through this module's atomic writer so files
stay sorted, validated and diff-friendly.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from seshat.data.io import load_jsonl, write_jsonl
from seshat.data.models import PromptRecord

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Case-, whitespace- and Unicode-normalized text used for duplicate detection."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()


def content_hash(prompt: PromptRecord) -> str:
    """Identity of a prompt's content: normalized text plus its tool names."""
    tools = ",".join(sorted(tool.name for tool in prompt.tools))
    payload = f"{normalize_text(prompt.text)}|tools:{tools}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bank_files(bank_dir: str | Path) -> list[Path]:
    """Top-level ``*.jsonl`` files of the bank (subdirectories are not part of it)."""
    return sorted(Path(bank_dir).glob("*.jsonl"))


def load_bank(bank_dir: str | Path) -> dict[Path, list[PromptRecord]]:
    """All prompts per bank file. Raises on an invalid line."""
    return {path: load_jsonl(path, PromptRecord) for path in bank_files(bank_dir)}


def all_prompts(bank: dict[Path, list[PromptRecord]]) -> list[PromptRecord]:
    return [prompt for prompts in bank.values() for prompt in prompts]


def write_prompt_file(path: str | Path, prompts: Iterable[PromptRecord]) -> int:
    """Atomically write a bank file sorted by ``prompt_id``, omitting default fields."""
    ordered = sorted(prompts, key=lambda prompt: prompt.prompt_id)
    payloads = [
        prompt.model_dump(mode="json", exclude_defaults=True) | {"prompt_id": prompt.prompt_id}
        for prompt in ordered
    ]
    return write_jsonl(path, payloads)
