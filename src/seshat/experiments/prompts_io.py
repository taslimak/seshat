"""Load prompt-bank files referenced by a collection config."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from seshat.data.io import load_jsonl
from seshat.data.models import PromptRecord


def load_prompt_files(paths: Sequence[str | Path]) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    seen: set[str] = set()
    for path in paths:
        for prompt in load_jsonl(path, PromptRecord):
            if prompt.prompt_id in seen:
                raise ValueError(f"Duplicate prompt_id {prompt.prompt_id} in {path}")
            seen.add(prompt.prompt_id)
            prompts.append(prompt)
    return prompts
