from __future__ import annotations

from pathlib import Path

import pytest

from seshat.data.models import PromptRecord
from seshat.data.store import DatasetStore
from seshat.experiments.manifest import load_prompts

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_BANK = REPO_ROOT / "prompts" / "hand_authored.jsonl"


@pytest.fixture
def prompts() -> list[PromptRecord]:
    return load_prompts([PROMPT_BANK])


@pytest.fixture
def store(tmp_path: Path) -> DatasetStore:
    return DatasetStore(tmp_path / "store")
