from pathlib import Path

import pandas as pd

from seshat.data.io import load_jsonl
from seshat.data.models import PromptRecord, RawObservation
from seshat.experiments.runner import run_manifest


def test_mock_runner_is_resumable(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "condition_id": "cond_1",
                "experiment_id": "test",
                "prompt_id": "p1",
                "prompt_family_id": "pf1",
                "category": "qa",
                "provider": "mock",
                "model": "mock-v1",
                "effort": "low",
                "repetition": 1,
                "max_output_tokens": 512,
                "temperature": None,
            }
        ]
    )
    prompts = [
        PromptRecord(
            prompt_id="p1",
            prompt_family_id="pf1",
            category="qa",
            text="Explain entropy briefly.",
            source="fixture",
        )
    ]
    raw = tmp_path / "raw.jsonl"
    assert run_manifest(manifest, prompts, raw, sleep=lambda _: None) == 1
    assert run_manifest(manifest, prompts, raw, sleep=lambda _: None) == 0
    observations = load_jsonl(raw, RawObservation)
    assert len(observations) == 1
    assert observations[0].normalized_usage is not None
    assert observations[0].normalized_usage.output_tokens is not None
