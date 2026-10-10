from seshat.data.models import ExperimentConfig, PromptRecord
from seshat.experiments.manifest import build_manifest


def test_manifest_is_deterministic_and_unique() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment_id": "test",
            "prompt_sources": ["unused.jsonl"],
            "providers": {
                "mock": {"models": ["mock-v1"], "efforts": ["low", "high"]}
            },
            "repetitions": 2,
            "seed": 42,
        }
    )
    prompts = [
        PromptRecord(
            prompt_id="p1",
            prompt_family_id="pf1",
            category="coding",
            text="write code",
            source="fixture",
        )
    ]
    first = build_manifest(config, prompts)
    second = build_manifest(config, prompts)
    assert first.equals(second)
    assert len(first) == 4
    assert first["run_id"].is_unique
    assert first["condition_id"].nunique() == 2
