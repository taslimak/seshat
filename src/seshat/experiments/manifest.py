"""Deterministic, cost-inspectable experiment manifest builder."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from seshat.data.io import load_json, load_jsonl, write_parquet
from seshat.data.models import EffortLevel, ExperimentConfig, ManifestRow, PromptRecord


def _hash_id(prefix: str, value: str, length: int = 20) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode()).hexdigest()[:length]}"


def load_prompts(paths: Sequence[str | Path]) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    seen: set[str] = set()
    for path in paths:
        for prompt in load_jsonl(path, PromptRecord):
            if prompt.prompt_id in seen:
                raise ValueError(f"Duplicate prompt_id: {prompt.prompt_id}")
            seen.add(prompt.prompt_id)
            prompts.append(prompt)
    return prompts


def build_manifest(config: ExperimentConfig, prompts: list[PromptRecord]) -> pd.DataFrame:
    conditions: list[ManifestRow] = []
    rng = random.Random(config.seed)
    provider_conditions: list[tuple[str, str, EffortLevel, int, float | None]] = []
    for provider, provider_config in sorted(config.providers.items()):
        for model in provider_config.models:
            for effort in provider_config.efforts:
                provider_conditions.append(
                    (
                        provider,
                        model,
                        effort,
                        provider_config.max_output_tokens,
                        provider_config.temperature,
                    )
                )
    for prompt in sorted(prompts, key=lambda item: item.prompt_id):
        selected = list(provider_conditions)
        if config.sample_models_per_prompt is not None:
            if config.sample_models_per_prompt > len(selected):
                raise ValueError(
                    "sample_models_per_prompt exceeds available provider/model/effort combinations"
                )
            local = random.Random(f"{config.seed}:{prompt.prompt_id}")
            selected = local.sample(selected, config.sample_models_per_prompt)
        for provider, model, effort, max_output_tokens, temperature in selected:
            condition_key = "|".join(
                [
                    config.experiment_id,
                    prompt.prompt_id,
                    provider,
                    model,
                    effort,
                    str(max_output_tokens),
                    str(temperature),
                ]
            )
            condition_id = _hash_id("cond", condition_key)
            for repetition in range(1, config.repetitions + 1):
                run_id = _hash_id("run", f"{condition_id}|{repetition}")
                conditions.append(
                    ManifestRow(
                        run_id=run_id,
                        condition_id=condition_id,
                        experiment_id=config.experiment_id,
                        prompt_id=prompt.prompt_id,
                        prompt_family_id=prompt.prompt_family_id,
                        category=prompt.category,
                        provider=provider,
                        model=model,
                        effort=effort,
                        repetition=repetition,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                    )
                )
    rng.shuffle(conditions)
    if config.max_requests is not None:
        conditions = conditions[: config.max_requests]
    frame = pd.DataFrame([item.model_dump(mode="json") for item in conditions])
    if frame.empty:
        raise ValueError("Experiment manifest contains no runs")
    return frame.sort_values("run_id").reset_index(drop=True)


def build_manifest_from_config(config_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    config = ExperimentConfig.model_validate(load_json(config_path))
    base = Path(config_path).resolve().parents[2]
    prompt_paths = [base / path for path in config.prompt_sources]
    prompts = load_prompts(prompt_paths)
    frame = build_manifest(config, prompts)
    write_parquet(output_path, frame)
    return frame
