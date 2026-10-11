"""Adaptive batch planning and the design extensions it relies on (no API calls)."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from seshat.data.generators import generate_prompts
from seshat.experiments.adaptive import fit_length_model, plan_next_batch
from seshat.experiments.config import CollectionConfig
from seshat.experiments.design import build_plan
from seshat.providers.catalog import load_catalog


def test_censored_only_category_is_pushed_above_the_cap() -> None:
    model = fit_length_model("code_generation", np.array([]), np.full(10, 512.0))
    assert model.median > 512
    assert model.tail(512) > 0.5


def test_uncensored_data_dominate_the_prior() -> None:
    values = np.full(40, 60.0) * np.exp(np.linspace(-0.3, 0.3, 40))
    model = fit_length_model("verbose_explanation", values, np.array([]))
    assert 45 < model.median < 80  # prior median is 1800


def test_expected_capped_is_bounded_by_cap() -> None:
    model = fit_length_model("x", np.array([]), np.array([]))
    assert model.expected_capped(256) <= 256
    assert model.expected_capped(4096) > model.expected_capped(256)


def _config(**prompts: object) -> CollectionConfig:
    return CollectionConfig.model_validate(
        {
            "experiment_id": "design_ext",
            "prompts": {"generate": {"count": 680, "seed": 1}, **prompts},
            "targets": [
                {
                    "provider": "mock-openai",
                    "model": "mock-gpt",
                    "max_output_tokens": [512],
                    "categories": ["short_answer", "classification"],
                },
                {
                    "provider": "mock-openai",
                    "model": "mock-gpt",
                    "max_output_tokens": [4096],
                    "categories": ["code_generation"],
                },
            ],
            "sampling": {"configs_per_prompt": 1},
            "repetitions": {"fraction": 0.0, "count": 1},
            "budget": {"max_requests": 60},
        }
    )


def test_category_restricted_targets_and_weights() -> None:
    config = _config(
        category_weights={"code_generation": 2, "short_answer": 1, "classification": 1}
    )
    prompts = generate_prompts(680, 1)
    plan = build_plan(config, prompts, load_catalog())
    caps = {(s.prompt_category, s.request.params.max_output_tokens) for s in plan.samples}
    assert caps <= {("short_answer", 512), ("classification", 512), ("code_generation", 4096)}
    counts = Counter(s.prompt_category for s in plan.samples)
    assert set(counts) == {"short_answer", "classification", "code_generation"}
    assert counts["code_generation"] == pytest.approx(30, abs=2)


def test_unweighted_order_is_unchanged() -> None:
    """Without weights the historical round-robin order (and sample ids) are kept."""
    base = CollectionConfig.model_validate(
        {
            "experiment_id": "stable",
            "prompts": {"generate": {"count": 68, "seed": 2}},
            "targets": [
                {"provider": "mock-openai", "model": "mock-gpt", "max_output_tokens": [256]}
            ],
            "sampling": {"configs_per_prompt": 1},
            "budget": {"max_requests": 40},
        }
    )
    prompts = generate_prompts(68, 2)
    first = build_plan(base, prompts, load_catalog())
    weighted = base.model_copy(
        update={"prompts": base.prompts.model_copy(update={"category_weights": {}})}
    )
    second = build_plan(weighted, prompts, load_catalog())
    assert [s.sample_id for s in first.samples] == [s.sample_id for s in second.samples]


def test_plan_from_priors_respects_budget_and_validates(tmp_path: Path) -> None:
    result = plan_next_batch(
        store_root=None,
        provider="openai",
        model="gpt-6.1-sol",
        experiment_id="adaptive_test",
        budget_usd=Decimal("2.00"),
        seed=11,
        prompt_pool_size=340,
    )
    report, config = result["report"], result["config"]
    CollectionConfig.model_validate(config)
    assert report["expected_cost_usd"] <= 2.00 * 0.85 + 1e-9
    assert report["efforts"] == ["low", "medium", "high"]
    caps = {t["max_output_tokens"][0]: t["categories"] for t in config["targets"]}
    assert any("code_generation" in cats for cap, cats in caps.items() if cap >= 2048)
    planned = {name: p["allocated_requests"] for name, p in report["categories"].items()}
    assert min(planned.values()) >= 4  # every workload stays represented


def test_plan_refuses_over_budget() -> None:
    with pytest.raises(ValueError, match="exceeds the approved"):
        plan_next_batch(
            store_root=None,
            provider="openai",
            model="gpt-6.1-sol",
            experiment_id="x",
            budget_usd=Decimal("5"),
        )
