from collections import Counter
from collections.abc import Callable

import pytest

from seshat.data.generators import TEMPLATES, generate_prompts
from seshat.experiments.config import CollectionConfig, load_collection_config
from seshat.experiments.design import build_plan, build_request, expand_target
from seshat.providers.catalog import load_catalog
from tests.data.conftest import REPO_ROOT


def test_generates_at_least_100_deterministic_prompts_offline() -> None:
    first, second = generate_prompts(120, 42), generate_prompts(120, 42)
    assert len(first) == 120
    assert [p.model_dump() for p in first] == [p.model_dump() for p in second]
    assert len({p.prompt_id for p in first}) == 120
    assert {p.category for p in first} == {category for category, _ in TEMPLATES.values()}
    strata = Counter(p.metadata["length_stratum"] for p in first)
    assert set(strata) == {"short", "medium", "long"}
    assert all(p.is_synthetic and p.prompt_template_id and p.generation_seed == 42 for p in first)
    assert any(p.tools for p in first) and any(p.message_count > 2 for p in first)
    assert generate_prompts(10, 43)[0].text != first[0].text


def test_variants_of_a_template_topic_share_a_family() -> None:
    prompts = generate_prompts(200, 1)
    by_family = Counter(p.prompt_family_id for p in prompts)
    assert max(by_family.values()) > 1
    for prompt in prompts:
        assert prompt.prompt_family_id.startswith(f"fam_{prompt.prompt_template_id}_")


def test_design_is_deterministic_balanced_and_far_smaller_than_cartesian(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config()
    prompts = generate_prompts(34, 3)
    plan, again = (
        build_plan(config, prompts, load_catalog()),
        build_plan(config, prompts, load_catalog()),
    )
    assert [s.sample_id for s in plan.samples] == [s.sample_id for s in again.samples]
    assert (
        len({s.sample_id for s in plan.samples}) == len(plan.samples) <= config.budget.max_requests
    )
    report = plan.report
    assert report["planned_requests"] < report["full_cartesian_requests"]
    assert report["by_factor"]["target"]["max_min_ratio"] <= 1.5
    assert report["repeated_cells"] > 0
    reps = Counter(s.design_cell_id + s.prompt_id for s in plan.samples)
    assert max(reps.values()) == config.repetitions.count


def test_budget_cap_trims_design(small_config: Callable[..., CollectionConfig]) -> None:
    plan = build_plan(
        small_config(budget={"max_requests": 20}), generate_prompts(34, 3), load_catalog()
    )
    assert len(plan.samples) == 20
    assert plan.report["trimmed_cells_for_budget"] > 0


def test_unsupported_settings_are_excluded_not_coerced(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config(
        targets=[
            {
                "provider": "mock-anthropic",
                "model": "mock-claude",
                "temperature": [None, 0.5],
                "max_output_tokens": [256],
            }
        ]
    )
    plan = build_plan(config, generate_prompts(17, 2), load_catalog())
    assert all(s.request.params.temperature is None for s in plan.samples)
    assert plan.report["excluded_combinations"]["temperature"] > 0


def test_live_targets_require_approved_prompts(
    small_config: Callable[..., CollectionConfig],
) -> None:
    prompts = [
        p.model_copy(update={"review_status": "unreviewed", "reviewed_by": None})
        for p in generate_prompts(17, 2)
    ]
    config = small_config(
        targets=[{"provider": "openai", "model": "gpt-6-luna", "max_output_tokens": [256]}]
    )
    plan = build_plan(config, prompts, load_catalog())
    assert plan.samples == []
    assert plan.report["excluded_combinations"] == {"prompt_not_approved_for_live": 17}
    approved = build_plan(config, generate_prompts(17, 2), load_catalog())
    assert len(approved.samples) > 0


def test_build_request_maps_prompt_structure_and_mock_hints() -> None:
    prompt = next(p for p in generate_prompts(60, 5) if p.tools)
    config = expand_target(
        load_collection_config(REPO_ROOT / "configs" / "data_collection.yaml").targets[0]
    )[0]
    request = build_request(prompt, config)
    assert request.tools and request.params.tool_choice == "auto"
    assert request.system is not None
    assert request.provider_options["_mock_category"] == prompt.category


def test_shipped_configs_load() -> None:
    for name in ("data_collection.yaml", "experiments/live_pilot.yaml"):
        assert load_collection_config(REPO_ROOT / "configs" / name).targets


def test_generated_near_duplicates_share_a_family() -> None:
    from seshat.prompts.dedup import near_duplicate_pairs

    prompts = generate_prompts(150, 42)
    family = {p.prompt_id: p.prompt_family_id for p in prompts}
    assert all(
        family[pair.prompt_id] == family[pair.other_id]
        for pair in near_duplicate_pairs(prompts, 0.8)
    )


def test_bank_files_load_and_reject_duplicate_ids() -> None:
    from seshat.experiments.prompts_io import load_prompt_files

    bank = REPO_ROOT / "prompts" / "hand_authored.jsonl"
    prompts = load_prompt_files([bank])
    assert len(prompts) == 36 and all(p.review_status == "approved" for p in prompts)
    with pytest.raises(ValueError, match="Duplicate prompt_id"):
        load_prompt_files([bank, bank])
