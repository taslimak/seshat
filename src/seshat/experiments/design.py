"""Experimental design: which prompt runs under which configuration.

The design avoids the full prompt x configuration Cartesian product:

1. **Strata.** Each prompt is stratified by ``(length_stratum, category)``.
2. **Configurations.** Each target expands to its factor-level combinations
   (effort x thinking x max_output_tokens x caching x ...). A combination a model
   does not support is excluded and reported, never coerced.
3. **Balanced stratified assignment.** Prompts are visited round-robin across
   strata. Each gets ``configs_per_prompt`` distinct configurations, chosen greedily
   to minimize how often each configuration *and each factor level* has already been
   used, both overall and within the prompt's stratum. Seeded tie-breaking keeps it
   reproducible. This spreads factor levels evenly (space-filling) at a fraction of
   the cost.
4. **Repetitions.** A seeded, stratum-balanced subset of cells is repeated to
   estimate within-condition output variance.
5. **Budget cap.** ``budget.max_requests`` truncates the round-robin order, so the
   kept design stays balanced.

``design_report`` records counts per factor level and imbalance ratios.
"""

from __future__ import annotations

import itertools
import random
import statistics
import uuid
from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict

from seshat.data.generators.generator import LengthThresholds
from seshat.data.models import PromptRecord
from seshat.data.schemas.records import stable_uuid
from seshat.data.schemas.request import CacheConfig, GenerationParams, LLMRequest, Message, ToolSpec
from seshat.experiments.config import CollectionConfig, TargetConfig
from seshat.providers.catalog import Catalog
from seshat.providers.registry import adapter_class

FACTORS = ("target", "effort", "thinking_mode", "max_output_tokens", "caching", "structured_output")


class RunConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config_id: str
    provider: str
    model: str
    effort: str | None
    thinking_mode: str | None
    thinking_budget_tokens: int | None
    max_output_tokens: int
    caching: bool
    cache_ttl: str | None
    verbosity: str | None
    temperature: float | None
    structured_output: bool
    service_tier: str | None
    categories: tuple[str, ...] | None = None

    def applies_to(self, category: str) -> bool:
        return self.categories is None or category in self.categories

    def factor_levels(self) -> dict[str, str]:
        return {
            "target": f"{self.provider}/{self.model}",
            "effort": str(self.effort),
            "thinking_mode": str(self.thinking_mode),
            "max_output_tokens": str(self.max_output_tokens),
            "caching": str(self.caching),
            "structured_output": str(self.structured_output),
        }


class PlannedSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: uuid.UUID
    request_id: str
    experiment_id: str
    design_cell_id: str
    variation_id: str
    repetition_index: int
    length_stratum: str
    prompt_id: str
    prompt_family_id: str
    prompt_template_id: str | None
    prompt_category: str
    prompt_subcategory: str | None
    prompt_language: str | None
    prompt_source: str
    data_license: str | None
    generation_seed: int | None
    review_status: str
    provider: str
    model: str
    request: LLMRequest


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    config_hash: str
    samples: list[PlannedSample]
    report: dict[str, Any]


def expand_target(target: TargetConfig) -> list[RunConfiguration]:
    configs: list[RunConfiguration] = []
    for (
        effort,
        thinking,
        budget,
        max_out,
        caching,
        verbosity,
        temperature,
        structured,
    ) in itertools.product(
        target.efforts,
        target.thinking_modes,
        target.thinking_budget_tokens,
        target.max_output_tokens,
        target.caching,
        target.verbosity,
        target.temperature,
        target.structured_output,
    ):
        key = "|".join(
            str(v)
            for v in (
                target.provider,
                target.model,
                effort,
                thinking,
                budget,
                max_out,
                caching,
                target.cache_ttl if caching else None,
                verbosity,
                temperature,
                structured,
                target.service_tier,
            )
        )
        categories = tuple(sorted(set(target.categories))) if target.categories else None
        if categories:
            # Only category-restricted targets change the key, so existing ids are stable.
            key += "|categories=" + ",".join(categories)
        configs.append(
            RunConfiguration(
                config_id=f"cfg_{stable_uuid(key).hex[:12]}",
                provider=target.provider,
                model=target.model,
                effort=effort,
                thinking_mode=thinking,
                thinking_budget_tokens=budget,
                max_output_tokens=max_out,
                caching=caching,
                cache_ttl=target.cache_ttl if caching else None,
                verbosity=verbosity,
                temperature=temperature,
                structured_output=structured,
                service_tier=target.service_tier,
                categories=categories,
            )
        )
    return configs


def build_request(prompt: PromptRecord, config: RunConfiguration) -> LLMRequest:
    """The exact request for ``prompt`` under ``config``."""
    messages = list(prompt.messages or [])
    system = None
    if messages and messages[0].role == "system":
        system = messages[0].content
        messages = messages[1:]
    if not messages:
        messages_out = [Message(role="user", content=prompt.text)]
    else:
        messages_out = [Message(role=m.role, content=m.content) for m in messages]
    from seshat.data.features.extract import requested_output_format

    params = GenerationParams(
        max_output_tokens=config.max_output_tokens,
        effort=config.effort,
        thinking_mode=config.thinking_mode,
        thinking_budget_tokens=config.thinking_budget_tokens,
        temperature=config.temperature,
        verbosity=config.verbosity,
        tool_choice="auto" if prompt.tools else None,
        cache=CacheConfig(
            enabled=config.caching,
            ttl=config.cache_ttl,
            key=prompt.prompt_family_id if config.caching else None,
        ),
        service_tier=config.service_tier,
    )
    request = LLMRequest(
        provider=config.provider,
        model=config.model,
        api_surface=adapter_class(config.provider).api_surface,
        system=system,
        messages=messages_out,
        tools=[
            ToolSpec(name=t.name, description=t.description, input_schema=t.input_schema)
            for t in prompt.tools
        ],
        params=params,
        provider_options=(
            {"_mock_category": prompt.category}
            if not adapter_class(config.provider).makes_external_calls
            else {}
        ),
    )
    if config.structured_output and requested_output_format(request) == "json":
        family = adapter_class(config.provider).family
        fmt = "json_schema" if family == "anthropic" else "json"
        request = request.model_copy(
            update={
                "params": params.model_copy(
                    update={
                        "response_format": fmt,
                        "response_schema": {"type": "object"} if fmt == "json_schema" else None,
                    }
                )
            }
        )
    return request


def _length_stratum(prompt: PromptRecord, thresholds: LengthThresholds) -> str:
    return thresholds.stratum(len(prompt.text.split()))


def _imbalance(counter: Counter[str]) -> dict[str, Any]:
    values = list(counter.values())
    if not values:
        return {"levels": {}, "max_min_ratio": None, "cv": None}
    mean = statistics.fmean(values)
    return {
        "levels": dict(sorted(counter.items())),
        "max_min_ratio": round(max(values) / min(values), 3),
        "cv": round(statistics.pstdev(values) / mean, 3) if mean else None,
    }


def _interleave(
    by_stratum: dict[tuple[str, str], list[PromptRecord]],
    weights: dict[str, float] | None,
) -> list[PromptRecord]:
    """Visit (stratum, category) queues round-robin, or by smooth weighted round-robin.

    Without weights every queue gets one turn per round (the historical order). With
    weights, a category's share of the order is proportional to its weight, split
    evenly over that category's length strata; weight 0 drops the category.
    """
    keys = sorted(by_stratum)
    queues = {key: list(by_stratum[key]) for key in keys}
    ordered: list[PromptRecord] = []
    if not weights:
        while any(queues.values()):
            for key in keys:
                if queues[key]:
                    ordered.append(queues[key].pop(0))
        return ordered
    strata_per_category = Counter(category for _, category in keys)
    share = {key: float(weights.get(key[1], 1.0)) / strata_per_category[key[1]] for key in keys}
    credit = dict.fromkeys(keys, 0.0)
    while True:
        active = [key for key in keys if queues[key] and share[key] > 0]
        if not active:
            return ordered
        total = sum(share[key] for key in active)
        for key in active:
            credit[key] += share[key]
        pick = max(active, key=lambda key: (credit[key], -keys.index(key)))
        credit[pick] -= total
        ordered.append(queues[pick].pop(0))


def build_plan(config: CollectionConfig, prompts: list[PromptRecord], catalog: Catalog) -> Plan:
    rng = random.Random(f"design|{config.experiment_id}|{config.seed}")
    thresholds = LengthThresholds(config.strata.short_max_words, config.strata.medium_max_words)
    if config.prompts.include_categories:
        prompts = [p for p in prompts if p.category in set(config.prompts.include_categories)]
    if not prompts:
        raise ValueError("No prompts to plan (check prompt sources and include_categories)")
    configs = [c for target in config.targets for c in expand_target(target)]
    excluded: Counter[str] = Counter()

    by_stratum: dict[tuple[str, str], list[PromptRecord]] = defaultdict(list)
    for prompt in sorted(prompts, key=lambda p: p.prompt_id):
        by_stratum[(_length_stratum(prompt, thresholds), prompt.category)].append(prompt)
    for bucket in by_stratum.values():
        rng.shuffle(bucket)
    ordered = _interleave(by_stratum, config.prompts.category_weights)

    global_use: Counter[str] = Counter()
    stratum_use: Counter[tuple[str, str]] = Counter()
    level_use: Counter[tuple[str, str, str]] = Counter()
    target_use: Counter[str] = Counter()
    stratum_target_use: Counter[tuple[str, str]] = Counter()
    base: list[tuple[PromptRecord, RunConfiguration, LLMRequest, str]] = []
    for prompt in ordered:
        stratum = _length_stratum(prompt, thresholds)
        by_target: dict[str, list[tuple[RunConfiguration, LLMRequest]]] = defaultdict(list)
        for cfg in configs:
            if not cfg.applies_to(prompt.category):
                continue
            caps = catalog.capabilities(cfg.provider, cfg.model)
            request = build_request(prompt, cfg)
            problems = caps.unsupported_settings(request)
            if problems:
                excluded[problems[0].split(" ")[0]] += 1
                continue
            live = adapter_class(cfg.provider).makes_external_calls
            if (
                live
                and config.prompts.require_approved_for_live
                and prompt.review_status != "approved"
            ):
                excluded["prompt_not_approved_for_live"] += 1
                continue
            by_target[f"{cfg.provider}/{cfg.model}"].append((cfg, request))
        chosen: set[str] = set()
        targets_used: Counter[str] = Counter()
        for _ in range(config.sampling.configs_per_prompt):
            # Stage 1: the least-used target (overall and in this stratum), preferring
            # targets this prompt has not used yet, so prompts span providers.
            open_targets = [
                t
                for t, options in by_target.items()
                if any(c.config_id not in chosen for c, _ in options)
            ]
            if not open_targets:
                break
            target = min(
                open_targets,
                key=lambda t: (
                    targets_used[t],
                    target_use[t] + 2 * stratum_target_use[(stratum, t)],
                    rng.random(),
                ),
            )
            # Stage 2: within the target, the configuration whose factor levels are rarest.
            options = [(c, r) for c, r in by_target[target] if c.config_id not in chosen]
            cfg, request = min(
                options,
                key=lambda item: (
                    global_use[item[0].config_id]
                    + 2 * stratum_use[(stratum, item[0].config_id)]
                    + sum(
                        level_use[(stratum, f, lvl)] for f, lvl in item[0].factor_levels().items()
                    ),
                    rng.random(),
                ),
            )
            chosen.add(cfg.config_id)
            targets_used[target] += 1
            target_use[target] += 1
            stratum_target_use[(stratum, target)] += 1
            global_use[cfg.config_id] += 1
            stratum_use[(stratum, cfg.config_id)] += 1
            for factor, level in cfg.factor_levels().items():
                level_use[(stratum, factor, level)] += 1
            base.append((prompt, cfg, request, stratum))

    reps = config.repetitions
    extra_per_rep_cell = max(reps.count - 1, 0)
    expected_factor = 1 + reps.fraction * extra_per_rep_cell
    base_limit = min(len(base), int(config.budget.max_requests / expected_factor) or 1)
    trimmed = len(base) - base_limit
    base = base[:base_limit]

    repeat_cells: set[int] = set()
    if extra_per_rep_cell and reps.fraction > 0:
        per_stratum: dict[str, list[int]] = defaultdict(list)
        for idx, (_, _, _, stratum) in enumerate(base):
            per_stratum[stratum].append(idx)
        repeat_target = round(len(base) * reps.fraction)
        pools = [rng.sample(v, len(v)) for _, v in sorted(per_stratum.items())]
        while len(repeat_cells) < repeat_target and any(pools):
            for pool in pools:
                if pool and len(repeat_cells) < repeat_target:
                    repeat_cells.add(pool.pop())

    samples: list[PlannedSample] = []
    for idx, (prompt, cfg, request, stratum) in enumerate(base):
        repetitions = reps.count if idx in repeat_cells else 1
        for repetition in range(repetitions):
            if len(samples) >= config.budget.max_requests:
                break
            sample_id = stable_uuid(
                config.experiment_id, prompt.prompt_id, cfg.config_id, repetition
            )
            samples.append(
                PlannedSample(
                    sample_id=sample_id,
                    request_id=f"req_{sample_id.hex[:24]}",
                    experiment_id=config.experiment_id,
                    design_cell_id=f"{stratum}|{prompt.category}|{cfg.config_id}",
                    variation_id=cfg.config_id,
                    repetition_index=repetition,
                    length_stratum=stratum,
                    prompt_id=prompt.prompt_id,
                    prompt_family_id=prompt.prompt_family_id,
                    prompt_template_id=prompt.prompt_template_id,
                    prompt_category=prompt.category,
                    prompt_subcategory=prompt.subcategory,
                    prompt_language=prompt.language,
                    prompt_source=prompt.source,
                    data_license=prompt.license_id,
                    generation_seed=prompt.generation_seed,
                    review_status=prompt.review_status,
                    provider=cfg.provider,
                    model=cfg.model,
                    request=request,
                )
            )

    factor_counts: dict[str, Counter[str]] = {factor: Counter() for factor in FACTORS}
    within_target: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    config_by_id = {c.config_id: c for c in configs}
    for sample in samples:
        levels = config_by_id[sample.variation_id].factor_levels()
        for factor, level in levels.items():
            factor_counts[factor][level] += 1
            if factor != "target":
                within_target[levels["target"]][factor][level] += 1
    report = {
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash(),
        "prompts_considered": len(prompts),
        "configurations": len(configs),
        "full_cartesian_requests": len(prompts) * len(configs),
        "planned_requests": len(samples),
        "base_cells": len(base),
        "trimmed_cells_for_budget": trimmed,
        "repeated_cells": len(repeat_cells),
        "repetitions_per_repeated_cell": reps.count,
        "excluded_combinations": dict(sorted(excluded.items())),
        "by_length_stratum": _imbalance(Counter(s.length_stratum for s in samples)),
        "by_category": _imbalance(Counter(s.prompt_category for s in samples)),
        "by_factor": {factor: _imbalance(counter) for factor, counter in factor_counts.items()},
        # Factor levels only exist on some targets, so balance is judged within a target.
        "by_factor_within_target": {
            target: {
                factor: _imbalance(counter)
                for factor, counter in sorted(factors.items())
                if len(counter) > 1
            }
            for target, factors in sorted(within_target.items())
        },
    }
    return Plan(
        experiment_id=config.experiment_id,
        config_hash=config.config_hash(),
        samples=samples,
        report=report,
    )
