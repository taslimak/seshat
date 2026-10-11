"""Adaptive, budget-bounded planning of the next real-data collection batch.

The planner reads the consolidated store, models each prompt category's output
length, and writes (1) a collection config the existing collector runs unchanged
and (2) a dry-run report for human approval. It never sends requests.

Model
-----
Per category, output tokens ``Y`` (provider-reported, reasoning included where
the provider counts it in output) are modelled as log-normal. Truncated
responses are **right-censored** at their cap: they contribute ``P(Y > cap)`` to
the likelihood, never ``Y = cap``. Parameters are the MAP estimate under weak
priors (median from :data:`PRIOR_MEDIAN_OUTPUT`, log-sd around 0.8), found by a
grid search -- so a category with *only* censored data is pushed above its cap
rather than silently estimated at it. Planned efforts scale ``Y`` by
:data:`EFFORT_OUTPUT_MULTIPLIER` (planning assumptions, replaced by data once
effort levels are observed).

Decisions
---------
* **Cap**: the smallest ladder value with ``P(Y > cap) <= max_censor_rate`` at the
  highest planned effort (else the top of the ladder, flagged).
* **Expected cost** per request: ``in_tokens * input_rate + E[min(Y, cap)] *
  output_rate``, averaged over planned efforts; the worst case uses the cap.
* **Allocation**: greedy maximisation of ``sum_k log(1 + usable_k)`` (diminishing
  returns => balanced coverage, extra weight on under-represented categories)
  per ``cost ** cost_exponent``. With ``cost_exponent = 1`` cheap short-output
  categories crowd out long ones; the default ``0.5`` tempers that, which is
  equivalent to valuing a label in proportion to the square root of its token
  volume -- appropriate for a *cost* predictor whose dollar error is dominated by
  long outputs. A per-category floor and share cap keep every workload
  represented. Allocation stops when expected spend reaches
  ``fill_fraction * budget``. The collector's reservation-based hard cap still
  bounds actual estimated spend at ``budget``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Floor of 512: reasoning tokens count toward the output cap on reasoning models, so
# even one-word answers can truncate at small caps under high effort.
DEFAULT_CAP_LADDER = (512, 1024, 2048, 4096)
MAX_BATCH_BUDGET_USD = Decimal("2.00")
# Planning priors (median output tokens at medium effort), overridden by data.
PRIOR_MEDIAN_OUTPUT: dict[str, float] = {
    "classification": 80,
    "code_generation": 1500,
    "code_review": 900,
    "creative_writing": 600,
    "debugging": 600,
    "extraction": 300,
    "long_context_analysis": 400,
    "mathematics": 600,
    "multi_turn": 300,
    "question_answering": 350,
    "reasoning": 500,
    "short_answer": 40,
    "structured_json": 150,
    "summarization": 200,
    "tool_calling": 120,
    "translation": 300,
    "verbose_explanation": 1800,
}
DEFAULT_PRIOR_MEDIAN = 400.0
PRIOR_LOG_SD = 0.8
EFFORT_OUTPUT_MULTIPLIER: dict[str | None, float] = {
    None: 1.0,
    "none": 0.7,
    "minimal": 0.7,
    "low": 0.75,
    "medium": 1.0,
    "high": 1.5,
    "xhigh": 2.0,
    "max": 2.5,
}
_MU_GRID = np.arange(math.log(5), math.log(50_000), 0.01)
_SIGMA_GRID = np.arange(0.15, 2.51, 0.025)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return np.asarray(ndtr(x), dtype=float)


def _norm_logsf(x: np.ndarray) -> np.ndarray:
    from scipy.special import log_ndtr

    return np.asarray(log_ndtr(-x), dtype=float)


@dataclass
class LengthModel:
    category: str
    mu: float
    sigma: float
    n_uncensored: int
    n_censored: int
    source: str  # "prior" | "map_censored_lognormal"

    @property
    def median(self) -> float:
        return math.exp(self.mu)

    def tail(self, cap: float, multiplier: float = 1.0) -> float:
        z = (math.log(cap) - self.mu - math.log(multiplier)) / self.sigma
        return float(1.0 - _norm_cdf(np.array([z]))[0])

    def expected_capped(self, cap: float, multiplier: float = 1.0) -> float:
        """E[min(Y, cap)] in closed form for a log-normal Y."""
        mu = self.mu + math.log(multiplier)
        s = self.sigma
        z = (math.log(cap) - mu) / s
        cdf = _norm_cdf(np.array([z - s, z]))
        return float(math.exp(mu + s * s / 2) * cdf[0] + cap * (1.0 - cdf[1]))


def fit_length_model(
    category: str, uncensored: np.ndarray, censored_caps: np.ndarray
) -> LengthModel:
    """MAP log-normal fit with right-censoring (grid search, no external optimiser)."""
    prior_mu = math.log(PRIOR_MEDIAN_OUTPUT.get(category, DEFAULT_PRIOR_MEDIAN))
    unc = np.log(np.clip(np.asarray(uncensored, dtype=float), 1.0, None))
    cen = np.log(np.clip(np.asarray(censored_caps, dtype=float), 1.0, None))
    if unc.size == 0 and cen.size == 0:
        return LengthModel(category, prior_mu, PRIOR_LOG_SD, 0, 0, "prior")
    mu = _MU_GRID[:, None]
    sigma = _SIGMA_GRID[None, :]
    log_post = (
        -0.5 * ((mu - prior_mu) / 1.0) ** 2
        - 0.5 * ((np.log(sigma) - math.log(PRIOR_LOG_SD)) / 0.5) ** 2
    )
    for value in unc:  # log-normal density on the log scale (Jacobian is constant)
        log_post = log_post - np.log(sigma) - 0.5 * ((value - mu) / sigma) ** 2
    for value in cen:
        log_post = log_post + _norm_logsf((value - mu) / sigma)
    i, j = np.unravel_index(int(np.argmax(log_post)), log_post.shape)
    return LengthModel(
        category,
        float(_MU_GRID[i]),
        float(_SIGMA_GRID[j]),
        int(unc.size),
        int(cen.size),
        "map_censored_lognormal",
    )


@dataclass
class CategoryPlan:
    category: str
    model: dict[str, Any]
    observed_rows: int
    observed_usable: int
    observed_censor_rate: float | None
    cap: int
    predicted_censor_rate: float
    cap_at_top_of_ladder: bool
    mean_input_tokens: float
    expected_output_tokens: float
    expected_cost_usd: float
    worst_case_cost_usd: float
    allocated_requests: int = 0
    expected_new_usable: float = 0.0
    reasons: list[str] = field(default_factory=list)


def observed_by_category(store_root: str | Path | None, provider: str, model: str) -> pd.DataFrame:
    """Real observations for ``provider/model`` from the store (empty frame if none)."""
    columns = [
        "prompt_category",
        "actual_output_tokens",
        "max_output_tokens",
        "censored",
        "label_quality",
    ]
    if store_root is None:
        return pd.DataFrame(columns=columns)
    from seshat.data.evaluation import real_observation_frame

    try:
        frame = real_observation_frame(store_root)
    except ValueError:
        return pd.DataFrame(columns=columns)
    frame = frame[(frame["provider"] == provider) & (frame["requested_model"] == model)]
    return pd.DataFrame(frame[columns])


def _input_tokens_by_category(prompts: list[Any]) -> dict[str, float]:
    """Approximate input tokens (chars/4 + small per-message overhead) per category."""
    totals: dict[str, list[float]] = {}
    for prompt in prompts:
        chars = sum(len(m.content) for m in (prompt.messages or [])) or len(prompt.text)
        tool_chars = sum(len(json.dumps(t.model_dump(mode="json"))) for t in prompt.tools)
        messages = max(1, len(prompt.messages or []))
        totals.setdefault(prompt.category, []).append((chars + tool_chars) / 4 + 4 * messages)
    return {k: float(np.mean(v)) for k, v in totals.items()}


def plan_next_batch(
    *,
    store_root: str | Path | None,
    provider: str,
    model: str,
    experiment_id: str,
    budget_usd: Decimal = MAX_BATCH_BUDGET_USD,
    efforts: list[str | None] | None = None,
    cap_ladder: tuple[int, ...] = DEFAULT_CAP_LADDER,
    max_censor_rate: float = 0.10,
    fill_fraction: float = 0.85,
    min_requests_per_category: int = 6,
    max_share_per_category: float = 0.12,
    cost_exponent: float = 0.5,
    repetition_fraction: float = 0.10,
    repetition_count: int = 2,
    prompt_pool_size: int = 1020,
    seed: int | None = None,
    allow_over_budget: bool = False,
    pricing_dir: str | Path | None = None,
    on: date | None = None,
) -> dict[str, Any]:
    """Return ``{"config": <collector config>, "report": <dry-run report>}``."""
    from seshat.data.generators import generate_prompts
    from seshat.data.pricing import PriceTable
    from seshat.providers.catalog import load_catalog

    if budget_usd > MAX_BATCH_BUDGET_USD and not allow_over_budget:
        raise ValueError(
            f"Budget ${budget_usd} exceeds the approved per-run maximum ${MAX_BATCH_BUDGET_USD}"
        )
    if not 0 < fill_fraction <= 1:
        raise ValueError("fill_fraction must be in (0, 1]")
    capabilities = load_catalog().capabilities(provider, model)
    supported = capabilities.efforts or frozenset()
    if efforts is None:
        efforts = [e for e in ("low", "medium", "high") if e in supported] or [None]
    unsupported = [e for e in efforts if e is not None and e not in supported]
    if unsupported:
        raise ValueError(f"{provider}/{model} does not support efforts {unsupported}")
    entry = PriceTable.load(pricing_dir).lookup(provider, model, on or date.today())
    if entry is None or entry.rates.input is None or entry.rates.output is None:
        raise ValueError(f"No price entry for {provider}/{model}; cannot plan a paid batch")
    in_rate = float(entry.rates.input) / 1e6
    out_rate = float(entry.rates.output) / 1e6
    seed = seed if seed is not None else int(datetime.now(UTC).strftime("%Y%m%d%H"))
    pool = generate_prompts(prompt_pool_size, seed)
    input_tokens = _input_tokens_by_category(pool)
    observed = observed_by_category(store_root, provider, model)

    multipliers = [EFFORT_OUTPUT_MULTIPLIER.get(e, 1.0) for e in efforts]
    plans: dict[str, CategoryPlan] = {}
    for category in sorted(input_tokens):
        rows = observed[observed["prompt_category"] == category]
        censored_mask = rows["censored"].astype(bool)
        usable = rows[(~censored_mask) & (rows["label_quality"] == "verified")]
        fitted = fit_length_model(
            category,
            rows.loc[~censored_mask, "actual_output_tokens"].to_numpy(dtype=float),
            rows.loc[censored_mask, "max_output_tokens"].to_numpy(dtype=float),
        )
        top = max(multipliers)
        cap = next(
            (c for c in sorted(cap_ladder) if fitted.tail(c, top) <= max_censor_rate),
            max(cap_ladder),
        )
        censor = float(np.mean([fitted.tail(cap, m) for m in multipliers]))
        expected_out = float(np.mean([fitted.expected_capped(cap, m) for m in multipliers]))
        reasons: list[str] = []
        if len(rows) and censored_mask.mean() > max_censor_rate:
            reasons.append(
                f"observed censor rate {censored_mask.mean():.0%} at cap "
                f"{int(rows['max_output_tokens'].max())} -> cap {cap}"
            )
        if len(usable) == 0:
            reasons.append("no usable labels yet")
        plans[category] = CategoryPlan(
            category=category,
            model=asdict(fitted) | {"median": round(fitted.median, 1)},
            observed_rows=int(len(rows)),
            observed_usable=int(len(usable)),
            observed_censor_rate=float(censored_mask.mean()) if len(rows) else None,
            cap=int(cap),
            predicted_censor_rate=round(censor, 3),
            cap_at_top_of_ladder=fitted.tail(max(cap_ladder), top) > max_censor_rate,
            mean_input_tokens=round(input_tokens[category], 1),
            expected_output_tokens=round(expected_out, 1),
            expected_cost_usd=input_tokens[category] * in_rate + expected_out * out_rate,
            worst_case_cost_usd=input_tokens[category] * in_rate + cap * out_rate,
            reasons=reasons,
        )

    # Greedy allocation: maximise sum log(1 + usable) per expected dollar.
    overhead = 1.0 + repetition_fraction * max(repetition_count - 1, 0)
    target_spend = float(budget_usd) * fill_fraction
    spend = 0.0

    def unit_cost(plan: CategoryPlan) -> float:
        return plan.expected_cost_usd * overhead

    for plan in plans.values():  # floor first: keep every workload represented
        for _ in range(min_requests_per_category):
            if spend + unit_cost(plan) > target_spend:
                break
            plan.allocated_requests += 1
            spend += unit_cost(plan)
    while True:
        total = sum(p.allocated_requests for p in plans.values())
        share_limit = max(
            min_requests_per_category, math.ceil(max_share_per_category * (total + 1))
        )
        best: tuple[float, str] | None = None
        for name, plan in plans.items():
            if plan.allocated_requests >= share_limit:
                continue
            if spend + unit_cost(plan) > target_spend:
                continue
            p_usable = 1.0 - plan.predicted_censor_rate
            current = plan.observed_usable + plan.allocated_requests * p_usable
            gain = math.log1p(current + p_usable) - math.log1p(current)
            score = gain / unit_cost(plan) ** cost_exponent
            if best is None or score > best[0]:
                best = (score, name)
        if best is None:
            break
        plans[best[1]].allocated_requests += 1
        spend += unit_cost(plans[best[1]])
    for plan in plans.values():
        plan.expected_new_usable = round(
            plan.allocated_requests * overhead * (1 - plan.predicted_censor_rate), 1
        )

    base_requests = sum(p.allocated_requests for p in plans.values())
    max_requests = math.ceil(base_requests * overhead)
    by_cap: dict[int, list[str]] = {}
    for plan in plans.values():
        if plan.allocated_requests:
            by_cap.setdefault(plan.cap, []).append(plan.category)
    config: dict[str, Any] = {
        "experiment_id": experiment_id,
        "description": "Adaptive calibration batch (censoring-aware caps, weighted coverage)",
        "seed": seed,
        "prompts": {
            "generate": {"count": prompt_pool_size, "seed": seed},
            "bank_files": [],
            "category_weights": {
                name: float(plan.allocated_requests) for name, plan in sorted(plans.items())
            },
            "require_approved_for_live": True,
        },
        "targets": [
            {
                "provider": provider,
                "model": model,
                "efforts": efforts,
                "max_output_tokens": [cap],
                "categories": sorted(categories),
            }
            for cap, categories in sorted(by_cap.items())
        ],
        "sampling": {"configs_per_prompt": 1},
        "repetitions": {"fraction": repetition_fraction, "count": repetition_count},
        "budget": {"max_requests": max_requests, "max_cost_usd": str(budget_usd)},
        "execution": {
            # 263 requests at 5/min took ~53 min, beyond the workflow timeout; long
            # (4096-token) outputs are latency-bound, so run a few in parallel.
            "concurrency": 4,
            "per_provider_concurrency": 4,
            "timeout_seconds": 180,
            "max_retries": 0,
            "preflight": "approximate",
            "rate_limits": {provider: {"requests_per_minute": 30}},
        },
        "privacy": {"store_prompt_text": False, "store_response_text": False},
    }
    # Verify against the plan the collector will actually build from this config, so
    # the report shows realized counts (the design's balancing shifts a few cells).
    from seshat.data.pricing import max_cost_estimate
    from seshat.experiments.config import CollectionConfig
    from seshat.experiments.design import build_plan
    from seshat.providers.registry import adapter_class

    family = adapter_class(provider).family

    def realize(limit: int) -> tuple[Any, dict[str, dict[str, float]]]:
        config["budget"]["max_requests"] = limit
        built = build_plan(CollectionConfig.model_validate(config), pool, load_catalog())
        rows: dict[str, dict[str, float]] = {
            name: {"requests": 0, "expected_cost_usd": 0.0, "worst_case_usd": 0.0, "usable": 0.0}
            for name in plans
        }
        for sample in built.samples:
            plan = plans[sample.prompt_category]
            request = sample.request
            chars = sum(len(m.content) for m in request.messages) + len(request.system or "")
            chars += sum(
                len(json.dumps(t.input_schema)) + len(t.description or "") for t in request.tools
            )
            in_tokens = chars / 4 + 4 * len(request.messages)
            cap = int(request.params.max_output_tokens or plan.cap)
            row = rows[sample.prompt_category]
            row["requests"] += 1
            row["expected_cost_usd"] += in_tokens * in_rate + plan.expected_output_tokens * out_rate
            row["worst_case_usd"] += float(
                max_cost_estimate(family, entry, input_tokens=int(in_tokens), max_output_tokens=cap)
            )
            row["usable"] += 1 - plan.predicted_censor_rate
        return built, rows

    # The design's balancing shifts a few cells between categories; trim the request
    # limit (the design trims in balanced round-robin order) until the realized
    # expected spend is back within fill_fraction * budget.
    limit = max_requests
    realized_plan, realized = realize(limit)
    for _ in range(20):
        expected = sum(r["expected_cost_usd"] for r in realized.values())
        if expected <= target_spend or limit <= len(plans):
            break
        limit = max(len(plans), min(limit - 1, int(limit * target_spend / expected)))
        realized_plan, realized = realize(limit)
    for name, plan in plans.items():
        plan.allocated_requests = int(realized[name]["requests"])
        plan.expected_new_usable = round(realized[name]["usable"], 1)
    expected_cost = sum(r["expected_cost_usd"] for r in realized.values())
    worst_cost = sum(r["worst_case_usd"] for r in realized.values())
    expected_usable = sum(r["usable"] for r in realized.values())
    if expected_cost > float(budget_usd):
        raise ValueError(
            f"Realized plan expects ${expected_cost:.2f}, above the ${budget_usd} budget"
        )
    max_requests = len(realized_plan.samples)
    report = {
        "experiment_id": experiment_id,
        "provider": provider,
        "model": model,
        "pricing": {
            "version": entry.pricing_version,
            "input_usd_per_million": str(entry.rates.input),
            "output_usd_per_million": str(entry.rates.output),
        },
        "efforts": efforts,
        "allocation": {
            "objective": "sum log(1 + usable) per cost**cost_exponent",
            "cost_exponent": cost_exponent,
            "min_requests_per_category": min_requests_per_category,
            "max_share_per_category": max_share_per_category,
            "fill_fraction": fill_fraction,
            "max_censor_rate": max_censor_rate,
        },
        "budget_usd": str(budget_usd),
        "planned_requests": max_requests,
        "base_requests": int(realized_plan.report["base_cells"]),
        "allocated_before_design_balancing": base_requests,
        "repetitions": {"fraction": repetition_fraction, "count": repetition_count},
        "expected_cost_usd": round(expected_cost, 4),
        "worst_case_cost_usd_uncapped": round(worst_cost, 4),
        "worst_case_basis": "collector reservation (max_cost_estimate) for every realized request",
        "hard_cap_note": (
            "The collector reserves the worst case per request and stops before the "
            f"estimated spend could exceed ${budget_usd}; provider billing can differ."
        ),
        "expected_new_usable_rows": round(expected_usable, 1),
        "expected_cost_per_usable_usd": round(expected_cost / expected_usable, 5)
        if expected_usable
        else None,
        "categories": {name: asdict(plan) for name, plan in sorted(plans.items())},
        "store_observations_used": int(len(observed)),
        "approval_required": (
            "Paid API calls. Requires explicit approval and a manual workflow dispatch with "
            "approve_paid_call=true. No request has been sent."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    return {"config": config, "report": report}


def render_plan_markdown(report: dict[str, Any], readiness: dict[str, Any] | None = None) -> str:
    lines = [
        f"# Dry-run plan: `{report['experiment_id']}`",
        "",
        "**No requests have been sent.** " + report["approval_required"],
        "",
        f"- Target: `{report['provider']}/{report['model']}` "
        f"(pricing {report['pricing']['version']}: "
        f"${report['pricing']['input_usd_per_million']} in / "
        f"${report['pricing']['output_usd_per_million']} out per 1M tokens)",
        f"- Efforts (balanced across prompts): {', '.join(str(e) for e in report['efforts'])}",
        f"- Planned requests: **{report['planned_requests']}** ({report['base_requests']} base "
        f"+ repetitions of {report['repetitions']['fraction']:.0%} of cells x"
        f"{report['repetitions']['count']})",
        f"- Expected estimated cost: **${report['expected_cost_usd']:.2f}** "
        f"(hard cap ${report['budget_usd']}; uncapped worst case "
        f"${report['worst_case_cost_usd_uncapped']:.2f})",
        f"- Expected new usable (uncensored) rows: **{report['expected_new_usable_rows']:.0f}** "
        f"(~${report['expected_cost_per_usable_usd'] or 0:.4f} per usable row)",
        f"- Store observations used for the length models: {report['store_observations_used']}",
        "",
        f"_{report['hard_cap_note']}_",
        "",
        "| Category | Observed rows / usable / censor | Model median (src) | Cap "
        "| Pred. censor | Requests | Exp. usable | Exp. $/req | Why |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, plan in report["categories"].items():
        observed = (
            f"{plan['observed_rows']} / {plan['observed_usable']} / "
            f"{plan['observed_censor_rate']:.0%}"
            if plan["observed_censor_rate"] is not None
            else "0 / 0 / -"
        )
        flag = " (top of ladder)" if plan["cap_at_top_of_ladder"] else ""
        lines.append(
            f"| {name} | {observed} | {plan['model']['median']:.0f} "
            f"({'prior' if plan['model']['source'] == 'prior' else 'fit'}) | "
            f"{plan['cap']}{flag} | {plan['predicted_censor_rate']:.0%} | "
            f"{plan['allocated_requests']} | {plan['expected_new_usable']:.1f} | "
            f"{plan['expected_cost_usd']:.4f} | {'; '.join(plan['reasons']) or '-'} |"
        )
    if readiness and "split_simulation" in readiness:
        sim = readiness["split_simulation"]["real_label"]
        current = readiness["totals"]["usable_rows"]
        projected = current + report["expected_new_usable_rows"]
        lines += [
            "",
            "## Readiness projection",
            "",
            f"Usable rows now: {current}; expected after this batch: ~{projected:.0f}. "
            f"`real_label` split ratios need about {sim['implied_minimum_usable']} usable rows "
            "(more if groups are uneven).",
        ]
    lines += [
        "",
        "Model notes: per-category log-normal output length, right-censored at the cap "
        "(truncated responses are lower bounds), MAP with weak priors; effort multipliers are "
        "planning assumptions until effort levels are observed.",
        "",
    ]
    return "\n".join(lines)
