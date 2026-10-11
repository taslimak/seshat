"""Versioned price tables and cost calculation.

Prices live in ``configs/pricing/*.json`` (USD per 1M tokens) with an effective date
range, source URL and retrieval date, so historical costs stay reproducible. Costs
computed here are **price-table estimates** (``cost_method=price_table_estimate``),
never invoice charges.

Billing rules per family (applied to normalized usage; see providers.normalization):

* OpenAI: uncached input x input + cached x cache_read + output (incl. reasoning) x output.
* Anthropic: uncached input x input + cache reads x cache_read + cache writes x the
  5m or 1h write rate (from the usage breakdown) + output (incl. thinking) x output.
* Gemini: (prompt - cached) x input + cached x cache_read + (candidates + thoughts)
  x output. ``tool_use_prompt_token_count`` is not priced (semantics undocumented)
  and is reported in ``billing_dimensions``.

Long-context tiers switch *all* rates when the documented basis exceeds the threshold.
Batch, regional, priority-tier and cache-storage charges are not modelled; when a
request uses them the record notes it in ``billing_dimensions``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

from seshat.data.io import load_json
from seshat.providers.normalization import NormalizedUsage

DEFAULT_PRICING_DIR = Path("configs") / "pricing"
MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class Rates:
    input: Decimal
    output: Decimal
    cache_read: Decimal | None = None
    cache_write_5m: Decimal | None = None
    cache_write_1h: Decimal | None = None


@dataclass(frozen=True)
class PriceEntry:
    provider: str
    model: str
    pricing_version: str
    effective_from: date
    effective_to: date | None
    rates: Rates
    long_context_threshold: int | None
    long_context_basis: str | None
    long_context_rates: Rates | None
    source_url: str | None
    is_fixture: bool

    def covers(self, on: date) -> bool:
        return self.effective_from <= on and (self.effective_to is None or on <= self.effective_to)


@dataclass(frozen=True)
class CostResult:
    calculated_cost_usd: Decimal | None
    rates: Rates | None
    entry: PriceEntry | None
    dimensions: dict[str, Any] = field(default_factory=dict)


def _dec(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _rates(raw: dict[str, Any]) -> Rates:
    return Rates(
        input=Decimal(str(raw["input"])),
        output=Decimal(str(raw["output"])),
        cache_read=_dec(raw.get("cache_read")),
        cache_write_5m=_dec(raw.get("cache_write_5m")),
        cache_write_1h=_dec(raw.get("cache_write_1h")),
    )


class PriceTable:
    def __init__(self, entries: list[PriceEntry]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, directory: str | Path | None = None) -> PriceTable:
        return _load_table(str(directory or DEFAULT_PRICING_DIR))

    def lookup(self, provider: str, model: str, on: date) -> PriceEntry | None:
        matches = [
            e for e in self.entries if e.provider == provider and e.model == model and e.covers(on)
        ]
        if len(matches) > 1:
            raise ValueError(f"Overlapping price entries for {provider}/{model} on {on}")
        return matches[0] if matches else None


@lru_cache(maxsize=4)
def _load_table(directory: str) -> PriceTable:
    entries: list[PriceEntry] = []
    for path in sorted(Path(directory).glob("*.json")):
        raw = load_json(path)
        fixture = str(raw["pricing_version"]).startswith("fixture")
        for item in raw["entries"]:
            long_context = item.get("long_context")
            entries.append(
                PriceEntry(
                    provider=item["provider"],
                    model=item["model"],
                    pricing_version=str(raw["pricing_version"]),
                    effective_from=date.fromisoformat(item["effective_from"]),
                    effective_to=date.fromisoformat(item["effective_to"])
                    if item.get("effective_to")
                    else None,
                    rates=_rates(item),
                    long_context_threshold=long_context["threshold_input_tokens"]
                    if long_context
                    else None,
                    long_context_basis=long_context["basis"] if long_context else None,
                    long_context_rates=_rates(long_context) if long_context else None,
                    source_url=item.get("source_url"),
                    is_fixture=fixture,
                )
            )
    return PriceTable(entries)


def _select_rates(entry: PriceEntry, basis_tokens: int | None) -> tuple[Rates, bool]:
    if (
        entry.long_context_rates is not None
        and entry.long_context_threshold is not None
        and basis_tokens is not None
        and basis_tokens > entry.long_context_threshold
    ):
        return entry.long_context_rates, True
    return entry.rates, False


def _cost(tokens: int | None, rate: Decimal | None) -> Decimal:
    if not tokens:
        return Decimal(0)
    if rate is None:
        raise ValueError("missing rate for billed tokens")
    return Decimal(tokens) * rate / MILLION


def compute_cost(family: str, entry: PriceEntry | None, usage: NormalizedUsage) -> CostResult:
    """Price-table cost of observed usage, or an empty result if it cannot be computed."""
    if entry is None:
        return CostResult(None, None, None, {"cost_unavailable": "no price entry for model/date"})
    if usage.actual_input_tokens is None or usage.actual_output_tokens is None:
        return CostResult(None, None, entry, {"cost_unavailable": "usage incomplete"})
    dims: dict[str, Any] = {}
    rates, long_context = _select_rates(entry, usage.actual_input_tokens)
    if long_context:
        dims["long_context_tier"] = True
    cached = usage.actual_cached_input_tokens or 0
    try:
        if family == "openai":
            uncached = usage.actual_uncached_input_tokens
            uncached = usage.actual_input_tokens - cached if uncached is None else uncached
            cost = (
                _cost(uncached, rates.input)
                + _cost(cached, rates.cache_read)
                + _cost(usage.actual_output_tokens, rates.output)
            )
        elif family == "anthropic":
            breakdown = usage.billing_dimensions
            writes = usage.actual_cache_creation_tokens or 0
            five = breakdown.get("cache_write_5m_tokens")
            hour = breakdown.get("cache_write_1h_tokens")
            if writes and five is None and hour is None:
                five, hour = writes, 0
                dims["cache_write_ttl_assumed"] = "5m"
            cost = (
                _cost(usage.actual_uncached_input_tokens, rates.input)
                + _cost(cached, rates.cache_read)
                + _cost(five or 0, rates.cache_write_5m)
                + _cost(hour or 0, rates.cache_write_1h)
                + _cost(usage.actual_output_tokens, rates.output)
            )
        elif family == "gemini":
            uncached = usage.actual_input_tokens - cached
            cost = (
                _cost(uncached, rates.input)
                + _cost(cached, rates.cache_read)
                + _cost(usage.actual_output_tokens, rates.output)
            )
            if usage.actual_tool_input_tokens:
                dims["tool_use_prompt_tokens_unpriced"] = usage.actual_tool_input_tokens
        else:
            raise ValueError(f"unknown pricing family {family!r}")
    except ValueError as exc:
        return CostResult(None, rates, entry, {"cost_unavailable": str(exc)})
    return CostResult(cost.quantize(Decimal("0.000000000001")), rates, entry, dims)


# Margin on (possibly approximate) preflight input counts when deciding whether a
# long-context price tier could apply to a request.
LONG_CONTEXT_SAFETY_FACTOR = Decimal("1.25")


def max_cost_estimate(
    family: str,
    entry: PriceEntry,
    *,
    input_tokens: int,
    max_output_tokens: int,
    cache_ttl: str | None = None,
) -> Decimal:
    """Conservative upper bound used for budget reservation before dispatch.

    Assumes every input token is billed at the most expensive input-side rate that
    could apply (uncached, or cache write when caching is enabled) and that the model
    generates ``max_output_tokens``, under the long-context tier if it could apply.

    Every documented long-context basis counts input-side tokens only, so the tier
    can apply only if the input could exceed the threshold. ``input_tokens`` may be
    an approximation, hence the :data:`LONG_CONTEXT_SAFETY_FACTOR` margin.
    """
    candidates = [entry.rates]
    if entry.long_context_rates is not None and (
        entry.long_context_threshold is None
        or input_tokens * LONG_CONTEXT_SAFETY_FACTOR > entry.long_context_threshold
    ):
        candidates.append(entry.long_context_rates)
    worst = Decimal(0)
    for rates in candidates:
        input_rate = rates.input
        if family == "anthropic" and cache_ttl:
            write = rates.cache_write_1h if cache_ttl == "1h" else rates.cache_write_5m
            input_rate = max(input_rate, write or input_rate)
        worst = max(worst, _cost(input_tokens, input_rate) + _cost(max_output_tokens, rates.output))
    return worst.quantize(Decimal("0.000000000001"))
