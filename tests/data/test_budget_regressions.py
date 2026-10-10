"""Regression tests for paid-request accounting and uncertain outcomes."""

from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

from seshat.data.collectors.collector import Collector
from seshat.data.collectors.control import Budget, SharedBudgetLedger
from seshat.data.collectors.events import EventLog, EventType
from seshat.data.generators import generate_prompts
from seshat.experiments.config import CollectionConfig
from seshat.experiments.design import build_plan
from seshat.providers.base import ErrorCategory, ProviderError
from seshat.providers.catalog import load_catalog
from seshat.providers.mock import MockOpenAIAdapter


def test_failed_attempts_still_consume_request_quota() -> None:
    budget = Budget(max_requests=2, max_cost_usd=Decimal("1"))
    assert budget.try_reserve(Decimal("0.10")) is None
    budget.release(Decimal("0.10"))
    assert budget.dispatched == 1
    assert budget.try_reserve(Decimal("0.10")) is None
    budget.release(Decimal("0.10"))
    assert budget.try_reserve(Decimal("0.10")) == "max_requests_reached"


def test_retryable_but_uncertain_error_is_not_retried(
    small_config: Callable[..., CollectionConfig], tmp_path: Path
) -> None:
    config = small_config(execution={"concurrency": 1, "per_provider_concurrency": 1})
    plan = build_plan(config, generate_prompts(17, config.seed), load_catalog())
    plan.samples[:] = [s for s in plan.samples if s.provider == "mock-openai"][:1]
    assert plan.samples
    calls = 0

    class RetryableUncertain(ProviderError):
        @property
        def retryable(self) -> bool:
            return True

    class Doubtful(MockOpenAIAdapter):
        def execute_request(self, request, *, timeout_seconds):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            raise RetryableUncertain(
                ErrorCategory.TIMEOUT, "provider might have completed this request"
            )

    result = Collector(
        config, plan, mode="offline", adapter_factory=lambda _: Doubtful()
    ).run()
    assert calls == 1
    assert result.dispatched_this_run == 1
    assert result.status_counts == {"uncertain": 1}
    events = EventLog(
        Path(config.storage.root) / "raw" / config.experiment_id / "events.jsonl"
    ).read()
    assert sum(e["event"] == EventType.DISPATCHED for e in events) == 1


def test_shared_caps_apply_across_independent_experiments(tmp_path: Path) -> None:
    ledger_file = tmp_path / "persistent" / "spend.sqlite3"
    first = Budget(
        max_requests=10, max_cost_usd=Decimal("10"),
        shared_ledger=SharedBudgetLedger(
            ledger_file, daily_cap_usd=Decimal("0.20"),
            monthly_cap_usd=Decimal("0.30"),
        ),
    )
    second = Budget(
        max_requests=10, max_cost_usd=Decimal("10"),
        shared_ledger=SharedBudgetLedger(
            ledger_file, daily_cap_usd=Decimal("0.20"),
            monthly_cap_usd=Decimal("0.30"),
        ),
    )
    assert first.try_reserve(Decimal("0.15")) is None
    assert second.try_reserve(Decimal("0.10")) == "shared_cost_cap_reached"
    first.settle(Decimal("0.15"), Decimal("0.15"))
    assert second.try_reserve(Decimal("0.05")) is None
    second.release(Decimal("0.05"))
    assert second.try_reserve(Decimal("0.05")) is None
