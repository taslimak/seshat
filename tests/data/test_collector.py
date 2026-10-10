from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

from seshat.data.collectors.collector import Collector
from seshat.data.collectors.events import EventLog, EventType, SampleStatus, replay
from seshat.data.collectors.normalize import build_records, normalize_experiment
from seshat.data.generators import generate_prompts
from seshat.data.pricing import PriceTable
from seshat.data.privacy import RetentionPolicy
from seshat.data.schemas import LabelQuality, ResponseStatus, SourceType
from seshat.data.storage import StorageLayout
from seshat.experiments.config import CollectionConfig
from seshat.experiments.design import Plan, build_plan
from seshat.providers.base import ErrorCategory, ProviderAdapter
from seshat.providers.catalog import load_catalog
from seshat.providers.mock import FailurePlan, MockAdapter, MockOpenAIAdapter
from seshat.providers.registry import ProviderCallsNotAllowedError, adapter_class


def _plan(config: CollectionConfig, count: int = 34) -> Plan:
    return build_plan(config, generate_prompts(count, config.seed), load_catalog())


def _events(config: CollectionConfig) -> list[dict[str, object]]:
    return EventLog(
        Path(config.storage.root) / "raw" / config.experiment_id / "events.jsonl"
    ).read()


def _factory(failures: dict[str, FailurePlan] | None = None) -> Callable[[str], ProviderAdapter]:
    adapters: dict[str, ProviderAdapter] = {}

    def make(provider: str) -> ProviderAdapter:
        if provider not in adapters:
            cls = adapter_class(provider)
            if not issubclass(cls, MockAdapter):
                cls = MockOpenAIAdapter  # stand-in for a live provider (no network in tests)
            adapters[provider] = cls((failures or {}).get(provider))
        return adapters[provider]

    return make


def test_offline_run_completes_and_resume_never_reexecutes(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config()
    plan = _plan(config)
    first = Collector(config, plan, mode="offline").run()
    assert first.status_counts == {"completed": len(plan.samples)}
    assert first.dispatched_this_run == len(plan.samples)
    second = Collector(config, plan, mode="offline").run()
    assert second.dispatched_this_run == 0
    dispatches = [e for e in _events(config) if e["event"] == EventType.DISPATCHED]
    assert len(dispatches) == len(plan.samples)


def test_resume_refuses_a_changed_design_but_allows_budget_changes(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config()
    Collector(config, _plan(config), mode="offline").run()
    rebudgeted = config.model_copy(
        update={"budget": config.budget.model_copy(update={"max_requests": 1000})}
    )
    assert Collector(rebudgeted, _plan(rebudgeted), mode="offline").run().dispatched_this_run == 0
    redesigned = config.model_copy(update={"seed": config.seed + 1})
    with pytest.raises(ValueError, match="different design"):
        Collector(redesigned, _plan(redesigned), mode="offline").run()


def test_dispatch_without_outcome_becomes_uncertain_and_is_not_resent(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config()
    plan = _plan(config)
    crashed = plan.samples[0]
    log = EventLog(Path(config.storage.root) / "raw" / config.experiment_id / "events.jsonl")
    log.append(
        EventType.DISPATCHED,
        sample_id=str(crashed.sample_id),
        run_id="crashed-run",
        attempt=1,
        reserved_cost_usd="0.01",
    )
    result = Collector(config, plan, mode="offline").run()
    assert result.status_counts["uncertain"] == 1
    sent = [
        e
        for e in _events(config)
        if e["event"] == EventType.DISPATCHED and e["sample_id"] == str(crashed.sample_id)
    ]
    assert len(sent) == 1  # only the pre-crash dispatch
    retried = Collector(config, plan, mode="offline", retry_uncertain=True).run()
    assert retried.dispatched_this_run == 1


def test_retryable_errors_back_off_and_succeed(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config(execution={"concurrency": 1, "per_provider_concurrency": 1})
    plan = _plan(config, 17)
    plan.samples[:] = [s for s in plan.samples if s.provider == "mock-openai"][:3]
    failures = {
        "mock-openai": FailurePlan(
            by_call={1: ErrorCategory.RATE_LIMIT, 2: ErrorCategory.OVERLOADED}
        )
    }
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = Collector(
        config, plan, mode="offline", adapter_factory=_factory(failures), sleep=fake_sleep
    ).run()
    assert result.status_counts == {"completed": 3}
    assert len(sleeps) == 2
    states = replay(_events(config))
    first = states[str(plan.samples[0].sample_id)]
    assert first.attempts == 3 and [f["category"] for f in first.failures] == [
        "rate_limit",
        "overloaded",
    ]
    requests, _, observations = build_records(
        plan.samples,
        _events(config),
        catalog=load_catalog(),
        prices=PriceTable.load(),
        policy=RetentionPolicy(),
        storage_root=Path(config.storage.root),
    )
    obs = {o.sample_id: o for o in observations}[plan.samples[0].sample_id]
    assert obs.retry_count == 2 and obs.rate_limit_encountered


def test_retries_are_bounded_and_permanent_errors_are_not_retried(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config(
        execution={"concurrency": 1, "per_provider_concurrency": 1, "max_retries": 1}
    )
    plan = _plan(config, 17)
    plan.samples[:] = [s for s in plan.samples if s.provider == "mock-openai"][:2]
    failures = {
        "mock-openai": FailurePlan(
            by_call={
                1: ErrorCategory.RATE_LIMIT,
                2: ErrorCategory.RATE_LIMIT,
                3: ErrorCategory.INVALID_REQUEST,
            }
        )
    }
    result = Collector(config, plan, mode="offline", adapter_factory=_factory(failures)).run()
    assert result.status_counts == {"failed": 2}
    _, _, observations = build_records(
        plan.samples,
        _events(config),
        catalog=load_catalog(),
        prices=PriceTable.load(),
        policy=RetentionPolicy(),
        storage_root=Path(config.storage.root),
    )
    assert {o.response_status for o in observations} == {ResponseStatus.FAILED}
    assert {o.label_quality for o in observations} == {LabelQuality.UNAVAILABLE}
    assert all(o.actual_output_tokens is None for o in observations)


def test_unknown_exceptions_are_uncertain(small_config: Callable[..., CollectionConfig]) -> None:
    config = small_config(execution={"concurrency": 1, "per_provider_concurrency": 1})
    plan = _plan(config, 17)
    plan.samples[:] = plan.samples[:2]

    class Exploding(MockOpenAIAdapter):
        def execute_request(self, request, *, timeout_seconds):  # type: ignore[no-untyped-def]
            raise RuntimeError("socket closed mid-response")

    result = Collector(config, plan, mode="offline", adapter_factory=lambda _: Exploding()).run()
    assert result.status_counts == {"uncertain": 2}


def test_fatal_authentication_error_stops_the_run(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config(execution={"concurrency": 1, "per_provider_concurrency": 1})
    plan = _plan(config, 17)
    failures = {
        p: FailurePlan(by_call={1: ErrorCategory.AUTHENTICATION})
        for p in ("mock-openai", "mock-anthropic", "mock-gemini")
    }
    result = Collector(config, plan, mode="offline", adapter_factory=_factory(failures)).run()
    assert result.stopped_reason == "fatal_authentication"
    assert result.dispatched_this_run == 1


def test_max_requests_budget_is_cumulative(small_config: Callable[..., CollectionConfig]) -> None:
    config = small_config(budget={"max_requests": 150})
    plan = _plan(config)
    tight = config.model_copy(
        update={"budget": config.budget.model_copy(update={"max_requests": 10})}
    )
    first = Collector(tight, plan, mode="offline").run()
    assert first.dispatched_this_run == 10 and first.stopped_reason == "max_requests_reached"
    again = Collector(tight, plan, mode="offline").run()
    assert again.dispatched_this_run == 0


def test_live_mode_requires_authorization_budget_and_enforces_cost_cap(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config(
        targets=[
            {
                "provider": "openai",
                "model": "gpt-6-luna",
                "efforts": ["low"],
                "max_output_tokens": [1000],
            }
        ]
    )
    plan = _plan(config, 17)
    with pytest.raises(ProviderCallsNotAllowedError):
        Collector(config, plan, mode="live", execute_live=False, max_cost_usd=Decimal(1))
    with pytest.raises(ValueError, match="max-cost-usd"):
        Collector(config, plan, mode="live", execute_live=True)
    with pytest.raises(ValueError, match="offline mode cannot run live"):
        Collector(config, plan, mode="offline")
    cap = Decimal("0.002")  # ~3 worst-case reservations at gpt-6-luna list prices
    result = Collector(
        config, plan, mode="live", execute_live=True, max_cost_usd=cap, adapter_factory=_factory()
    ).run()
    assert result.stopped_reason == "max_cost_reached"
    assert 0 < result.dispatched_this_run < len(plan.samples)
    assert result.spent_or_reserved_usd <= cap


def test_dry_run_sends_nothing_and_estimates_cost(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config(
        targets=[
            {
                "provider": "anthropic",
                "model": "claude-haiku-5-5",
                "efforts": ["low"],
                "max_output_tokens": [500],
            }
        ]
    )
    plan = _plan(config, 17)
    result = Collector(config, plan, mode="dry_run").run()
    assert (
        result.dispatched_this_run == 0
        and result.estimated_max_cost_usd
        and result.estimated_max_cost_usd > 0
    )
    assert _events(config) == []
    report = Path(config.storage.root) / "raw" / config.experiment_id / "dry_run" / "report.json"
    assert report.exists()


def test_normalization_labels_fixtures_and_is_reproducible(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config()
    plan = _plan(config)
    Collector(config, plan, mode="offline").run()
    kwargs = {"catalog": load_catalog(), "prices": PriceTable.load(), "policy": RetentionPolicy()}
    counts = normalize_experiment(config.storage.root, config.experiment_id, **kwargs)
    assert counts == {
        "requests": len(plan.samples),
        "preflight": len(plan.samples),
        "observations": len(plan.samples),
    }
    layout = StorageLayout(config.storage.root)
    before = layout.read_frame("observations").sort_values("observation_id").reset_index(drop=True)
    normalize_experiment(config.storage.root, config.experiment_id, **kwargs)
    after = layout.read_frame("observations").sort_values("observation_id").reset_index(drop=True)
    assert len(after) == len(before)
    assert before[["observation_id", "actual_output_tokens"]].equals(
        after[["observation_id", "actual_output_tokens"]]
    )
    assert set(after["source_type"]) == {SourceType.TEST_FIXTURE.value}
    assert set(after["label_quality"]) == {LabelQuality.SYNTHETIC.value}
    assert after["calculated_cost_usd"].notna().all()
    requests = layout.read_frame("requests")
    assert (
        requests["prompt_hash"].str.len().min() == 64
        and requests["full_prompt_storage_ref"].isna().all()
    )


def test_prompt_text_is_stored_only_when_enabled(
    small_config: Callable[..., CollectionConfig],
) -> None:
    config = small_config(privacy={"store_prompt_text": True})
    plan = _plan(config, 17)
    plan.samples[:] = plan.samples[:2]
    Collector(config, plan, mode="offline").run()
    normalize_experiment(
        config.storage.root,
        config.experiment_id,
        catalog=load_catalog(),
        prices=PriceTable.load(),
        policy=RetentionPolicy(store_prompt_text=True),
    )
    refs = StorageLayout(config.storage.root).read_frame("requests")["full_prompt_storage_ref"]
    assert refs.notna().all()
    assert (Path(config.storage.root) / refs.iloc[0]).exists()


def test_event_log_tolerates_truncated_last_line(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append(EventType.DISPATCHED, sample_id="s1", run_id="r", attempt=1)
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "response_rec')  # crash mid-write
    states = replay(log.read())
    assert states["s1"].open_attempt == 1 and states["s1"].status == SampleStatus.PENDING
