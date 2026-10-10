"""Resumable, budget-enforced collection orchestrator.

Pipeline per sample::

    validate request -> preflight token count -> cost reservation -> log dispatch
    -> execute -> log response / classified failure -> (retry with backoff)

Modes:

* ``dry_run`` (default): plan + offline preflight + cost estimates; nothing is sent.
* ``offline``: executes mock providers only. Any live provider in the plan is an error.
* ``live``: requires ``execute_live=True`` (CLI ``--execute-live``) **and** a cost cap.

Idempotency: ``sample_id`` is a deterministic UUID of (experiment, prompt, config,
repetition). Completed samples are never re-run; samples dispatched without a recorded
outcome become ``uncertain`` and are not re-sent unless ``retry_uncertain`` is set.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from seshat.data.collectors.control import Budget, RateLimiter, SharedBudgetLedger
from seshat.data.collectors.events import EventLog, EventType, SampleState, SampleStatus, replay
from seshat.data.io import write_json, write_jsonl
from seshat.data.pricing import PriceTable, compute_cost, max_cost_estimate
from seshat.data.storage import StorageLayout
from seshat.experiments.config import CollectionConfig
from seshat.experiments.design import Plan, PlannedSample
from seshat.providers.base import (
    ErrorCategory,
    PreflightResult,
    ProviderAdapter,
    ProviderError,
    approximate_preflight,
    unavailable_preflight,
)
from seshat.providers.catalog import load_catalog
from seshat.providers.registry import adapter_class, ensure_live_calls_authorized, live_providers

Mode = Literal["dry_run", "offline", "live"]
log = logging.getLogger("seshat.collector")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "at": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        return json.dumps(payload, default=str)


def _attach_run_log(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(_JsonFormatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return handler


def _info(message: str, **fields: Any) -> None:
    log.info(message, extra={"fields": fields})


@dataclass
class CollectionResult:
    run_id: str
    mode: Mode
    experiment_dir: Path
    planned: int
    status_counts: dict[str, int]
    dispatched_this_run: int
    spent_or_reserved_usd: Decimal
    estimated_max_cost_usd: Decimal | None = None
    stopped_reason: str | None = None
    provider_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "experiment_dir": str(self.experiment_dir),
            "planned": self.planned,
            "status_counts": self.status_counts,
            "dispatched_this_run": self.dispatched_this_run,
            "spent_or_reserved_usd": str(self.spent_or_reserved_usd),
            "estimated_max_cost_usd": None
            if self.estimated_max_cost_usd is None
            else str(self.estimated_max_cost_usd),
            "stopped_reason": self.stopped_reason,
            "provider_stats": self.provider_stats,
        }


class Collector:
    def __init__(
        self,
        config: CollectionConfig,
        plan: Plan,
        *,
        mode: Mode = "dry_run",
        execute_live: bool = False,
        max_cost_usd: Decimal | None = None,
        retry_failed: bool = False,
        retry_uncertain: bool = False,
        adapter_factory: Callable[[str], ProviderAdapter] | None = None,
        price_table: PriceTable | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self.config = config
        self.plan = plan
        self.mode = mode
        self.retry_failed = retry_failed
        self.retry_uncertain = retry_uncertain
        self.layout = StorageLayout(config.storage.root)
        self.exp_dir = self.layout.experiment_dir(config.experiment_id)
        self.events = EventLog(self.exp_dir / "events.jsonl")
        self.prices = price_table or PriceTable.load()
        self._adapter_factory = adapter_factory or (lambda name: adapter_class(name)())
        self._adapters: dict[str, ProviderAdapter] = {}
        self._sleep = sleep or asyncio.sleep
        self.run_id = str(uuid.uuid4())
        if config.privacy.store_response_text:
            raise ValueError(
                "store_response_text is not supported: adapters never retain generated text"
            )
        providers = {s.provider for s in plan.samples}
        live = live_providers(providers)
        if mode == "offline" and live:
            raise ValueError(f"offline mode cannot run live providers {live}; use --execute-live")
        if mode == "live":
            ensure_live_calls_authorized(providers, execute_live)
            if max_cost_usd is None:
                raise ValueError("live collection requires --max-cost-usd")
            for provider in live:
                try:
                    self.adapter(provider)
                except Exception as exc:  # noqa: BLE001 - surface setup problems before any dispatch
                    raise ValueError(
                        f"cannot initialize live provider {provider!r} "
                        f"({type(exc).__name__}: {exc}). Install SDKs with "
                        "pip install -e '.[providers]' and set the provider API key."
                    ) from exc
        cap = max_cost_usd if max_cost_usd is not None else config.budget.max_cost_usd
        if max_cost_usd is not None and config.budget.max_cost_usd is not None:
            cap = min(max_cost_usd, config.budget.max_cost_usd)
        shared_ledger = None
        ledger_path = os.environ.get("SESHAT_SHARED_LEDGER_PATH")
        if mode == "live" and os.environ.get("SESHAT_UNATTENDED_COLLECTION") == "1":
            if not ledger_path:
                raise ValueError(
                    "Unattended live collection requires durable SESHAT_SHARED_LEDGER_PATH"
                )
        if mode == "live" and ledger_path:
            day_cap = os.environ.get("SESHAT_DAILY_COST_CAP_USD")
            month_cap = os.environ.get("SESHAT_MONTHLY_COST_CAP_USD")
            if not day_cap or not month_cap:
                raise ValueError("Shared ledger requires daily and monthly cost caps")
            shared_ledger = SharedBudgetLedger(
                ledger_path,
                daily_cap_usd=Decimal(day_cap),
                monthly_cap_usd=Decimal(month_cap),
            )
        self.budget = Budget(
            max_requests=config.budget.max_requests,
            max_cost_usd=cap,
            shared_ledger=shared_ledger,
        )
        # Live providers default to 60 requests/minute; mocks are unthrottled unless configured.
        self._limiters = {
            provider: RateLimiter(config.execution.rate_limits[provider].requests_per_minute)
            if provider in config.execution.rate_limits
            else RateLimiter(_default_limit().requests_per_minute)
            for provider in providers
            if provider in config.execution.rate_limits
            or adapter_class(provider).makes_external_calls
        }
        self._global = asyncio.Semaphore(config.execution.concurrency)
        self._per_provider = {
            p: asyncio.Semaphore(config.execution.per_provider_concurrency) for p in providers
        }
        self._stop_reason: str | None = None
        self._dispatched = 0
        self._stats: dict[str, Counter[str]] = {}

    def adapter(self, provider: str) -> ProviderAdapter:
        if provider not in self._adapters:
            self._adapters[provider] = self._adapter_factory(provider)
        return self._adapters[provider]

    # -- preflight & cost -------------------------------------------------------
    def _preflight(self, sample: PlannedSample) -> PreflightResult:
        method = self.config.execution.preflight
        if method == "none":
            return unavailable_preflight(sample.provider, "preflight disabled by configuration")
        offline_only = (
            self.mode == "dry_run" and adapter_class(sample.provider).makes_external_calls
        )
        if method == "approximate" or offline_only:
            return approximate_preflight(sample.request, sample.provider)
        return self.adapter(sample.provider).count_input_tokens(sample.request)

    def _estimate(self, sample: PlannedSample, preflight: PreflightResult) -> Decimal | None:
        entry = self.prices.lookup(sample.provider, sample.model, date.today())
        if entry is None or sample.request.params.max_output_tokens is None:
            return None
        tokens = preflight.input_tokens
        if tokens is None:
            tokens = approximate_preflight(sample.request, sample.provider).input_tokens or 0
        # Approximate counts can undershoot; pad them by 25% for the reservation.
        if preflight.method != "provider_native":
            tokens = int(tokens * 1.25) + 16
        return max_cost_estimate(
            adapter_class(sample.provider).family,
            entry,
            input_tokens=tokens,
            max_output_tokens=sample.request.params.max_output_tokens,
            cache_ttl=sample.request.params.cache.ttl
            if sample.request.params.cache.enabled
            else None,
        )

    def _actual_cost(self, sample: PlannedSample, response: dict[str, Any]) -> Decimal | None:
        from seshat.providers.base import ProviderResponse

        family = adapter_class(sample.provider).family
        entry = self.prices.lookup(sample.provider, sample.model, date.today())
        usage = self.adapter(sample.provider).normalize_usage(
            ProviderResponse.model_validate(response)
        )
        return compute_cost(family, entry, usage).calculated_cost_usd

    # -- main loop ----------------------------------------------------------------
    def dry_run(self) -> CollectionResult:
        rows: list[dict[str, Any]] = []
        total = Decimal(0)
        unbounded = 0
        for sample in self.plan.samples:
            pre = self._preflight(sample)
            estimate = self._estimate(sample, pre)
            if estimate is None:
                unbounded += 1
            else:
                total += estimate
            rows.append(
                {
                    "sample_id": str(sample.sample_id),
                    "provider": sample.provider,
                    "model": sample.model,
                    "prompt_id": sample.prompt_id,
                    "preflight_input_tokens": pre.input_tokens,
                    "preflight_method": pre.method.value,
                    "max_cost_estimate_usd": None if estimate is None else str(estimate),
                }
            )
        out = self.exp_dir / "dry_run"
        write_jsonl(out / "estimates.jsonl", rows)
        report = {
            "experiment_id": self.config.experiment_id,
            "planned_requests": len(self.plan.samples),
            "live_providers": live_providers({s.provider for s in self.plan.samples}),
            "estimated_max_cost_usd": str(total),
            "samples_without_price": unbounded,
            "note": (
                "Upper-bound estimate: assumes max_output_tokens are generated for every request."
            ),
        }
        write_json(out / "report.json", report)
        return CollectionResult(
            run_id=self.run_id,
            mode="dry_run",
            experiment_dir=self.exp_dir,
            planned=len(self.plan.samples),
            status_counts={"dry_run": len(self.plan.samples)},
            dispatched_this_run=0,
            spent_or_reserved_usd=Decimal(0),
            estimated_max_cost_usd=total,
        )

    def run(self) -> CollectionResult:
        if self.mode == "dry_run":
            return self.dry_run()
        handler = _attach_run_log(self.exp_dir / "runs" / f"{self.run_id}.log.jsonl")
        try:
            return asyncio.run(self._run())
        finally:
            log.removeHandler(handler)
            handler.close()

    async def _run(self) -> CollectionResult:
        prior_events = self.events.read()
        design_hash = self.config.design_hash()
        previous = {
            e.get("design_hash")
            for e in prior_events
            if e.get("event") == EventType.RUN_STARTED.value and e.get("design_hash")
        }
        if previous - {design_hash}:
            raise ValueError(
                f"experiment {self.config.experiment_id!r} was collected with a different "
                "design (prompts/targets/sampling/repetitions/strata/seed); use a new "
                "experiment_id instead of mixing designs in one experiment"
            )
        write_jsonl(
            self.exp_dir / "plan.jsonl", [s.model_dump(mode="json") for s in self.plan.samples]
        )
        write_json(self.exp_dir / "design_report.json", self.plan.report)
        states = replay(prior_events)
        for state in states.values():
            if state.open_attempt is not None and state.status == SampleStatus.PENDING:
                self.events.append(
                    EventType.UNCERTAIN,
                    sample_id=state.sample_id,
                    run_id=self.run_id,
                    attempt=state.open_attempt,
                    reason="dispatched_without_recorded_outcome",
                )
                state.status = SampleStatus.UNCERTAIN
                state.open_attempt = None
        for state in states.values():
            self.budget.dispatched += state.attempts
            if state.status == SampleStatus.COMPLETED and state.actual_cost_usd is not None:
                self.budget.committed_usd += Decimal(str(state.actual_cost_usd))
            elif state.status == SampleStatus.UNCERTAIN:
                self.budget.committed_usd += Decimal(str(state.reserved_cost_usd))
        self.events.append(
            EventType.RUN_STARTED,
            run_id=self.run_id,
            mode=self.mode,
            config_hash=self.plan.config_hash,
            design_hash=design_hash,
            planned=len(self.plan.samples),
            max_cost_usd=None
            if self.budget.max_cost_usd is None
            else str(self.budget.max_cost_usd),
        )
        _info("run started", run_id=self.run_id, mode=self.mode, planned=len(self.plan.samples))
        todo: list[tuple[PlannedSample, SampleState | None]] = []
        for sample in self.plan.samples:
            prior = states.get(str(sample.sample_id))
            status = prior.status if prior else SampleStatus.PENDING
            if status == SampleStatus.COMPLETED:
                continue
            if status == SampleStatus.UNCERTAIN and not self.retry_uncertain:
                continue
            if status == SampleStatus.FAILED and not self.retry_failed:
                continue
            todo.append((sample, prior))
        await asyncio.gather(*(self._collect(sample, state) for sample, state in todo))
        final = replay(self.events.read())
        counts = Counter(
            final[str(s.sample_id)].status.value if str(s.sample_id) in final else "pending"
            for s in self.plan.samples
        )
        result = CollectionResult(
            run_id=self.run_id,
            mode=self.mode,
            experiment_dir=self.exp_dir,
            planned=len(self.plan.samples),
            status_counts=dict(sorted(counts.items())),
            dispatched_this_run=self._dispatched,
            spent_or_reserved_usd=self.budget.spent_or_reserved_usd,
            stopped_reason=self._stop_reason,
            provider_stats={p: dict(c) for p, c in sorted(self._stats.items())},
        )
        self.events.append(EventType.RUN_FINISHED, run_id=self.run_id, summary=result.to_dict())
        _info("run finished", **result.to_dict())
        return result

    async def _collect(self, sample: PlannedSample, state: SampleState | None) -> None:
        sid = str(sample.sample_id)
        stats = self._stats.setdefault(sample.provider, Counter())
        async with self._global, self._per_provider[sample.provider]:
            if self._stop_reason:
                self.events.append(
                    EventType.SKIPPED, sample_id=sid, run_id=self.run_id, reason=self._stop_reason
                )
                stats["skipped"] += 1
                return
            caps = load_catalog().capabilities(sample.provider, sample.model)
            problems = caps.unsupported_settings(sample.request)
            if problems:
                self.events.append(
                    EventType.SKIPPED,
                    sample_id=sid,
                    run_id=self.run_id,
                    reason="unsupported:" + "; ".join(problems),
                )
                stats["skipped"] += 1
                return
            preflight_dict = state.preflight if state is not None and state.preflight else None
            if preflight_dict is None:
                preflight = await asyncio.to_thread(self._preflight, sample)
                preflight_dict = preflight.model_dump(mode="json")
                self.events.append(
                    EventType.PREFLIGHT, sample_id=sid, run_id=self.run_id, result=preflight_dict
                )
            preflight = PreflightResult.model_validate(preflight_dict)
            attempt = state.attempts if state is not None else 0
            failures_this_sample = 0
            while True:
                if self._stop_reason:
                    self.events.append(
                        EventType.SKIPPED,
                        sample_id=sid,
                        run_id=self.run_id,
                        reason=self._stop_reason,
                    )
                    stats["skipped"] += 1
                    return
                estimate = self._estimate(sample, preflight)
                if estimate is None and adapter_class(sample.provider).makes_external_calls:
                    self.events.append(
                        EventType.SKIPPED,
                        sample_id=sid,
                        run_id=self.run_id,
                        reason="no_price_for_model",
                    )
                    stats["skipped"] += 1
                    return
                refusal = self.budget.try_reserve(estimate)
                if refusal:
                    self._stop_reason = self._stop_reason or refusal
                    self.events.append(
                        EventType.SKIPPED, sample_id=sid, run_id=self.run_id, reason=refusal
                    )
                    stats["skipped"] += 1
                    return
                limiter = self._limiters.get(sample.provider)
                if limiter is not None:
                    await limiter.acquire()
                attempt += 1
                self.events.append(
                    EventType.DISPATCHED,
                    sample_id=sid,
                    request_id=sample.request_id,
                    run_id=self.run_id,
                    attempt=attempt,
                    provider=sample.provider,
                    model=sample.model,
                    reserved_cost_usd=None if estimate is None else str(estimate),
                )
                self._dispatched += 1
                stats["dispatched"] += 1
                try:
                    response = await asyncio.to_thread(
                        self.adapter(sample.provider).execute_request,
                        sample.request,
                        timeout_seconds=self.config.execution.timeout_seconds,
                    )
                except ProviderError as error:
                    failures_this_sample += 1
                    # A retryable transport failure can still have been billed.
                    # Uncertain outcomes always win over retryability.
                    retry = (
                        error.retryable
                        and not error.outcome_uncertain
                        and failures_this_sample <= self.config.execution.max_retries
                    )
                    final = not retry
                    if error.outcome_uncertain:
                        self.budget.settle(estimate, None)
                    else:
                        self.budget.release(estimate)
                    self.events.append(
                        EventType.FAILED,
                        sample_id=sid,
                        run_id=self.run_id,
                        attempt=attempt,
                        category=error.category.value,
                        status_code=error.status_code,
                        code=error.code,
                        message=str(error)[:300],
                        retryable=error.retryable,
                        outcome_uncertain=error.outcome_uncertain,
                        final=final,
                    )
                    stats[f"error_{error.category.value}"] += 1
                    if error.fatal:
                        self._stop_reason = f"fatal_{error.category.value}"
                        return
                    if final:
                        return
                    delay = error.retry_after_seconds or min(
                        self.config.execution.backoff_max_seconds,
                        self.config.execution.backoff_base_seconds
                        * 2 ** (failures_this_sample - 1),
                    ) * (1 + random.random() * 0.25)
                    await self._sleep(delay)
                    continue
                except Exception as error:  # noqa: BLE001 - unknown => uncertain, never repeated
                    self.budget.settle(estimate, None)
                    self.events.append(
                        EventType.FAILED,
                        sample_id=sid,
                        run_id=self.run_id,
                        attempt=attempt,
                        category=ErrorCategory.UNKNOWN.value,
                        message=f"{type(error).__name__}: {error}"[:300],
                        retryable=False,
                        outcome_uncertain=True,
                        final=True,
                    )
                    stats["error_unknown"] += 1
                    return
                payload = response.model_dump(mode="json")
                cost = self._actual_cost(sample, payload)
                self.budget.settle(estimate, cost)
                self.events.append(
                    EventType.RESPONSE,
                    sample_id=sid,
                    run_id=self.run_id,
                    attempt=attempt,
                    response=payload,
                    calculated_cost_usd=None if cost is None else str(cost),
                )
                stats["completed"] += 1
                return


def _default_limit() -> Any:
    from seshat.experiments.config import RateLimitConfig

    return RateLimitConfig()
