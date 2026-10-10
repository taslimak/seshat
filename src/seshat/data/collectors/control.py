"""Budget enforcement and per-provider rate limiting."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class Budget:
    """Cumulative limits for an experiment (prior runs included).

    Before dispatch, a conservative maximum cost is *reserved*. After a response the
    reservation is replaced by the computed cost. Uncertain requests keep their
    reservation, because they may have been billed.
    """

    max_requests: int
    max_cost_usd: Decimal | None
    dispatched: int = 0
    committed_usd: Decimal = Decimal(0)
    reserved_usd: Decimal = Decimal(0)

    def try_reserve(self, estimate: Decimal | None) -> str | None:
        """Reserve one request; returns a refusal reason or ``None`` when allowed."""
        if self.dispatched >= self.max_requests:
            return "max_requests_reached"
        if self.max_cost_usd is not None:
            if estimate is None:
                return "cost_unbounded"
            if self.committed_usd + self.reserved_usd + estimate > self.max_cost_usd:
                return "max_cost_reached"
        self.dispatched += 1
        self.reserved_usd += estimate or Decimal(0)
        return None

    def settle(self, reserved: Decimal | None, actual: Decimal | None) -> None:
        self.reserved_usd -= reserved or Decimal(0)
        self.committed_usd += actual if actual is not None else (reserved or Decimal(0))

    def release(self, reserved: Decimal | None) -> None:
        """Undo a reservation for a call the provider rejected before generating."""
        self.reserved_usd -= reserved or Decimal(0)
        self.dispatched -= 1

    @property
    def spent_or_reserved_usd(self) -> Decimal:
        return self.committed_usd + self.reserved_usd


class RateLimiter:
    """Minimum spacing between dispatches to one provider (requests per minute)."""

    def __init__(self, requests_per_minute: float) -> None:
        self.interval = 60.0 / requests_per_minute
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next = max(now, self._next) + self.interval
