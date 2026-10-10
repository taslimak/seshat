"""Budget enforcement and per-provider rate limiting."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path


class BudgetExceededError(RuntimeError):
    pass


class SharedBudgetLedger:
    """Atomic spending ledger for experiments sharing one durable SQLite database.

    All collector processes must point at the SAME persistent file. Do not use
    an ephemeral Actions checkout or artifact as a substitute for durable storage.
    Reservations survive crashes; reconcile them manually against provider billing.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        daily_cap_usd: Decimal,
        monthly_cap_usd: Decimal,
    ) -> None:
        if daily_cap_usd < 0 or monthly_cap_usd < 0:
            raise ValueError("Shared cost caps must be nonnegative")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_cap_usd = daily_cap_usd
        self.monthly_cap_usd = monthly_cap_usd
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS charges "
                "(id INTEGER PRIMARY KEY, day TEXT NOT NULL, month TEXT NOT NULL, "
                "amount TEXT NOT NULL, state TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def reserve(self, amount: Decimal) -> int | None:
        """Return reservation ID or None if either shared cost cap would be exceeded."""
        if amount < 0:
            raise ValueError("Cannot reserve negative costs")
        today = date.today().isoformat()
        month = today[:7]
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT day, amount FROM charges WHERE month = ?", (month,)
            ).fetchall()
            total_month = sum((Decimal(value) for _, value in rows), Decimal(0))
            total_day = sum((Decimal(value) for day, value in rows if day == today), Decimal(0))
            if (
                total_day + amount > self.daily_cap_usd
                or total_month + amount > self.monthly_cap_usd
            ):
                return None
            cursor = db.execute(
                "INSERT INTO charges (day, month, amount, state) VALUES (?, ?, ?, ?)",
                (today, month, str(amount), "reserved"),
            )
            return int(cursor.lastrowid)

    def settle(self, reservation_id: int, actual: Decimal) -> None:
        """Conservatively retain the reservation if a response has no known cost."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT amount FROM charges WHERE id = ?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Missing shared budget reservation")
            if actual < 0:
                raise ValueError("Cannot settle negative costs")
            db.execute(
                "UPDATE charges SET amount=?, state='settled' WHERE id=?",
                (str(actual), reservation_id),
            )

    def release(self, reservation_id: int) -> None:
        """Release known rejected charges; request attempt accounting remains local."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE charges SET amount='0', state='released' WHERE id=?",
                (reservation_id,),
            )


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
    shared_ledger: SharedBudgetLedger | None = None
    _reservations: dict[Decimal, list[int]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._reservations = {}

    def try_reserve(self, estimate: Decimal | None) -> str | None:
        """Reserve one request; returns a refusal reason or ``None`` when allowed."""
        if self.dispatched >= self.max_requests:
            return "max_requests_reached"
        if self.max_cost_usd is not None:
            if estimate is None:
                return "cost_unbounded"
            if self.committed_usd + self.reserved_usd + estimate > self.max_cost_usd:
                return "max_cost_reached"
        if self.shared_ledger is not None:
            if estimate is None:
                return "cost_unbounded"
            reservation_id = self.shared_ledger.reserve(estimate)
            if reservation_id is None:
                return "shared_cost_cap_reached"
            self._reservations.setdefault(estimate, []).append(reservation_id)
        self.dispatched += 1
        self.reserved_usd += estimate or Decimal(0)
        return None

    def settle(self, reserved: Decimal | None, actual: Decimal | None) -> None:
        self.reserved_usd -= reserved or Decimal(0)
        amount = actual if actual is not None else (reserved or Decimal(0))
        if self.shared_ledger is not None:
            self.shared_ledger.settle(self._reservation(reserved), amount)
        self.committed_usd += amount

    def release(self, reserved: Decimal | None) -> None:
        """Undo a reservation for a call the provider rejected before generating."""
        if self.shared_ledger is not None:
            self.shared_ledger.release(self._reservation(reserved))
        self.reserved_usd -= reserved or Decimal(0)
        # Attempted external calls still count toward the request limit.

    def _reservation(self, amount: Decimal | None) -> int:
        ids = self._reservations.get(amount or Decimal(0), [])
        if not ids:
            raise ValueError("Missing in-memory shared budget reservation")
        return ids.pop()

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
