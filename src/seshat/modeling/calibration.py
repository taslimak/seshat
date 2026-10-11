"""Bias correction (stage B) and conformal interval calibration (stage D).

Both work on log-scale residuals of *out-of-sample* forecasts
(``e = z - mu`` with ``z = log1p(actual)``; note ``e`` is actual minus forecast
on the log scale, so the token error convention ``prediction - actual`` has the
opposite sign). Censored rows only bound ``e`` from below; residual
distributions are therefore estimated with the Kaplan-Meier estimator, which
uses censored rows as "at least this large" instead of dropping them (dropping
would understate long outputs and make upper bounds too tight).

* :class:`BiasCorrector` -- per-segment Kaplan-Meier median residual with
  hierarchical shrinkage (segment -> parent -> global): ``mu' = mu + delta``.
  Small segments move little.
* :class:`ConformalCalibrator` -- normalised residuals ``s = (z - mu') / sigma``
  give asymmetric split-conformal quantiles ``q_lo, q_hi`` at ``alpha/2`` and
  ``1 - alpha/2`` (finite-sample corrected), optionally per segment with
  shrinkage toward the global quantiles. Interval: ``expm1(mu' + q * sigma)``.

Segments must be computable at inference time (provider/model, effort,
``task_group``), never design labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def km_quantile(values: np.ndarray, censored: np.ndarray, level: float) -> float:
    """``level``-quantile of a right-censored sample (Kaplan-Meier). ``inf`` if unreached."""
    values = np.asarray(values, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if values.size == 0:
        return math.nan
    if level <= 0:
        return float(values.min())
    order = np.lexsort((censored, values))  # events before censorings at ties
    values, censored = values[order], censored[order]
    survival = 1.0
    at_risk = values.size
    for value, is_censored in zip(values, censored, strict=True):
        if not is_censored:
            survival *= 1.0 - 1.0 / at_risk
            if 1.0 - survival >= level - 1e-12:
                return float(value)
        at_risk -= 1
    return math.inf


def conformal_level(n: int, level: float) -> float:
    """Finite-sample corrected quantile level ``ceil((n + 1) level) / n`` (capped at 1)."""
    if n <= 0:
        return level
    return min(1.0, math.ceil((n + 1) * level) / n)


def _segment_keys(segments: dict[str, np.ndarray], index: int, hierarchy: list[str]) -> list[str]:
    keys = []
    for depth in range(len(hierarchy), 0, -1):
        keys.append("|".join(f"{h}={segments[h][index]}" for h in hierarchy[:depth]))
    return keys


@dataclass
class BiasCorrector:
    hierarchy: list[str] = field(default_factory=lambda: ["provider_model", "effort", "task_group"])
    shrinkage: float = 15.0
    max_abs_delta: float = 1.0  # at most a factor e (~2.7x) in either direction
    global_delta: float = 0.0
    deltas: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def fit(
        self, residual: np.ndarray, censored: np.ndarray, segments: dict[str, np.ndarray]
    ) -> BiasCorrector:
        self.global_delta = float(np.clip(km_quantile(residual, censored, 0.5), -1, 1))
        if not math.isfinite(self.global_delta):
            self.global_delta = 0.0
        parent: dict[str, float] = {"": self.global_delta}
        for depth in range(1, len(self.hierarchy) + 1):
            keys = np.array(
                [
                    "|".join(f"{h}={segments[h][i]}" for h in self.hierarchy[:depth])
                    for i in range(residual.size)
                ]
            )
            for key in np.unique(keys):
                mask = keys == key
                n = int(mask.sum())
                parent_key = key.rsplit("|", 1)[0] if depth > 1 else ""
                base = parent.get(parent_key, self.global_delta)
                median = km_quantile(residual[mask], censored[mask], 0.5)
                if not math.isfinite(median):
                    median = base
                weight = n / (n + self.shrinkage)
                value = float(np.clip(weight * median + (1 - weight) * base, -1, 1))
                self.deltas[str(key)] = value
                self.counts[str(key)] = n
                parent[str(key)] = value
        return self

    def delta(self, segments: dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(segments.values())))
        out = np.full(n, self.global_delta)
        for i in range(n):
            for key in _segment_keys(segments, i, self.hierarchy):
                if key in self.deltas:
                    out[i] = self.deltas[key]
                    break
        return np.asarray(np.clip(out, -self.max_abs_delta, self.max_abs_delta), dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hierarchy": self.hierarchy,
            "shrinkage": self.shrinkage,
            "global_delta": self.global_delta,
            "deltas": self.deltas,
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BiasCorrector:
        return cls(
            hierarchy=list(data["hierarchy"]),
            shrinkage=float(data["shrinkage"]),
            global_delta=float(data["global_delta"]),
            deltas={k: float(v) for k, v in data["deltas"].items()},
            counts={k: int(v) for k, v in data["counts"].items()},
        )


@dataclass
class ConformalCalibrator:
    confidence_level: float = 0.90
    segment_column: str | None = "task_group"
    shrinkage: float = 20.0
    q_lo: float = -1.645
    q_hi: float = 1.645
    segment_q: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    n_calibration: int = 0
    upper_tail_censoring_limited: bool = False

    def fit(
        self,
        score: np.ndarray,
        censored: np.ndarray,
        segments: dict[str, np.ndarray] | None = None,
    ) -> ConformalCalibrator:
        alpha = 1 - self.confidence_level
        n = int(score.size)
        self.n_calibration = n
        lo_level = 1 - conformal_level(n, 1 - alpha / 2)  # lower tail, corrected outward
        hi_level = conformal_level(n, 1 - alpha / 2)
        self.q_lo = km_quantile(score, censored, max(lo_level, 0.0))
        q_hi = km_quantile(score, censored, hi_level)
        if not math.isfinite(q_hi):
            # Too much censoring at the top to reach the level: use the largest
            # observed score (a lower bound on the needed quantile) and flag it.
            self.upper_tail_censoring_limited = True
            q_hi = float(np.max(score))
        self.q_hi = q_hi
        if segments is not None and self.segment_column is not None:
            keys = segments[self.segment_column]
            # A segment can only estimate its own alpha/2 tail once the finite-sample
            # level ceil((m + 1)(1 - alpha/2)) / m stays below 1, i.e. m >= 2/alpha - 1;
            # smaller segments would fall back to their maximum residual.
            min_rows = math.ceil(2 / alpha - 1)
            for key in np.unique(keys):
                mask = keys == key
                m = int(mask.sum())
                if m < min_rows:
                    continue
                lo = km_quantile(
                    score[mask], censored[mask], max(1 - conformal_level(m, 1 - alpha / 2), 0.0)
                )
                hi = km_quantile(score[mask], censored[mask], conformal_level(m, 1 - alpha / 2))
                if not math.isfinite(hi):
                    hi = max(float(np.max(score[mask])), self.q_hi)
                weight = m / (m + self.shrinkage)
                self.segment_q[str(key)] = (
                    weight * lo + (1 - weight) * self.q_lo,
                    weight * hi + (1 - weight) * self.q_hi,
                    m,
                )
        return self

    def quantiles(
        self, segments: dict[str, np.ndarray] | None, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = np.full(n, self.q_lo)
        hi = np.full(n, self.q_hi)
        if segments is not None and self.segment_column is not None and self.segment_q:
            for i, key in enumerate(segments[self.segment_column]):
                if str(key) in self.segment_q:
                    lo[i], hi[i], _ = self.segment_q[str(key)]
        return lo, hi

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "split-conformal on sigma-normalised log residuals, Kaplan-Meier "
            "quantiles (censoring-aware), segment quantiles shrunk to global",
            "confidence_level": self.confidence_level,
            "segment_column": self.segment_column,
            "shrinkage": self.shrinkage,
            "q_lo": self.q_lo,
            "q_hi": self.q_hi,
            "segment_q": {k: list(v) for k, v in self.segment_q.items()},
            "n_calibration": self.n_calibration,
            "upper_tail_censoring_limited": self.upper_tail_censoring_limited,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConformalCalibrator:
        return cls(
            confidence_level=float(data["confidence_level"]),
            segment_column=data["segment_column"],
            shrinkage=float(data["shrinkage"]),
            q_lo=float(data["q_lo"]),
            q_hi=float(data["q_hi"]),
            segment_q={
                k: (float(v[0]), float(v[1]), int(v[2])) for k, v in data["segment_q"].items()
            },
            n_calibration=int(data["n_calibration"]),
            upper_tail_censoring_limited=bool(data["upper_tail_censoring_limited"]),
        )


def content_version(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return f"{prefix}-{digest[:10]}"
