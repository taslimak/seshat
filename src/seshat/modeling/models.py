"""Point/uncertainty forecasters for output tokens.

Every model exposes ``fit(frame)`` and ``predict_log(frame) -> (mu, sigma)`` on
``z = log1p(output_tokens)``. The point forecast is ``expm1(mu)`` (the median of
the predictive distribution, which minimises expected absolute error); ``sigma``
is the model's own scale estimate used by the conformal calibration layer.

``frame`` columns: the model features from :mod:`seshat.modeling.features`,
``actual_output_tokens``, ``censored`` (bool) and ``cap`` (``max_output_tokens``).
Censored rows mean ``y >= cap``; only :class:`CensoredLogNormal` uses them as
such. Models that cannot handle censoring drop them (and are biased low for
long outputs; that is why they are candidates, not defaults).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr

from seshat.modeling.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SIGNAL_FEATURES

LOG_2PI = math.log(2 * math.pi)
# The log-scale sd is mapped smoothly into [SIGMA_MIN, SIGMA_MAX]:
#   log_sigma = LO + (HI - LO) * sigmoid(raw)
# The floor stops the variance model from becoming overconfident on small segments
# (out-of-fold residuals then explode); the ceiling (a factor ~4.5 per sd) stops
# fully-censored segments from pushing the scale to infinity. Gradients never vanish.
SIGMA_MIN, SIGMA_MAX = 0.15, 1.5
LOG_SIGMA_LO, LOG_SIGMA_HI = math.log(SIGMA_MIN), math.log(SIGMA_MAX)


def _bounded_log_sigma(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(log_sigma, d log_sigma / d raw)."""
    sig = 1.0 / (1.0 + np.exp(-raw))
    span = LOG_SIGMA_HI - LOG_SIGMA_LO
    return LOG_SIGMA_LO + span * sig, span * sig * (1.0 - sig)


def _raw_for_sigma(sigma: float) -> float:
    pos = (math.log(sigma) - LOG_SIGMA_LO) / (LOG_SIGMA_HI - LOG_SIGMA_LO)
    pos = min(max(pos, 1e-3), 1 - 1e-3)
    return math.log(pos / (1 - pos))


class Forecaster(Protocol):
    name: str

    def fit(self, frame: pd.DataFrame) -> Forecaster: ...

    def predict_log(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]: ...

    def to_dict(self) -> dict[str, Any]: ...


def _targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    y = frame["actual_output_tokens"].to_numpy(dtype=float)
    censored = frame["censored"].to_numpy(dtype=bool)
    return np.log1p(np.maximum(y, 0.0)), censored


@dataclass
class DesignMatrix:
    """Standardised numeric + one-hot categorical columns; fitted on training rows only."""

    numeric: tuple[str, ...] = NUMERIC_FEATURES + SIGNAL_FEATURES
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES
    means: dict[str, float] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    levels: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, min_level_count: int = 2) -> DesignMatrix:
        for column in self.numeric:
            values = frame[column].astype(float)
            self.means[column] = float(values.mean())
            std = float(values.std(ddof=0))
            self.scales[column] = std if std > 1e-9 else 1.0
        for column in self.categorical:
            counts = frame[column].astype(str).value_counts()
            self.levels[column] = sorted(str(k) for k, v in counts.items() if v >= min_level_count)
        return self

    @property
    def names(self) -> list[str]:
        cols = ["intercept", *self.numeric]
        for column in self.categorical:
            cols += [f"{column}={level}" for level in self.levels[column]]
        return cols

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = [np.ones((len(frame), 1))]
        for column in self.numeric:
            numbers = frame[column].astype(float).to_numpy()
            parts.append(((numbers - self.means[column]) / self.scales[column])[:, None])
        for column in self.categorical:
            labels = frame[column].astype(str).to_numpy()
            for level in self.levels[column]:
                parts.append((labels == level).astype(float)[:, None])
        return np.hstack(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "means": self.means,
            "scales": self.scales,
            "levels": self.levels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignMatrix:
        return cls(
            numeric=tuple(data["numeric"]),
            categorical=tuple(data["categorical"]),
            means=dict(data["means"]),
            scales=dict(data["scales"]),
            levels={k: list(v) for k, v in data["levels"].items()},
        )


@dataclass
class CensoredLogNormal:
    """Heteroscedastic censored log-normal (Tobit-type) regression with L2 penalties.

    ``log1p(y) ~ Normal(mu = X beta, sigma = exp(X gamma))``, right-censored at
    ``log1p(cap)`` for truncated responses. MAP estimate by L-BFGS with an
    analytic gradient. ``scale_penalty`` is larger than ``penalty`` so the
    variance model stays simpler than the mean model.
    """

    penalty: float = 1.0
    scale_penalty: float = 4.0
    name: str = "censored_lognormal"
    design: DesignMatrix = field(default_factory=DesignMatrix)
    beta: np.ndarray | None = None
    gamma: np.ndarray | None = None

    def _unpack(self, theta: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
        return theta[:p], theta[p:]

    def fit(self, frame: pd.DataFrame) -> CensoredLogNormal:
        self.design = DesignMatrix().fit(frame)
        x = self.design.transform(frame)
        z, censored = _targets(frame)
        p = x.shape[1]
        observed = ~censored
        mask = np.ones(p)
        mask[0] = 0.0  # intercepts are not penalised

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            beta, gamma = self._unpack(theta, p)
            mu = x @ beta
            log_sigma, slope = _bounded_log_sigma(x @ gamma)
            sigma = np.exp(log_sigma)
            r = (z - mu) / sigma
            ll = np.where(observed, -0.5 * r**2 - log_sigma - 0.5 * LOG_2PI, 0.0)
            log_surv = log_ndtr(-r)
            ll = ll + np.where(censored, log_surv, 0.0)
            # d ll / d mu and d ll / d log sigma
            mills = np.exp(-0.5 * r**2 - 0.5 * LOG_2PI - log_surv)
            d_mu = np.where(observed, r / sigma, mills / sigma)
            d_ls = np.where(observed, r**2 - 1.0, mills * r) * slope
            penalty = self.penalty * np.sum((beta * mask) ** 2) + self.scale_penalty * np.sum(
                (gamma * mask) ** 2
            )
            value = -float(ll.sum()) + penalty
            grad_beta = -(x.T @ d_mu) + 2 * self.penalty * beta * mask
            grad_gamma = -(x.T @ d_ls) + 2 * self.scale_penalty * gamma * mask
            return value, np.concatenate([grad_beta, grad_gamma])

        start = np.zeros(2 * p)
        start[0] = float(np.mean(z))
        start[p] = _raw_for_sigma(min(max(float(np.std(z)), SIGMA_MIN * 1.5), SIGMA_MAX / 1.5))
        result = minimize(objective, start, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.beta, self.gamma = self._unpack(result.x, p)
        return self

    def predict_log(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.beta is None or self.gamma is None:
            raise RuntimeError("model is not fitted")
        x = self.design.transform(frame)
        return x @ self.beta, np.exp(_bounded_log_sigma(x @ self.gamma)[0])

    def to_dict(self) -> dict[str, Any]:
        if self.beta is None or self.gamma is None:
            raise RuntimeError("model is not fitted")
        return {
            "type": self.name,
            "penalty": self.penalty,
            "scale_penalty": self.scale_penalty,
            "design": self.design.to_dict(),
            "beta": dict(zip(self.design.names, map(float, self.beta), strict=True)),
            "gamma": dict(zip(self.design.names, map(float, self.gamma), strict=True)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CensoredLogNormal:
        model = cls(penalty=data["penalty"], scale_penalty=data["scale_penalty"])
        model.design = DesignMatrix.from_dict(data["design"])
        names = model.design.names
        model.beta = np.array([data["beta"][n] for n in names])
        model.gamma = np.array([data["gamma"][n] for n in names])
        return model


@dataclass
class UncensoredRidge:
    """Ridge regression on log1p(y), uncensored rows only (current trainer's approach)."""

    penalty: float = 1.0
    name: str = "ridge_log_uncensored"
    design: DesignMatrix = field(default_factory=DesignMatrix)
    beta: np.ndarray | None = None
    sigma: float = 1.0

    def fit(self, frame: pd.DataFrame) -> UncensoredRidge:
        rows = frame[~frame["censored"].astype(bool)]
        self.design = DesignMatrix().fit(rows)
        x = self.design.transform(rows)
        z, _ = _targets(rows)
        reg = self.penalty * np.eye(x.shape[1])
        reg[0, 0] = 0.0
        self.beta = np.linalg.solve(x.T @ x + reg, x.T @ z)
        self.sigma = float(np.std(z - x @ self.beta)) or 1.0
        return self

    def predict_log(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.beta is None:
            raise RuntimeError("model is not fitted")
        mu = self.design.transform(frame) @ self.beta
        return mu, np.full(len(frame), self.sigma)

    def to_dict(self) -> dict[str, Any]:
        if self.beta is None:
            raise RuntimeError("model is not fitted")
        return {
            "type": self.name,
            "penalty": self.penalty,
            "design": self.design.to_dict(),
            "beta": dict(zip(self.design.names, map(float, self.beta), strict=True)),
            "sigma": self.sigma,
        }


@dataclass
class GroupMedian:
    """Median log output per (effort, task_group) with shrinkage to the effort median."""

    shrinkage: float = 5.0
    name: str = "group_median"
    table: dict[str, float] = field(default_factory=dict)
    fallback: dict[str, float] = field(default_factory=dict)
    global_value: float = 0.0
    sigma: float = 1.0

    def fit(self, frame: pd.DataFrame) -> GroupMedian:
        rows = frame[~frame["censored"].astype(bool)]
        z, _ = _targets(rows)
        rows = rows.assign(_z=z)
        self.global_value = float(np.median(z))
        for level, part in rows.groupby("effort"):
            self.fallback[str(level)] = float(np.median(part["_z"]))
        for (effort, group), part in rows.groupby(["effort", "task_group"]):
            base = self.fallback.get(str(effort), self.global_value)
            n = len(part)
            weight = n / (n + self.shrinkage)
            self.table[f"{effort}|{group}"] = (
                weight * float(np.median(part["_z"])) + (1 - weight) * base
            )
        self.sigma = float(np.std(z)) or 1.0
        return self

    def predict_log(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = np.array(
            [
                self.table.get(f"{e}|{g}", self.fallback.get(str(e), self.global_value))
                for e, g in zip(frame["effort"], frame["task_group"], strict=True)
            ]
        )
        return mu, np.full(len(frame), self.sigma)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "table": self.table,
            "fallback": self.fallback,
            "global": self.global_value,
        }


@dataclass
class GradientBoostedMedian:
    """Histogram GBM with absolute-error loss on log1p(y), uncensored rows only."""

    name: str = "gbm_median_uncensored"
    model: Any = None
    design: DesignMatrix = field(default_factory=DesignMatrix)
    sigma: float = 1.0

    def fit(self, frame: pd.DataFrame) -> GradientBoostedMedian:
        from sklearn.ensemble import HistGradientBoostingRegressor

        rows = frame[~frame["censored"].astype(bool)]
        self.design = DesignMatrix().fit(rows)
        z, _ = _targets(rows)
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=8,
            min_samples_leaf=8,
            l2_regularization=1.0,
            random_state=0,
        ).fit(self.design.transform(rows), z)
        self.sigma = float(np.std(z - self.model.predict(self.design.transform(rows)))) or 1.0
        return self

    def predict_log(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        mu = np.asarray(self.model.predict(self.design.transform(frame)), dtype=float)
        return mu, np.full(len(frame), self.sigma)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name, "note": "not serialisable to JSON; evaluation only"}


@dataclass
class HeuristicBaseline:
    """The original heuristic (word count x effort multiplier), for like-for-like comparison."""

    name: str = "baseline-0.1"

    def fit(self, frame: pd.DataFrame) -> HeuristicBaseline:
        return self

    def predict_log(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        from seshat.predictor import BaselinePredictor

        baseline = BaselinePredictor()
        points = [
            baseline.predict_from_word_count(
                int(w), "", None if e == "unset" else e
            ).predicted_output_tokens
            for w, e in zip(frame["prompt_word_count"], frame["effort"], strict=True)
        ]
        return np.log1p(np.array(points, dtype=float)), np.ones(len(frame))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.name}


def candidate_models() -> list[Forecaster]:
    return [
        HeuristicBaseline(),
        GroupMedian(),
        UncensoredRidge(penalty=3.0),
        GradientBoostedMedian(),
        CensoredLogNormal(penalty=10.0, scale_penalty=30.0, name="censored_lognormal_p10_s30"),
        CensoredLogNormal(penalty=30.0, scale_penalty=30.0, name="censored_lognormal_p30_s30"),
        CensoredLogNormal(penalty=100.0, scale_penalty=30.0, name="censored_lognormal_p100_s30"),
        CensoredLogNormal(penalty=30.0, scale_penalty=100.0, name="censored_lognormal_p30_s100"),
    ]
