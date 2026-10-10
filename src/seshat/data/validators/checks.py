"""Semantic validation of the normalized layer.

Every observation gets a decision (PASS / WARNING / QUARANTINE / REJECT) plus a list
of flags. Nothing is modified or deleted: decisions are written alongside the data
(``normalized/_validation/decisions.parquet``), and training preparation uses only
eligible rows. Unavailable labels (null) are kept distinct from observed zeros.

Checks (codes in parentheses):

* identity: observation has a request (``orphan_observation``); provider/model match
  (``identity_mismatch``); duplicates (``duplicate_observation``,
  ``duplicate_response_id``).
* provenance: mock providers must be ``test_fixture`` (``source_label_mismatch``).
* outcome: failed (``failed_request``), uncertain (``uncertain_outcome``), missing
  usage (``missing_usage``), truncated (``censored_output``, WARNING).
* accounting: ``total_accounting``, ``cache_accounting``, ``reasoning_accounting``,
  provider-reported contradictions (``provider_accounting``).
* cost: currency/negatives/upper bound (``cost_units``).
* preflight vs observed input (``preflight_discrepancy``, WARNING).
* routing: resolved model differs from the requested one without a routing mode
  (``unexpected_model_routing``, WARNING).
* point-in-time: request time after observation time (``temporal_inconsistency``).
* leakage-risk: identical prompt content in different families (``family_collision``,
  WARNING; resolved by group-aware splitting).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from seshat.data.models import QualityDecision
from seshat.data.provenance import data_origin
from seshat.providers.registry import adapter_class

_RANK = {
    QualityDecision.PASS: 0,
    QualityDecision.WARNING: 1,
    QualityDecision.QUARANTINE: 2,
    QualityDecision.REJECT: 3,
}
ELIGIBLE_LABELS = {"verified", "partial", "censored", "synthetic"}


@dataclass
class RowResult:
    flags: list[tuple[str, QualityDecision, str]] = field(default_factory=list)

    def add(self, code: str, severity: QualityDecision, message: str) -> None:
        self.flags.append((code, severity, message))

    @property
    def decision(self) -> QualityDecision:
        return max((f[1] for f in self.flags), key=_RANK.__getitem__, default=QualityDecision.PASS)


def _is(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and pd.isna(value))


def _n(value: Any) -> int:
    """int() of a value already checked with ``_is`` (keeps mypy precise)."""
    return int(value)


def _family(provider: str) -> str | None:
    try:
        return adapter_class(provider).family
    except ValueError:
        return None


def check_accounting(row: pd.Series, result: RowResult) -> None:
    inp, out = row.get("actual_input_tokens"), row.get("actual_output_tokens")
    total = row.get("normalized_total_tokens")
    if _is(inp) and _is(out) and _is(total) and _n(total) != _n(inp) + _n(out):
        result.add(
            "total_accounting", QualityDecision.QUARANTINE, "normalized_total != input + output"
        )
    cached, uncached, created = (
        row.get("actual_cached_input_tokens"),
        row.get("actual_uncached_input_tokens"),
        row.get("actual_cache_creation_tokens"),
    )
    if _is(inp) and _is(cached) and _n(cached) > _n(inp):
        result.add("cache_accounting", QualityDecision.QUARANTINE, "cached input exceeds input")
    family = _family(str(row.get("provider")))
    if _is(inp) and _is(uncached):
        expected = _n(uncached) + (_n(cached) if _is(cached) else 0)
        if family == "anthropic":
            expected += _n(created) if _is(created) else 0
        if expected != _n(inp):
            result.add(
                "cache_accounting",
                QualityDecision.QUARANTINE,
                f"cache components sum to {expected}, input is {_n(inp)}",
            )
    reasoning, visible = row.get("actual_reasoning_tokens"), row.get("actual_visible_output_tokens")
    included = row.get("reasoning_included_in_output")
    if _is(out) and _is(visible):
        if included is True and _is(reasoning) and _n(visible) + _n(reasoning) != _n(out):
            result.add(
                "reasoning_accounting", QualityDecision.QUARANTINE, "visible + reasoning != output"
            )
        if included is False and _n(visible) + (_n(reasoning) if _is(reasoning) else 0) != _n(out):
            result.add(
                "reasoning_accounting",
                QualityDecision.QUARANTINE,
                "output != candidates + thoughts",
            )
    if _is(reasoning) and _is(out) and included is True and _n(reasoning) > _n(out):
        result.add("reasoning_accounting", QualityDecision.QUARANTINE, "reasoning exceeds output")
    dims = row.get("billing_dimensions_json")
    issues = json.loads(dims).get("accounting_issues") if isinstance(dims, str) and dims else None
    if issues:
        result.add("provider_accounting", QualityDecision.QUARANTINE, ", ".join(issues))


def check_cost(row: pd.Series, result: RowResult) -> None:
    cost = row.get("calculated_cost_usd")
    if not _is(cost):
        return
    cost = Decimal(str(cost))
    if row.get("currency") != "USD" or cost < 0:
        result.add("cost_units", QualityDecision.QUARANTINE, "cost must be non-negative USD")
        return
    rates = [
        row.get(k)
        for k in (
            "input_rate_usd_per_million",
            "output_rate_usd_per_million",
            "cache_write_rate_usd_per_million",
        )
    ]
    known = [Decimal(str(r)) for r in rates if _is(r)]
    inp, out = row.get("actual_input_tokens"), row.get("actual_output_tokens")
    if known and _is(inp) and _is(out):
        bound = (Decimal(_n(inp)) * max(known) + Decimal(_n(out)) * max(known)) / Decimal(1_000_000)
        if cost > bound * Decimal("1.000001"):
            result.add(
                "cost_units", QualityDecision.QUARANTINE, "cost exceeds the price-table upper bound"
            )


def validate_layer(
    requests: pd.DataFrame,
    preflight: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    preflight_tolerance: float = 0.1,
) -> pd.DataFrame:
    """One row per observation with ``decision``, ``flags_json`` and eligibility."""
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "observation_id",
                "sample_id",
                "decision",
                "flags_json",
                "eligible_for_training",
            ]
        )
    req = (
        requests.drop_duplicates("sample_id").set_index("sample_id")
        if not requests.empty
        else pd.DataFrame()
    )
    native = (
        preflight[preflight["preflight_input_tokens"].notna()]
        .sort_values("preflight_method")
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        if not preflight.empty
        else pd.DataFrame()
    )
    dup_obs = set(observations.loc[observations["observation_id"].duplicated(), "observation_id"])
    responses = observations["response_id"].dropna()
    dup_resp = set(responses[responses.duplicated(keep=False)])
    collisions: set[str] = set()
    if not requests.empty and "prompt_family_id" in requests:
        families = requests.groupby("prompt_hash")["prompt_family_id"].nunique(dropna=True)
        collisions = set(families[families > 1].index)
    seen: set[str] = set()
    out_rows: list[dict[str, Any]] = []
    for _, row in observations.iterrows():
        result = RowResult()
        oid, sid = row["observation_id"], row["sample_id"]
        if oid in dup_obs and oid in seen:
            result.add("duplicate_observation", QualityDecision.REJECT, "observation_id repeated")
        seen.add(oid)
        if _is(row.get("response_id")) and row["response_id"] in dup_resp:
            result.add(
                "duplicate_response_id",
                QualityDecision.QUARANTINE,
                "response_id shared by several observations",
            )
        request = req.loc[sid] if not req.empty and sid in req.index else None
        if request is None:
            result.add("orphan_observation", QualityDecision.REJECT, "no request record for sample")
        else:
            if (
                request["provider"] != row["provider"]
                or request["requested_model"] != row["requested_model"]
            ):
                result.add(
                    "identity_mismatch",
                    QualityDecision.REJECT,
                    "provider/model differ between request and observation",
                )
            if request["source_type"] != row["source_type"]:
                result.add(
                    "source_label_mismatch",
                    QualityDecision.REJECT,
                    "request and observation source types differ",
                )
            if pd.Timestamp(request["timestamp_utc"]) > pd.Timestamp(row["collected_at_utc"]):
                result.add(
                    "temporal_inconsistency",
                    QualityDecision.QUARANTINE,
                    "request time is after observation time",
                )
            if request["prompt_hash"] in collisions:
                result.add(
                    "family_collision",
                    QualityDecision.WARNING,
                    "same prompt content in several families",
                )
        try:
            mock = not adapter_class(str(row["provider"])).makes_external_calls
        except ValueError:
            mock = False
        if mock and row["source_type"] != "test_fixture":
            result.add(
                "source_label_mismatch",
                QualityDecision.REJECT,
                "mock provider output must be test_fixture",
            )
        status = row["response_status"]
        if status == "failed":
            result.add("failed_request", QualityDecision.QUARANTINE, str(row.get("error_type")))
        elif status == "uncertain":
            result.add("uncertain_outcome", QualityDecision.QUARANTINE, "execution outcome unknown")
        elif row["label_quality"] == "unavailable":
            result.add("missing_usage", QualityDecision.QUARANTINE, "response without usable usage")
        if row["label_quality"] == "inconsistent":
            result.add(
                "provider_accounting",
                QualityDecision.QUARANTINE,
                "normalizer marked usage inconsistent",
            )
        if row.get("is_truncated") is True or row["label_quality"] == "censored":
            result.add(
                "censored_output",
                QualityDecision.WARNING,
                "truncated: output length is right-censored",
            )
        check_accounting(row, result)
        check_cost(row, result)
        if not native.empty and sid in native.index and _is(row.get("actual_input_tokens")):
            pre = native.loc[sid]
            if bool(pre["preflight_complete"]) and pre["preflight_method"] == "provider_native":
                actual = _n(row["actual_input_tokens"])
                diff = abs(actual - _n(pre["preflight_input_tokens"])) / max(actual, 1)
                if diff > preflight_tolerance:
                    result.add(
                        "preflight_discrepancy",
                        QualityDecision.WARNING,
                        f"relative difference {diff:.1%}",
                    )
        resolved = row.get("resolved_model")
        routed = request is not None and _is(request.get("routing_mode"))
        if (
            _is(resolved)
            and not str(resolved).startswith(str(row["requested_model"]))
            and not routed
        ):
            result.add("unexpected_model_routing", QualityDecision.WARNING, f"resolved {resolved}")
        decision = result.decision
        eligible = (
            decision in {QualityDecision.PASS, QualityDecision.WARNING}
            and row["label_quality"] in ELIGIBLE_LABELS
        )
        out_rows.append(
            {
                "observation_id": oid,
                "sample_id": sid,
                "provider": row["provider"],
                "requested_model": row["requested_model"],
                "source_type": row["source_type"],
                "label_quality": row["label_quality"],
                "data_origin": data_origin(row["source_type"], row["label_quality"]).value,
                "decision": decision.value,
                "flag_codes": sorted({f[0] for f in result.flags}),
                "flags_json": json.dumps(
                    [{"code": c, "severity": s.value, "message": m} for c, s, m in result.flags]
                ),
                "eligible_for_training": eligible,
            }
        )
    return pd.DataFrame(out_rows)


def flag_counts(decisions: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    codes: Iterable[list[str]] = decisions.get("flag_codes", pd.Series(dtype=object))
    for row_codes in codes:
        for code in row_codes:
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))
