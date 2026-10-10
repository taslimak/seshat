"""Explicit data-quality policy for normalized observations."""

from __future__ import annotations

import json

import pandas as pd

from seshat.data.models import QualityDecision, QualityFlag

_DECISION_RANK = {
    QualityDecision.PASS: 0,
    QualityDecision.WARNING: 1,
    QualityDecision.QUARANTINE: 2,
    QualityDecision.REJECT: 3,
}


def _max_decision(flags: list[QualityFlag]) -> QualityDecision:
    if not flags:
        return QualityDecision.PASS
    return max((flag.severity for flag in flags), key=_DECISION_RANK.__getitem__)


def validate_row(row: pd.Series) -> list[QualityFlag]:
    flags: list[QualityFlag] = []
    for required in ("observation_id", "run_id", "provider", "model_requested", "prompt_id"):
        if pd.isna(row.get(required)) or str(row.get(required)).strip() == "":
            flags.append(
                QualityFlag(
                    code=f"missing_{required}",
                    message=f"Required field {required} is missing",
                    severity=QualityDecision.REJECT,
                )
            )
    if row.get("status") != "success":
        flags.append(
            QualityFlag(
                code="failed_run",
                message="Provider request did not produce a successful observation",
                severity=QualityDecision.QUARANTINE,
            )
        )
    output_tokens = row.get("output_tokens")
    if pd.isna(output_tokens):
        flags.append(
            QualityFlag(
                code="missing_output_tokens",
                message="Primary target output_tokens is missing",
                severity=QualityDecision.QUARANTINE,
            )
        )
    elif float(output_tokens) < 0:
        flags.append(
            QualityFlag(
                code="negative_output_tokens",
                message="output_tokens cannot be negative",
                severity=QualityDecision.REJECT,
            )
        )
    elif float(output_tokens) == 0:
        flags.append(
            QualityFlag(
                code="zero_output_tokens",
                message="Successful response reported zero output tokens",
                severity=QualityDecision.QUARANTINE,
            )
        )
    for column in ("input_tokens", "reasoning_tokens", "cached_input_tokens", "total_tokens"):
        value = row.get(column)
        if not pd.isna(value) and float(value) < 0:
            flags.append(
                QualityFlag(
                    code=f"negative_{column}",
                    message=f"{column} cannot be negative",
                    severity=QualityDecision.REJECT,
                )
            )
    total = row.get("total_tokens")
    input_tokens = row.get("input_tokens")
    if not pd.isna(total) and not pd.isna(input_tokens) and not pd.isna(output_tokens):
        if float(total) < float(input_tokens) + float(output_tokens):
            flags.append(
                QualityFlag(
                    code="inconsistent_total_tokens",
                    message="total_tokens is smaller than input_tokens + output_tokens",
                    severity=QualityDecision.QUARANTINE,
                )
            )
    if bool(row.get("is_truncated", False)):
        flags.append(
            QualityFlag(
                code="right_censored_output",
                message="Output appears truncated; treat the target as right-censored",
                severity=QualityDecision.WARNING,
            )
        )
    if bool(row.get("is_refusal", False)):
        flags.append(
            QualityFlag(
                code="provider_refusal",
                message="Provider produced a refusal; retain but segment in evaluation",
                severity=QualityDecision.WARNING,
            )
        )
    return flags


def validate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    all_flags: list[str] = []
    decisions: list[str] = []
    duplicate_runs = set(result.loc[result["run_id"].duplicated(keep=False), "run_id"].tolist())
    request_ids = result.get("provider_request_id", pd.Series(index=result.index, dtype=object))
    duplicate_request_ids = set(
        request_ids[request_ids.notna() & request_ids.duplicated(keep=False)].tolist()
    )
    for _, row in result.iterrows():
        flags = validate_row(row)
        if row.get("run_id") in duplicate_runs:
            flags.append(
                QualityFlag(
                    code="duplicate_run_id",
                    message="run_id appears more than once",
                    severity=QualityDecision.REJECT,
                )
            )
        request_id = row.get("provider_request_id")
        if request_id in duplicate_request_ids:
            flags.append(
                QualityFlag(
                    code="duplicate_provider_request_id",
                    message="provider request ID appears more than once",
                    severity=QualityDecision.QUARANTINE,
                )
            )
        decision = _max_decision(flags)
        decisions.append(decision.value)
        all_flags.append(
            json.dumps([flag.model_dump(mode="json") for flag in flags], sort_keys=True)
        )
    result["quality_decision"] = decisions
    result["quality_flags"] = all_flags
    return result
