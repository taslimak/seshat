"""Training-readiness analysis on the real-label store.

Readiness is judged by *simulating the exact split* ``prepare-training`` would
make (same leakage groups, same seeded group assignment) and counting, per split,
the rows the trainer can actually use: real, controlled-experiment, verified,
uncensored labels. A raw total (e.g. "140 usable rows") is not sufficient: rows
are allocated by leakage *group*, censored rows still occupy groups, and the
calibration share of :data:`~seshat.data.split.DEFAULT_RATIOS` (8%) needs ~250
usable rows to reach the trainer's 20-row calibration minimum.

Nothing here sends requests or trains models.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.split import DEFAULT_RATIOS
from seshat.data.storage import StorageLayout
from seshat.data.training import (
    DEFAULT_GROUP_BY,
    REAL_LABEL_RATIOS,
    assign_dataset_splits,
    select_labeled_rows,
)

# Must match seshat.train_predictor.MIN_TRAIN / MIN_CALIBRATION / MIN_TEST.
TRAINER_MINIMUMS = {"train": 100, "calibration": 20, "test": 20}
RATIO_PRESETS = {"default": DEFAULT_RATIOS, "real_label": REAL_LABEL_RATIOS}
MIN_GROUPS_PER_CATEGORY = 4  # so a category can appear in train and in held-out splits


def usable_mask(frame: pd.DataFrame) -> pd.Series:
    """Rows the trainer accepts (mirrors ``seshat.train_predictor._eligible``)."""
    output = pd.to_numeric(frame["actual_output_tokens"], errors="coerce")
    return (
        (frame["label_origin"] == "real")
        & (frame["source_type"] == "controlled_experiment")
        & (frame["label_quality"] == "verified")
        & ~frame["target_is_censored"].fillna(True).astype(bool)
        & output.notna()
    )


def implied_minimum_usable(ratios: dict[str, float]) -> int:
    """Usable rows needed so every minimum is met if usable rows split by ``ratios``."""
    return math.ceil(max(minimum / ratios[name] for name, minimum in TRAINER_MINIMUMS.items()))


def _dist(values: pd.Series) -> dict[str, float] | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "n": int(len(values)),
        "mean": round(float(values.mean()), 1),
        "p10": float(values.quantile(0.1)),
        "p50": float(values.median()),
        "p90": float(values.quantile(0.9)),
        "max": float(values.max()),
    }


def _by_category(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for category, group in frame.groupby("prompt_category", sort=True):
        usable = group[group["_usable"]]
        censored = group["target_is_censored"].astype(bool)
        out[str(category)] = {
            "rows": int(len(group)),
            "usable": int(len(usable)),
            "censored": int(censored.sum()),
            "censor_rate": round(float(censored.mean()), 3),
            "families": int(group["prompt_family_id"].nunique()),
            "groups": int(group["split_group_id"].nunique()),
            "usable_groups": int(usable["split_group_id"].nunique()),
            "max_output_tokens_levels": sorted(
                int(v) for v in group["max_output_tokens"].dropna().unique()
            ),
            "usable_output_tokens": _dist(usable["actual_output_tokens"]),
        }
    return out


def _simulate(frame: pd.DataFrame, ratios: dict[str, float], seed: int) -> dict[str, Any]:
    splits, used_ratios, _ = assign_dataset_splits(frame, seed=seed, ratios=ratios)
    per_split: dict[str, Any] = {}
    shortfalls: dict[str, int] = {}
    for name in ("train", "validation", "calibration", "test"):
        part = frame[splits == name]
        usable = part[part["_usable"]]
        minimum = TRAINER_MINIMUMS.get(name)
        short = max(0, minimum - len(usable)) if minimum is not None else 0
        if minimum is not None:
            shortfalls[name] = short
        per_split[name] = {
            "rows": int(len(part)),
            "usable": int(len(usable)),
            "groups": int(part["split_group_id"].nunique()),
            "usable_categories": int(usable["prompt_category"].nunique()),
            "minimum_usable": minimum,
            "shortfall": short,
        }
    # New independent groups are spread over splits by ratio, so covering the
    # worst shortfall s requires about shortfall_s / ratio_s new usable rows.
    additional = max(
        (math.ceil(short / used_ratios[name]) for name, short in shortfalls.items() if short),
        default=0,
    )
    return {
        "ratios": used_ratios,
        "implied_minimum_usable": implied_minimum_usable(used_ratios),
        "splits": per_split,
        "meets_trainer_minimums": not any(shortfalls.values()),
        "estimated_additional_usable_rows": additional,
    }


def readiness_report(
    root: str | Path,
    *,
    seed: int = 1729,
    group_by: tuple[str, ...] = DEFAULT_GROUP_BY,
) -> dict[str, Any]:
    layout = StorageLayout(root)
    base: dict[str, Any] = {
        "status": "no_data",
        "trainer_minimums": TRAINER_MINIMUMS,
        "seed": seed,
        "group_by": list(group_by),
        "note": (
            "Usable = real, controlled-experiment, verified, uncensored labels. Meeting the "
            "minimums makes training *possible*; it does not establish that a model "
            "generalizes or beats the baseline."
        ),
    }
    observations = layout.read_frame("observations")
    if observations.empty or not (layout.validation / "decisions.parquet").exists():
        return base
    try:
        frame, excluded = select_labeled_rows(layout, label_origin="real", group_by=group_by)
    except ValueError as exc:
        return {**base, "status": "no_real_labels", "detail": str(exc)}
    frame = frame.copy()
    frame["_usable"] = usable_mask(frame)
    usable = frame[frame["_usable"]]
    by_category = _by_category(frame)
    group_sizes = frame.groupby("split_group_id").size().sort_values(ascending=False)
    simulations = {name: _simulate(frame, ratios, seed) for name, ratios in RATIO_PRESETS.items()}
    gaps: list[str] = []
    for category, info in by_category.items():
        if info["usable"] == 0:
            gaps.append(f"{category}: no usable (uncensored) labels")
        elif info["usable_groups"] < MIN_GROUPS_PER_CATEGORY:
            gaps.append(
                f"{category}: only {info['usable_groups']} independent usable group(s); "
                f"needs >= {MIN_GROUPS_PER_CATEGORY} to be represented in train and held-out splits"
            )
        if info["censor_rate"] > 0.2:
            gaps.append(f"{category}: censor rate {info['censor_rate']:.0%}")
    for column, label in (("requested_model", "model"), ("requested_effort", "effort level")):
        levels = frame[column].fillna("unset").astype(str).unique()
        if len(levels) < 2:
            gaps.append(f"only one {label} observed ({levels[0]}): no evidence on its effect")
    status = (
        "ready_for_training"
        if simulations["real_label"]["meets_trainer_minimums"]
        or simulations["default"]["meets_trainer_minimums"]
        else "insufficient_data"
    )
    return {
        **base,
        "status": status,
        "excluded_by_source_policy": excluded,
        "totals": {
            "eligible_real_rows": int(len(frame)),
            "usable_rows": int(len(usable)),
            "censored_rows": int(frame["target_is_censored"].astype(bool).sum()),
            "non_verified_rows": int((frame["label_quality"] != "verified").sum()),
            "leakage_groups": int(frame["split_group_id"].nunique()),
            "usable_leakage_groups": int(usable["split_group_id"].nunique()),
            "prompt_families": int(frame["prompt_family_id"].nunique()),
            "categories": int(frame["prompt_category"].nunique()),
            "experiments": sorted(str(x) for x in frame["experiment_id"].dropna().unique()),
            "models": sorted(str(x) for x in frame["requested_model"].dropna().unique()),
            "efforts": sorted(str(x) for x in frame["requested_effort"].fillna("unset").unique()),
            "largest_group_share": round(float(group_sizes.iloc[0] / len(frame)), 3),
        },
        "usable_output_tokens": _dist(usable["actual_output_tokens"]),
        "by_category": by_category,
        "split_simulation": simulations,
        "coverage_gaps": gaps,
    }


def render_readiness_markdown(report: dict[str, Any]) -> str:
    lines = ["# Training readiness", "", f"**Status:** `{report['status']}`", ""]
    if "totals" not in report:
        lines.append(report.get("detail", "No validated observations in the store."))
        return "\n".join(lines) + "\n"
    totals = report["totals"]
    lines += [
        f"- Eligible real rows: {totals['eligible_real_rows']} "
        f"(usable {totals['usable_rows']}, censored {totals['censored_rows']}, "
        f"non-verified {totals['non_verified_rows']})",
        f"- Leakage groups: {totals['leakage_groups']} "
        f"(with usable rows: {totals['usable_leakage_groups']}); "
        f"prompt families: {totals['prompt_families']}; "
        f"largest group holds {totals['largest_group_share']:.1%} of rows",
        f"- Models: {', '.join(totals['models'])}; efforts: {', '.join(totals['efforts'])}",
        f"- Experiments: {', '.join(totals['experiments'])}",
        "",
        "## Split simulation (exact `prepare-training` assignment)",
        "",
    ]
    for name, sim in report["split_simulation"].items():
        ratios = ", ".join(f"{k} {v:.2f}" for k, v in sim["ratios"].items())
        lines += [
            f"### `{name}` ratios ({ratios})",
            "",
            f"Implied minimum usable rows: **{sim['implied_minimum_usable']}**. "
            f"Meets trainer minimums: **{'yes' if sim['meets_trainer_minimums'] else 'no'}**."
            + (
                f" Estimated additional usable rows needed: "
                f"**{sim['estimated_additional_usable_rows']}**."
                if not sim["meets_trainer_minimums"]
                else ""
            ),
            "",
            "| Split | Rows | Usable | Groups | Usable categories | Minimum | Shortfall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for split, info in sim["splits"].items():
            lines.append(
                f"| {split} | {info['rows']} | {info['usable']} | {info['groups']} | "
                f"{info['usable_categories']} | {info['minimum_usable'] or '-'} | "
                f"{info['shortfall']} |"
            )
        lines.append("")
    lines += [
        "## By category",
        "",
        "| Category | Rows | Usable | Censored | Censor rate | Families | Usable groups "
        "| Caps | Usable output p50 / p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for category, info in report["by_category"].items():
        dist = info["usable_output_tokens"]
        quant = f"{dist['p50']:.0f} / {dist['p90']:.0f}" if dist else "-"
        caps = ",".join(str(c) for c in info["max_output_tokens_levels"])
        lines.append(
            f"| {category} | {info['rows']} | {info['usable']} | {info['censored']} | "
            f"{info['censor_rate']:.0%} | {info['families']} | {info['usable_groups']} | "
            f"{caps} | {quant} |"
        )
    lines += ["", "## Coverage gaps", ""]
    lines += [f"- {gap}" for gap in report["coverage_gaps"]] or ["- none detected"]
    lines += ["", f"_{report['note']}_", ""]
    return "\n".join(lines)
