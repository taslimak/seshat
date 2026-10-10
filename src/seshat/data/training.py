"""Training layer: point-in-time-correct feature/label datasets with leakage-safe splits.

``prepare_training_dataset`` joins, per eligible observation:

* request-time features from ``requests`` (as recorded at request time),
* the preflight count measured *before dispatch* (provider-native preferred),
* observed labels from ``observations`` (only rows validation marked eligible).

Then it:

1. applies the label-origin policy (``real`` by default; ``synthetic`` only on
   request, never mixed) and the source policy: real labels come from controlled
   experiments only, unless consented telemetry / authorized log imports are
   included explicitly (production data never becomes training data by default);
2. builds leakage groups with union-find over prompt family, prompt content hash,
   conversation context hash and conversation id (plus template id if requested),
   so repeats, shared conversations and identical prompts never straddle splits;
3. optionally holds out the most recent groups as a temporal test set;
4. assigns the remaining groups to train / validation / calibration (/ test),
   stratified by prompt category;
5. writes an immutable versioned dataset with manifest, schema and split stats.

Model training is out of scope; this produces its inputs.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from seshat.data.feature_registry import (
    TARGET_COLUMNS,
    TRAINING_FEATURES,
    assert_training_features,
    feature_registry,
)
from seshat.data.features.extract import FEATURE_SET_VERSION
from seshat.data.io import read_parquet, write_json, write_parquet
from seshat.data.provenance import PIPELINE_VERSION
from seshat.data.schemas.enums import SCHEMA_VERSION, SourceType
from seshat.data.split import DEFAULT_RATIOS, assign_grouped_splits
from seshat.data.storage import StorageLayout

LabelOrigin = Literal["real", "synthetic"]
# Real-label sources that enter training only when included explicitly.
OPTIONAL_REAL_SOURCES = frozenset(
    {SourceType.CONSENTED_API_TELEMETRY.value, SourceType.AUTHORIZED_LOG_IMPORT.value}
)
DEFAULT_GROUP_BY = ("prompt_family_id", "prompt_hash", "conversation_hash", "conversation_id")
SYNTHETIC_INTENDED_USE = (
    "Pipeline testing only. Labels come from test fixtures and must not be used to train "
    "or evaluate predictors of real provider usage, nor in performance reporting."
)
REAL_INTENDED_USE = "Training and evaluation of token/cost predictors on real observed usage."


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def leakage_groups(frame: pd.DataFrame, group_by: tuple[str, ...]) -> pd.Series:
    """Connected components of rows sharing any non-null value in ``group_by``."""
    uf = _UnionFind()
    for index, row in frame.iterrows():
        node = f"row:{index}"
        uf.find(node)
        for column in group_by:
            value = row.get(column)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                uf.union(node, f"{column}:{value}")
    roots = {index: uf.find(f"row:{index}") for index in frame.index}
    return pd.Series(
        {i: "grp_" + hashlib.sha256(r.encode()).hexdigest()[:16] for i, r in roots.items()},
        name="split_group_id",
    )


def _join(layout: StorageLayout) -> pd.DataFrame:
    requests = layout.read_frame("requests")
    preflight = layout.read_frame("preflight")
    observations = layout.read_frame("observations")
    decisions_path = layout.validation / "decisions.parquet"
    if not decisions_path.exists():
        raise ValueError("Run `validate` before `prepare-training` (no validation decisions found)")
    decisions = read_parquet(decisions_path)
    if observations.empty:
        raise ValueError("No observations in the normalized layer")
    eligible = decisions[decisions["eligible_for_training"]][
        ["observation_id", "data_origin", "decision"]
    ]
    obs = observations.merge(eligible, on="observation_id", how="inner")
    req = requests.drop_duplicates("sample_id")
    req = req.assign(
        role_sequence_str=req["role_sequence"].map(
            lambda v: ">".join(v) if v is not None else None
        ),
        enabled_tool_types_str=req["enabled_tool_types"].map(
            lambda v: ",".join(v) if v is not None else None
        ),
        modality_types_str=req["modality_types"].map(
            lambda v: ",".join(v) if v is not None else None
        ),
    )
    if not preflight.empty:
        pre = preflight[preflight["preflight_input_tokens"].notna()].copy()
        pre["_rank"] = (pre["preflight_method"] != "provider_native").astype(int)
        pre = pre.sort_values(["sample_id", "_rank"]).drop_duplicates("sample_id")
        pre = pre[
            [
                "sample_id",
                "preflight_input_tokens",
                "preflight_method",
                "preflight_complete",
                "preflight_timestamp_utc",
            ]
        ]
    else:
        pre = pd.DataFrame(
            columns=[
                "sample_id",
                "preflight_input_tokens",
                "preflight_method",
                "preflight_complete",
                "preflight_timestamp_utc",
            ]
        )
    request_cols = [
        c
        for c in req.columns
        if c
        not in {
            "provider",
            "requested_model",
            "source_type",
            "dataset_origin",
            "consent_status",
            "run_id",
            "request_id",
            "parent_request_id",
            "schema_version",
            "retain_until",
        }
    ]
    joined = obs.merge(req[request_cols], on="sample_id", how="inner", validate="many_to_one")
    joined = joined.merge(pre, on="sample_id", how="left", validate="many_to_one")
    # Point-in-time: features must predate the observation they predict.
    late = joined["timestamp_utc"] > joined["collected_at_utc"]
    late_preflight = joined["preflight_timestamp_utc"].notna() & (
        joined["preflight_timestamp_utc"] > joined["collected_at_utc"]
    )
    if late.any() or late_preflight.any():
        raise ValueError(
            f"{int(late.sum() + late_preflight.sum())} rows have request-time data "
            "after the observation"
        )
    return pd.DataFrame(joined)


def _stats(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(frame)),
        "groups": int(frame["split_group_id"].nunique()) if len(frame) else 0,
        "censored": int(frame["target_is_censored"].sum()) if len(frame) else 0,
        "by_provider": {
            str(k): int(v) for k, v in frame["provider"].value_counts().sort_index().items()
        },
        "by_category": {
            str(k): int(v) for k, v in frame["prompt_category"].value_counts().sort_index().items()
        },
    }
    for target in (
        "actual_input_tokens",
        "actual_output_tokens",
        "actual_reasoning_tokens",
        "actual_total_tokens",
    ):
        values = pd.to_numeric(frame[target], errors="coerce").dropna()
        out[target] = (
            None
            if values.empty
            else {
                "mean": round(float(values.mean()), 2),
                "p50": float(values.median()),
                "p90": float(values.quantile(0.9)),
                "missing": int(frame[target].isna().sum()),
            }
        )
    return out


def _git_revision() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": sha, "dirty": dirty}


def prepare_training_dataset(
    root: str | Path,
    version: str,
    *,
    label_origin: LabelOrigin = "real",
    seed: int = 1729,
    ratios: dict[str, float] | None = None,
    temporal_holdout_fraction: float = 0.0,
    group_by: tuple[str, ...] = DEFAULT_GROUP_BY,
    include_sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build ``<root>/training/dataset_v<version>[_synthetic]/``.

    ``include_sources`` adds ``consented_api_telemetry`` and/or
    ``authorized_log_import`` rows to a real-label dataset; by default only
    ``controlled_experiment`` labels are used.
    """
    if not 0.0 <= temporal_holdout_fraction < 1.0:
        raise ValueError("temporal_holdout_fraction must be in [0, 1)")
    layout = StorageLayout(root)
    suffix = "_synthetic" if label_origin == "synthetic" else ""
    target_dir = layout.training / f"dataset_v{version}{suffix}"
    if target_dir.exists():
        raise FileExistsError(f"{target_dir} already exists; training datasets are immutable")
    joined = _join(layout)
    unknown = set(include_sources) - OPTIONAL_REAL_SOURCES
    if unknown:
        raise ValueError(
            f"include_sources must be among {sorted(OPTIONAL_REAL_SOURCES)}, got {sorted(unknown)}"
        )
    selected = joined[joined["data_origin"] == label_origin].copy()
    if label_origin == "real":
        allowed = {SourceType.CONTROLLED_EXPERIMENT.value, *include_sources}
        excluded_sources = (
            selected.loc[~selected["source_type"].isin(allowed), "source_type"]
            .value_counts()
            .to_dict()
        )
        selected = selected[selected["source_type"].isin(allowed)]
        if selected.empty and excluded_sources:
            raise ValueError(
                f"No eligible controlled-experiment labels; excluded by source policy: "
                f"{excluded_sources}. Pass --include-source to use them."
            )
    else:
        excluded_sources = {}
    if selected.empty:
        available = sorted(joined["data_origin"].unique())
        hint = (
            " Pass --label-origin synthetic for a pipeline-test dataset."
            if label_origin == "real" and "synthetic" in available
            else ""
        )
        raise ValueError(
            f"No eligible rows with {label_origin} labels (available: {available}).{hint}"
        )
    selected = selected.reset_index(drop=True)
    selected["split_group_id"] = leakage_groups(selected, group_by)
    selected["target_is_censored"] = selected["is_truncated"].fillna(False).astype(bool) | (
        selected["label_quality"] == "censored"
    )
    selected["label_origin"] = label_origin
    features = [c for c in TRAINING_FEATURES if c in selected.columns]
    assert_training_features(features)
    identifiers = [
        "sample_id",
        "observation_id",
        "prompt_id",
        "prompt_family_id",
        "prompt_template_id",
        "split_group_id",
        "experiment_id",
        "design_cell_id",
        "repetition_index",
    ]
    meta = [
        "timestamp_utc",
        "source_type",
        "label_origin",
        "label_quality",
        "target_is_censored",
        "decision",
    ]
    targets = [c for c in TARGET_COLUMNS if c in selected.columns]
    dataset = selected[identifiers + meta + features + targets].copy()

    test_groups: set[str] = set()
    if temporal_holdout_fraction > 0:
        # Newest groups (by first observation) become the temporal test set.
        first_seen = (
            dataset.groupby("split_group_id")["timestamp_utc"].min().sort_values(ascending=False)
        )
        sizes = dataset.groupby("split_group_id").size()
        target_rows = round(len(dataset) * temporal_holdout_fraction)
        taken = 0
        for group in first_seen.index:
            if taken >= target_rows:
                break
            test_groups.add(group)
            taken += int(sizes[group])
    remaining = dataset[~dataset["split_group_id"].isin(test_groups)]
    split_ratios = dict(ratios or DEFAULT_RATIOS)
    if test_groups:
        split_ratios.pop("test", None)
        total = sum(split_ratios.values())
        split_ratios = {k: v / total for k, v in split_ratios.items()}
    assignments = assign_grouped_splits(
        remaining,
        seed=seed,
        ratios=split_ratios,
        group_column="split_group_id",
        stratify_column="prompt_category",
    )[["split_group_id", "split"]]
    split_map = dict(zip(assignments["split_group_id"], assignments["split"], strict=True))
    split_map.update({g: "test" for g in test_groups})
    dataset["split"] = dataset["split_group_id"].map(split_map)
    if dataset["split"].isna().any():
        raise AssertionError("unassigned rows after splitting")
    if dataset.groupby("split_group_id")["split"].nunique().max() > 1:
        raise AssertionError("a leakage group spans several splits")

    target_dir.mkdir(parents=True)
    files: dict[str, str] = {}
    split_stats: dict[str, Any] = {}
    for split in ("train", "validation", "calibration", "test"):
        part = dataset[dataset["split"] == split].drop(columns=["split"])
        path = target_dir / f"{split}.parquet"
        write_parquet(path, part)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        split_stats[split] = _stats(part)
    registry = feature_registry()
    schema = {
        "schema_version": SCHEMA_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "features": features,
        "targets": targets,
        "columns": {
            c: {
                "dtype": str(dataset[c].dtype),
                "role": registry[c].phase.value if c in registry else "IDENTIFIER",
            }
            for c in dataset.columns
            if c != "split"
        },
    }
    write_json(target_dir / "schema.json", schema)
    write_json(target_dir / "split_stats.json", split_stats)
    manifest = {
        "dataset_version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "label_origin": label_origin,
        "source_types": sorted(str(x) for x in dataset["source_type"].unique()),
        "excluded_by_source_policy": {str(k): int(v) for k, v in excluded_sources.items()},
        "intended_use": SYNTHETIC_INTENDED_USE
        if label_origin == "synthetic"
        else REAL_INTENDED_USE,
        "rows": int(len(dataset)),
        "split_algorithm": {
            "method": (
                "union-find leakage groups; category-stratified group assignment "
                "by within-category position"
            ),
            "group_by": list(group_by),
            "seed": seed,
            "ratios": split_ratios,
            "temporal_holdout_fraction": temporal_holdout_fraction,
            "temporal_test_groups": len(test_groups),
        },
        "split_rows": {k: v["rows"] for k, v in split_stats.items()},
        "censored_rows": int(dataset["target_is_censored"].sum()),
        "source_experiments": sorted(str(x) for x in dataset["experiment_id"].dropna().unique()),
        "providers": sorted(dataset["provider"].unique().tolist()),
        "models": sorted(dataset["requested_model"].unique().tolist()),
        "code_revision": _git_revision(),
        "files": files,
    }
    write_json(target_dir / "manifest.json", manifest)
    return manifest
