"""Modelling frame built from the canonical store (point-in-time safe).

One row per validated observation of one label origin, with:

* model features recomputed from the **exact stored request** (``plan.jsonl``)
  by :func:`seshat.modeling.features.model_features` -- the same function the
  API uses, so training and serving features cannot drift apart;
* labels (``actual_output_tokens``), censoring (``censored``, ``cap``) and
  realized-usage fields for cost evaluation;
* leakage groups (union-find over prompt family, prompt hash, conversation);
* a deterministic **protected-test** flag: a leakage group is test data iff the
  smallest SHA-256 bucket among its prompt families is below
  ``PROTECTED_TEST_FRACTION``. Membership depends only on family ids, so it is
  stable as data accumulates and every model is compared on the same rows;
* design labels (``prompt_category``, ``experiment_id``) for *slicing only*.

Real and synthetic (fixture) labels are never mixed: ``label_origin`` selects one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from seshat.data.io import read_parquet
from seshat.data.storage import StorageLayout
from seshat.data.training import DEFAULT_GROUP_BY, leakage_groups
from seshat.modeling.features import model_features

PROTECTED_TEST_FRACTION = 0.2
PROTECTED_TEST_RULE = "min_family_sha256_bucket<0.2/v1"


def family_bucket(family_id: str) -> float:
    digest = hashlib.sha256(f"seshat-protected-test|{family_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def stored_requests(root: str | Path) -> dict[str, dict[str, Any]]:
    """sample_id -> planned sample (including the exact request) from raw plans."""
    plans: dict[str, dict[str, Any]] = {}
    for path in sorted((Path(root) / "raw").glob("*/plan.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                plans[str(row["sample_id"])] = row
    return plans


def load_modeling_frame(root: str | Path, *, label_origin: str = "real") -> pd.DataFrame:
    from seshat.data.schemas.request import LLMRequest

    layout = StorageLayout(root)
    observations = layout.read_frame("observations")
    decisions_path = layout.validation / "decisions.parquet"
    if observations.empty or not decisions_path.exists():
        raise ValueError("Need validated observations in the store")
    decisions = read_parquet(decisions_path)[
        ["observation_id", "eligible_for_training", "data_origin"]
    ]
    obs = observations.merge(decisions, on="observation_id", how="inner")
    obs = obs[
        obs["eligible_for_training"]
        & (obs["data_origin"] == label_origin)
        & obs["actual_output_tokens"].notna()
    ]
    if label_origin == "real":
        obs = obs[obs["source_type"] == "controlled_experiment"]
    requests = layout.read_frame("requests").drop_duplicates("sample_id")
    request_cols = [
        "sample_id",
        "experiment_id",
        "prompt_family_id",
        "prompt_hash",
        "conversation_hash",
        "conversation_id",
        "prompt_category",
        "prompt_subcategory",
        "prompt_word_count",
        "requested_effort",
        "max_output_tokens",
        "requested_output_format",
        "tool_count",
        "context_character_count",
        "timestamp_utc",
        "design_cell_id",
    ]
    frame = obs.merge(requests[request_cols], on="sample_id", how="inner", validate="many_to_one")
    preflight = layout.read_frame("preflight")
    if not preflight.empty:
        pre = preflight[preflight["preflight_input_tokens"].notna()].copy()
        pre["_rank"] = (pre["preflight_method"] != "provider_native").astype(int)
        pre = pre.sort_values(["sample_id", "_rank"]).drop_duplicates("sample_id")
        frame = frame.merge(
            pre[["sample_id", "preflight_input_tokens", "preflight_method"]],
            on="sample_id",
            how="left",
        )
    plans = stored_requests(root)
    feature_rows = []
    for sample_id in frame["sample_id"].astype(str):
        plan = plans.get(sample_id)
        if plan is None:
            raise ValueError(f"No stored request for sample {sample_id}; cannot build features")
        feature_rows.append(model_features(LLMRequest.model_validate(plan["request"])))
    features = pd.DataFrame(feature_rows, index=frame.index)
    # Features recomputed from the stored request take precedence over same-named columns.
    frame = pd.concat([frame.drop(columns=[c for c in features if c in frame]), features], axis=1)
    frame["censored"] = frame["is_truncated"].fillna(False).astype(bool) | (
        frame["label_quality"] == "censored"
    )
    frame["cap"] = frame["max_output_tokens"].astype(float)
    frame["label_origin"] = label_origin
    frame = frame.reset_index(drop=True)
    frame["split_group_id"] = leakage_groups(frame, DEFAULT_GROUP_BY)
    buckets = frame.groupby("split_group_id")["prompt_family_id"].agg(
        lambda fams: min(family_bucket(str(f)) for f in fams)
    )
    frame["protected_test"] = frame["split_group_id"].map(buckets) < PROTECTED_TEST_FRACTION
    return pd.DataFrame(frame)


def dataset_version(frame: pd.DataFrame) -> str:
    ids = "\n".join(sorted(frame["observation_id"].astype(str)))
    return "ds-" + hashlib.sha256(ids.encode()).hexdigest()[:12]
