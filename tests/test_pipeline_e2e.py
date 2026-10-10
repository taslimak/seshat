"""End-to-end mock run of the data factory, mirroring the CI smoke test."""

import json
from pathlib import Path

import pandas as pd

from seshat.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "experiments" / "pilot_001.json"
PROMPTS = REPO_ROOT / "prompts" / "hand_authored.jsonl"


def test_mock_pilot_produces_a_complete_release(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.parquet"
    raw = tmp_path / "raw.jsonl"
    normalized = tmp_path / "normalized.parquet"
    validated = tmp_path / "validated.parquet"
    features = tmp_path / "features.parquet"
    splits = tmp_path / "splits.parquet"
    releases = tmp_path / "releases"

    steps = [
        ["build-manifest", "--config", str(CONFIG), "--output", str(manifest)],
        ["run", "--manifest", str(manifest), "--prompts", str(PROMPTS), "--output", str(raw)],
        ["normalize", "--input", str(raw), "--output", str(normalized)],
        ["validate", "--input", str(normalized), "--output", str(validated)],
        [
            "features",
            "--input", str(validated),
            "--output", str(features),
            "--label-origin", "synthetic",
        ],
        ["split", "--input", str(features), "--output", str(splits)],
        [
            "release",
            "--features", str(features),
            "--splits", str(splits),
            "--output-dir", str(releases),
            "--version", "test",
        ],
    ]
    for argv in steps:
        assert main(argv) == 0, argv

    decisions = pd.read_parquet(validated)["quality_decision"]
    assert (decisions == "PASS").all(), decisions.value_counts().to_dict()
    assert len(pd.read_parquet(features)) == len(decisions) == 144

    assert (pd.read_parquet(features)["label_origin"] == "synthetic").all()
    # Mock outcomes are synthetic: the default (real-label) build must refuse them.
    refused = ["features", "--input", str(validated), "--output", str(tmp_path / "x.parquet")]
    assert main(refused) == 2

    release_dir = releases / "dataset_vtest_synthetic"
    family_splits: dict[str, str] = {}
    for split_name in ("train", "validation", "calibration", "test"):
        frame = pd.read_parquet(release_dir / f"{split_name}.parquet")
        assert not frame.empty, f"{split_name} split is empty"
        for family_id in frame["prompt_family_id"].unique():
            assert family_splits.setdefault(family_id, split_name) == split_name
    manifest = json.loads((release_dir / "manifest.json").read_text())
    assert manifest["label_origin"] == "synthetic"
    assert manifest["source_experiments"] == ["pilot_001"]
