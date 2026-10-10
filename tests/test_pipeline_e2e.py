"""End-to-end mock run of the data factory, mirroring the CI smoke test."""

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
        ["features", "--input", str(validated), "--output", str(features)],
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

    release_dir = releases / "dataset_vtest"
    family_splits: dict[str, str] = {}
    for split_name in ("train", "validation", "calibration", "test"):
        frame = pd.read_parquet(release_dir / f"{split_name}.parquet")
        assert not frame.empty, f"{split_name} split is empty"
        for family_id in frame["prompt_family_id"].unique():
            assert family_splits.setdefault(family_id, split_name) == split_name
    assert (release_dir / "manifest.json").exists()
