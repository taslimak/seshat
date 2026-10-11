"""Safe configuration generation for the calibration campaign."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.prepare_calibration_campaign import campaign_config


def test_campaign_is_bounded_and_rotated() -> None:
    first = campaign_config(
        provider="openai", model="gpt-6.1-sol", run_id="987654", root="/tmp/seshat"
    )
    second = campaign_config(
        provider="openai", model="gpt-6.1-sol", run_id="987655", root="/tmp/seshat"
    )
    assert first["budget"] == {"max_requests": 180, "max_cost_usd": "2.00"}
    assert first["targets"][0]["max_output_tokens"] == [512]
    assert first["execution"]["max_retries"] == 0
    assert first["seed"] != second["seed"]
    assert first["experiment_id"] != second["experiment_id"]


def test_refuses_unverified_model_and_malformed_run_id() -> None:
    with pytest.raises(ValueError, match="not in the catalog"):
        campaign_config(provider="openai", model="unknown", run_id="123", root="/tmp/seshat")
    with pytest.raises(ValueError, match="numeric"):
        campaign_config(
            provider="openai", model="gpt-6.1-sol", run_id="../../bad", root="/tmp/seshat"
        )


def test_workflow_runs_campaign_as_importable_module() -> None:
    workflow = Path(".github/workflows/manual-calibration-batch.yml").read_text()
    assert "python -m scripts.prepare_calibration_campaign" in workflow
    result = subprocess.run(
        [sys.executable, "-m", "scripts.prepare_calibration_campaign", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
