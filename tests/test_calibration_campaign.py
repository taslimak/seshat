"""Safe configuration generation for the calibration campaign."""

import pytest

from scripts.prepare_calibration_campaign import campaign_config


def test_campaign_is_bounded_and_rotated() -> None:
    first = campaign_config(
        provider="openai", model="gpt-6.1-sol", run_id="987654", root="/tmp/seshat"
    )
    second = campaign_config(
        provider="openai", model="gpt-6.1-sol", run_id="987655", root="/tmp/seshat"
    )
    assert first["budget"] == {"max_requests": 40, "max_cost_usd": "0.75"}
    assert first["targets"][0]["max_output_tokens"] == [512]
    assert first["execution"]["max_retries"] == 0
    assert first["seed"] != second["seed"]
    assert first["experiment_id"] != second["experiment_id"]


def test_refuses_unverified_model_and_malformed_run_id() -> None:
    with pytest.raises(ValueError, match="Unverified"):
        campaign_config(
            provider="openai", model="unknown", run_id="123", root="/tmp/seshat"
        )
    with pytest.raises(ValueError, match="numeric"):
        campaign_config(
            provider="openai", model="gpt-6.1-sol",
            run_id="../../bad", root="/tmp/seshat"
        )
