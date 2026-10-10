import pytest
from pydantic import ValidationError

from seshat.data.models import NormalizedUsage, PromptRecord


def test_prompt_requires_stable_family_id() -> None:
    with pytest.raises(ValidationError):
        PromptRecord(
            prompt_id="p1",
            prompt_family_id="",
            category="coding",
            text="hello",
            source="fixture",
        )


def test_usage_preserves_impossible_total_for_downstream_quarantine() -> None:
    # Cross-field plausibility is a data-quality rule (seshat.data.validate), not a
    # parse error, so anomalous provider usage is kept and quarantined, not lost.
    usage = NormalizedUsage(input_tokens=100, output_tokens=50, total_tokens=120)
    assert usage.total_tokens == 120


def test_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        NormalizedUsage(output_tokens=-1)
