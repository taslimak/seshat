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


def test_usage_rejects_impossible_total() -> None:
    with pytest.raises(ValidationError):
        NormalizedUsage(input_tokens=100, output_tokens=50, total_tokens=120)
