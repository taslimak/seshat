import pandas as pd

from seshat.data.validate import validate_dataframe


def _row() -> dict[str, object]:
    return {
        "observation_id": "obs1",
        "run_id": "run1",
        "provider": "mock",
        "model_requested": "mock-v1",
        "prompt_id": "p1",
        "status": "success",
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "is_truncated": False,
        "is_refusal": False,
    }


def test_valid_record_passes() -> None:
    result = validate_dataframe(pd.DataFrame([_row()]))
    assert result.loc[0, "quality_decision"] == "PASS"


def test_truncated_record_is_warning() -> None:
    row = _row()
    row["is_truncated"] = True
    result = validate_dataframe(pd.DataFrame([row]))
    assert result.loc[0, "quality_decision"] == "WARNING"
    assert "right_censored_output" in result.loc[0, "quality_flags"]


def test_duplicate_run_is_rejected() -> None:
    result = validate_dataframe(pd.DataFrame([_row(), _row()]))
    assert set(result["quality_decision"]) == {"REJECT"}
