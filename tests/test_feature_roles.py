import pytest

from seshat.data.feature_registry import assert_training_features


def test_request_time_features_are_allowed() -> None:
    assert_training_features(["provider", "model_requested", "word_count", "effort"])


def test_leakage_guard_blocks_target_and_post_inference_fields() -> None:
    with pytest.raises(ValueError, match="Leakage guard"):
        assert_training_features(["word_count", "output_tokens", "latency_ms"])
