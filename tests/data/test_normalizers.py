import pytest

from seshat.data.schemas import LabelQuality, ResponseStatus
from seshat.providers.normalization import normalize_anthropic, normalize_gemini, normalize_openai
from seshat.providers.payloads import parse_anthropic, parse_gemini, parse_openai
from tests.data.conftest import provider_fixture


def test_openai_cached_and_reasoning_are_subsets() -> None:
    usage = normalize_openai(provider_fixture("openai_reasoning_cached.json")["usage"])
    assert usage.actual_input_tokens == 2048
    assert usage.actual_cached_input_tokens == 1024
    assert usage.actual_uncached_input_tokens == 1024
    assert usage.actual_output_tokens == 1186  # includes reasoning
    assert usage.actual_reasoning_tokens == 1024
    assert usage.actual_visible_output_tokens == 162
    assert usage.reasoning_included_in_output is True
    assert usage.actual_total_tokens == usage.provider_total_tokens == 3234
    assert usage.actual_cache_creation_tokens is None  # not reported => unknown, not 0
    assert usage.issues == []


def test_openai_contradictions_are_flagged_not_repaired() -> None:
    usage = normalize_openai(
        {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 9},
            "total_tokens": 99,
        }
    )
    assert set(usage.issues) == {
        "cached_tokens_exceed_input_tokens",
        "reasoning_tokens_exceed_output_tokens",
        "total_tokens_mismatch",
    }
    assert usage.actual_input_tokens == 10 and usage.provider_total_tokens == 99
    assert usage.label_quality(is_truncated=False, completed=True) == LabelQuality.INCONSISTENT


def test_anthropic_total_input_reconstructed_from_cache_fields() -> None:
    usage = normalize_anthropic(provider_fixture("anthropic_cache_thinking.json")["usage"])
    # Documented example: 1800 (read) + 248 (write) + 2048 (after breakpoint) = 4096
    assert usage.actual_input_tokens == 4096
    assert usage.actual_uncached_input_tokens == 2048
    assert usage.actual_cached_input_tokens == 1800
    assert usage.actual_cache_creation_tokens == 248
    assert usage.actual_reasoning_tokens == 200
    assert usage.actual_visible_output_tokens == 303
    assert usage.provider_total_tokens is None  # Anthropic reports no total
    assert usage.billing_dimensions["cache_write_1h_tokens"] == 100


def test_anthropic_compaction_sums_iterations() -> None:
    usage = normalize_anthropic(provider_fixture("anthropic_compaction.json")["usage"])
    assert usage.actual_input_tokens == 203_000  # top-level alone would undercount (23,000)
    assert usage.actual_output_tokens == 4_500
    assert usage.usage_iteration_count == 2
    assert usage.billing_dimensions["compaction_iterations"] == 1
    assert usage.issues == []


def test_anthropic_breakdown_mismatch_is_an_issue() -> None:
    usage = normalize_anthropic(
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 50,
            "cache_read_input_tokens": 0,
            "output_tokens": 5,
            "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 10},
        }
    )
    assert "cache_creation_breakdown_mismatch" in usage.issues


def test_gemini_thoughts_are_not_inside_candidates() -> None:
    usage = normalize_gemini(provider_fixture("gemini_thinking_cached.json")["usage_metadata"])
    assert usage.actual_input_tokens == 500  # includes cached content
    assert usage.actual_uncached_input_tokens == 200
    assert usage.actual_output_tokens == 520  # candidates 120 + thoughts 400
    assert usage.actual_visible_output_tokens == 120
    assert usage.reasoning_included_in_output is False
    assert usage.actual_total_tokens == usage.provider_total_tokens == 1020
    assert usage.billing_dimensions["total_relationship"] == "prompt+candidates+thoughts"


def test_gemini_tool_use_prompt_kept_separate() -> None:
    usage = normalize_gemini(provider_fixture("gemini_tool_use_total.json")["usage_metadata"])
    assert usage.actual_tool_input_tokens == 50
    assert usage.actual_input_tokens == 200  # not added to input
    assert usage.actual_reasoning_tokens is None  # absent => unknown
    assert usage.actual_total_tokens == 215
    assert (
        usage.billing_dimensions["total_relationship"]
        == "prompt+candidates+thoughts+tool_use_prompt"
    )
    assert usage.issues == []


@pytest.mark.parametrize("normalizer", [normalize_openai, normalize_anthropic, normalize_gemini])
def test_missing_usage_is_unavailable_not_zero(normalizer) -> None:  # type: ignore[no-untyped-def]
    usage = normalizer(None)
    assert usage.actual_input_tokens is None and usage.actual_output_tokens is None
    assert usage.label_quality(is_truncated=False, completed=True) == LabelQuality.UNAVAILABLE


def test_non_integer_usage_raises() -> None:
    with pytest.raises(ValueError):
        normalize_openai({"input_tokens": "12"})


def test_negative_values_are_flagged() -> None:
    usage = normalize_openai({"input_tokens": -1, "output_tokens": 3})
    assert "negative_actual_input_tokens" in usage.issues


def test_label_quality_rules() -> None:
    complete = normalize_openai(provider_fixture("openai_reasoning_cached.json")["usage"])
    assert complete.label_quality(is_truncated=False, completed=True) == LabelQuality.VERIFIED
    assert complete.label_quality(is_truncated=True, completed=False) == LabelQuality.CENSORED
    partial = normalize_gemini(provider_fixture("gemini_tool_use_total.json")["usage_metadata"])
    assert partial.label_quality(is_truncated=False, completed=True) == LabelQuality.PARTIAL


def test_payload_parsers_status_truncation_tools_and_no_text() -> None:
    oa = parse_openai(provider_fixture("openai_truncated.json"), latency_ms=10)
    assert (
        oa.status == ResponseStatus.INCOMPLETE
        and oa.is_truncated
        and oa.stop_reason == "max_output_tokens"
    )
    tools = parse_openai(provider_fixture("openai_tool_call.json"), latency_ms=1)
    assert tools.client_tool_call_count == 1
    an = parse_anthropic(provider_fixture("anthropic_max_tokens_tool.json"), latency_ms=1)
    assert (
        an.is_truncated
        and an.status == ResponseStatus.INCOMPLETE
        and an.client_tool_call_count == 1
    )
    ge = parse_gemini(provider_fixture("gemini_thinking_cached.json"), latency_ms=1)
    assert ge.resolved_model == "gemini-3.8-flash-001" and ge.output_char_count == len(
        "Sample visible answer."
    )
    for response in (oa, an, ge):
        assert "content" not in response.raw_response and "output" not in response.raw_response
        assert "candidates" not in response.raw_response
    gm = parse_gemini(provider_fixture("gemini_max_tokens.json"), latency_ms=1)
    assert gm.is_truncated
