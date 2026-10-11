from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from seshat.data.pricing import PriceTable, compute_cost, max_cost_estimate
from seshat.data.privacy import (
    PromptHashScheme,
    RetentionPolicy,
    detect_sensitive,
    hash_prompt,
    redact_text,
)
from seshat.data.schemas import SourceType
from seshat.providers.normalization import normalize_anthropic, normalize_gemini, normalize_openai
from tests.data.conftest import provider_fixture

TABLE = PriceTable.load()
DAY = date(2026, 10, 10)


def test_anthropic_cost_uses_cache_write_ttls() -> None:
    usage = normalize_anthropic(provider_fixture("anthropic_cache_thinking.json")["usage"])
    cost = compute_cost("anthropic", TABLE.lookup("anthropic", "claude-opus-5-5", DAY), usage)
    expected = (
        Decimal(2048) * 4
        + Decimal(1800) * Decimal("0.20")
        + Decimal(148) * 5
        + Decimal(100) * 8
        + Decimal(503) * 20
    ) / 1_000_000
    assert cost.calculated_cost_usd == expected.quantize(Decimal("0.000000000001"))


def test_anthropic_cache_write_without_breakdown_is_flagged_as_assumed() -> None:
    usage = normalize_anthropic(
        {
            "input_tokens": 10,
            "cache_creation_input_tokens": 1000,
            "cache_read_input_tokens": 0,
            "output_tokens": 10,
        }
    )
    cost = compute_cost("anthropic", TABLE.lookup("anthropic", "claude-sonnet-5-5", DAY), usage)
    assert cost.dimensions["cache_write_ttl_assumed"] == "5m"


def test_openai_cached_tokens_at_cache_rate_and_long_context_tier() -> None:
    usage = normalize_openai(provider_fixture("openai_reasoning_cached.json")["usage"])
    entry = TABLE.lookup("openai", "gpt-6-sol", DAY)
    cost = compute_cost("openai", entry, usage)
    assert cost.calculated_cost_usd == (
        (Decimal(1024) * 2 + Decimal(1024) * Decimal("0.20") + Decimal(1186) * 10) / 1_000_000
    ).quantize(Decimal("0.000000000001"))
    long = normalize_openai(
        {
            "input_tokens": 300_000,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 300_010,
        }
    )
    long_cost = compute_cost("openai", entry, long)
    assert long_cost.dimensions["long_context_tier"] is True
    assert long_cost.rates is not None and long_cost.rates.input == Decimal("4.00")


def test_gemini_output_includes_thoughts_and_dated_prices() -> None:
    usage = normalize_gemini(provider_fixture("gemini_thinking_cached.json")["usage_metadata"])
    before = compute_cost(
        "gemini", TABLE.lookup("gemini", "gemini-3.8-flash", date(2026, 12, 31)), usage
    )
    after = compute_cost(
        "gemini", TABLE.lookup("gemini", "gemini-3.8-flash", date(2027, 1, 1)), usage
    )
    expected = (
        Decimal(200) * Decimal("0.75")
        + Decimal(300) * Decimal("0.075")
        + Decimal(520) * Decimal("3.75")
    ) / 1_000_000
    assert before.calculated_cost_usd == expected.quantize(Decimal("0.000000000001"))
    assert after.calculated_cost_usd == (expected * 2).quantize(Decimal("0.000000000001"))


def test_unpriced_model_and_incomplete_usage_give_no_cost() -> None:
    assert TABLE.lookup("anthropic", "unknown-model", DAY) is None
    assert (
        compute_cost(
            "openai", None, normalize_openai({"input_tokens": 1, "output_tokens": 1})
        ).calculated_cost_usd
        is None
    )
    entry = TABLE.lookup("openai", "gpt-6-luna", DAY)
    assert compute_cost("openai", entry, normalize_openai(None)).calculated_cost_usd is None


def test_max_cost_estimate_bounds_actual_cost() -> None:
    entry = TABLE.lookup("anthropic", "claude-haiku-5-5", DAY)
    assert entry is not None
    usage = normalize_anthropic(
        {
            "input_tokens": 50_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 900,
        }
    )
    actual = compute_cost("anthropic", entry, usage).calculated_cost_usd
    bound = max_cost_estimate("anthropic", entry, input_tokens=50_000, max_output_tokens=1000)
    assert actual is not None and bound >= actual


def test_long_context_tier_only_reserved_when_reachable() -> None:
    entry = TABLE.lookup("openai", "gpt-6.1-sol", DAY)
    assert entry is not None and entry.long_context_threshold == 272_000
    small = max_cost_estimate("openai", entry, input_tokens=400, max_output_tokens=512)
    assert small == Decimal("0.005920000000")  # standard $2 / $10 per 1M
    near = max_cost_estimate("openai", entry, input_tokens=250_000, max_output_tokens=512)
    assert near == Decimal("1.007680000000")  # within the safety margin: $4 / $15
    for tokens in (200_000, 260_000, 300_000):
        usage = normalize_openai({"input_tokens": tokens, "output_tokens": 512})
        actual = compute_cost("openai", entry, usage).calculated_cost_usd
        bound = max_cost_estimate("openai", entry, input_tokens=tokens, max_output_tokens=512)
        assert actual is not None and bound >= actual


def test_fixture_prices_are_marked() -> None:
    entry = TABLE.lookup("mock-openai", "mock-gpt", DAY)
    assert entry is not None and entry.is_fixture
    assert not TABLE.lookup("openai", "gpt-6-luna", DAY).is_fixture  # type: ignore[union-attr]


def test_prompt_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESHAT_PROMPT_HASH_KEY", raising=False)
    plain, scheme = hash_prompt("hello")
    assert scheme == PromptHashScheme.SHA256
    keyed, keyed_scheme = hash_prompt("hello", key="secret")
    assert keyed_scheme == PromptHashScheme.HMAC_SHA256 and keyed != plain


def test_redaction_and_detection() -> None:
    text = "mail jane@example.com, Bearer abc.def, key sk-ABCDEFGHIJKLMNOP, card 4111111111111111"
    assert set(detect_sensitive(text)) == {"email", "bearer_token", "api_key", "long_number"}
    redacted = redact_text(text) or ""
    for secret in ("jane@example.com", "abc.def", "sk-ABCDEFGHIJKLMNOP", "4111111111111111"):
        assert secret not in redacted


def test_retention_by_source_type() -> None:
    policy = RetentionPolicy(telemetry_days=30, log_import_days=7)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert policy.retain_until(SourceType.CONSENTED_API_TELEMETRY, now) == now + timedelta(days=30)
    assert policy.retain_until(SourceType.AUTHORIZED_LOG_IMPORT, now) == now + timedelta(days=7)
    assert policy.retain_until(SourceType.CONTROLLED_EXPERIMENT, now) is None
    assert policy.retain_until(SourceType.TEST_FIXTURE, now) is None
    with pytest.raises(ValueError):
        RetentionPolicy(telemetry_days=0)
