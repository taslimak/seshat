"""Provider-specific usage normalization.

Each function maps a provider's *original* usage payload (as returned by its official
SDK's ``model_dump(mode="json")``) to Seshat's normalized label fields. Rules follow
the providers' documentation as of 2026-10-10 and are described in
``docs/DATA_SCHEMA.md#normalized-usage``. Key differences:

==============  =====================================  ==================================
Provider        Input tokens                           Output / reasoning
==============  =====================================  ==================================
OpenAI          ``input_tokens`` (cached is a subset)  ``output_tokens`` includes
                                                       ``reasoning_tokens``
Anthropic       ``input_tokens`` = uncached part only; ``output_tokens`` includes
                total = input + cache_read +           ``thinking_tokens``; with
                cache_creation                         compaction, sum ``iterations``
Gemini          ``prompt_token_count`` includes        ``candidates`` EXCLUDES
                cached content                         ``thoughts``; total = prompt +
                                                       candidates + thoughts
==============  =====================================  ==================================

Nothing is added blindly: totals are reconstructed from documented definitions, and
the provider's own total is kept separately for cross-checking. Unreported fields
stay ``None`` (unknown), never ``0``. Contradictions become ``issues`` and the
record is labelled ``inconsistent`` — never silently repaired.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from seshat.data.schemas.enums import LabelQuality


class NormalizedUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actual_input_tokens: int | None = None
    actual_output_tokens: int | None = None
    actual_reasoning_tokens: int | None = None
    actual_total_tokens: int | None = None
    actual_cached_input_tokens: int | None = None
    actual_cache_creation_tokens: int | None = None
    actual_uncached_input_tokens: int | None = None
    actual_visible_output_tokens: int | None = None
    actual_tool_input_tokens: int | None = None
    actual_tool_output_tokens: int | None = None
    input_usage_semantics: str
    output_usage_semantics: str
    reasoning_included_in_output: bool | None = None
    provider_total_tokens: int | None = None
    normalized_total_tokens: int | None = None
    usage_iteration_count: int = 1
    usage_observation_source: str
    billing_dimensions: dict[str, Any] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)

    def label_quality(self, *, is_truncated: bool | None, completed: bool) -> LabelQuality:
        """Quality of the primary labels given the response outcome."""
        if self.actual_output_tokens is None or self.actual_input_tokens is None:
            return LabelQuality.UNAVAILABLE
        if self.issues:
            return LabelQuality.INCONSISTENT
        if is_truncated:
            return LabelQuality.CENSORED
        if not completed:
            return LabelQuality.PARTIAL
        secondary = (self.actual_reasoning_tokens, self.actual_total_tokens)
        return (
            LabelQuality.PARTIAL
            if any(value is None for value in secondary)
            else LabelQuality.VERIFIED
        )


def _int(payload: dict[str, Any] | None, key: str) -> int | None:
    if not payload:
        return None
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"usage field {key} is not numeric: {value!r}")
    if int(value) != value:
        raise ValueError(f"usage field {key} is not an integer: {value!r}")
    return int(value)


def _sum(*values: int | None) -> int | None:
    return (
        None if any(value is None for value in values) else sum(v for v in values if v is not None)
    )


def _iteration_or_top_level(
    usage: dict[str, Any], iterations: list[dict[str, Any]], key: str
) -> int | None:
    """Sum ``key`` over iterations when they report it, else use the top-level value."""
    if any(key in item for item in iterations):
        return _sum(*[_int(item, key) or 0 for item in iterations])
    return _int(usage, key)


def _negative_issues(usage: NormalizedUsage) -> list[str]:
    return [
        f"negative_{name}"
        for name, value in usage.model_dump().items()
        if isinstance(value, int) and not isinstance(value, bool) and value < 0
    ]


def normalize_openai(usage: dict[str, Any] | None) -> NormalizedUsage:
    """OpenAI Responses API ``usage``.

    ``input_tokens_details.cached_tokens`` is a subset of ``input_tokens``;
    ``output_tokens_details.reasoning_tokens`` is a subset of ``output_tokens``;
    ``total_tokens`` = input + output.
    """
    semantics_in = "openai: input_tokens includes cached_tokens (subset)"
    semantics_out = "openai: output_tokens includes reasoning_tokens (subset)"
    if not usage:
        return NormalizedUsage(
            input_usage_semantics=semantics_in,
            output_usage_semantics=semantics_out,
            usage_observation_source="openai.responses.usage (missing)",
        )
    input_tokens = _int(usage, "input_tokens")
    output_tokens = _int(usage, "output_tokens")
    cached = _int(usage.get("input_tokens_details"), "cached_tokens")
    reasoning = _int(usage.get("output_tokens_details"), "reasoning_tokens")
    provider_total = _int(usage, "total_tokens")
    issues: list[str] = []
    uncached = None
    if input_tokens is not None and cached is not None:
        if cached > input_tokens:
            issues.append("cached_tokens_exceed_input_tokens")
        else:
            uncached = input_tokens - cached
    visible = None
    if output_tokens is not None and reasoning is not None:
        if reasoning > output_tokens:
            issues.append("reasoning_tokens_exceed_output_tokens")
        else:
            visible = output_tokens - reasoning
    normalized_total = _sum(input_tokens, output_tokens)
    if (
        provider_total is not None
        and normalized_total is not None
        and provider_total != normalized_total
    ):
        issues.append("total_tokens_mismatch")
    result = NormalizedUsage(
        actual_input_tokens=input_tokens,
        actual_output_tokens=output_tokens,
        actual_reasoning_tokens=reasoning,
        actual_total_tokens=normalized_total,
        actual_cached_input_tokens=cached,
        actual_cache_creation_tokens=None,  # not reported by OpenAI
        actual_uncached_input_tokens=uncached,
        actual_visible_output_tokens=visible,
        input_usage_semantics=semantics_in,
        output_usage_semantics=semantics_out,
        reasoning_included_in_output=True,
        provider_total_tokens=provider_total,
        normalized_total_tokens=normalized_total,
        usage_observation_source="openai.responses.usage",
        issues=issues,
    )
    result.issues.extend(_negative_issues(result))
    return result


def normalize_anthropic(usage: dict[str, Any] | None) -> NormalizedUsage:
    """Anthropic Messages API ``usage``.

    ``input_tokens`` covers only tokens after the last cache breakpoint; total input =
    input_tokens + cache_read_input_tokens + cache_creation_input_tokens.
    ``output_tokens_details.thinking_tokens`` <= ``output_tokens``. When
    ``usage.iterations`` is present (compaction), top-level counts exclude compaction
    iterations, so billed totals are summed over every iteration.
    """
    semantics_in = (
        "anthropic: input = input_tokens (uncached, after last breakpoint) "
        "+ cache_read_input_tokens + cache_creation_input_tokens"
    )
    semantics_out = "anthropic: output_tokens includes thinking_tokens"
    if not usage:
        return NormalizedUsage(
            input_usage_semantics=semantics_in,
            output_usage_semantics=semantics_out,
            usage_observation_source="anthropic.messages.usage (missing)",
        )
    issues: list[str] = []
    iterations = usage.get("iterations")
    dimensions: dict[str, Any] = {}
    if isinstance(iterations, list) and iterations:
        uncached = _sum(*[_int(item, "input_tokens") for item in iterations])
        output_tokens = _sum(*[_int(item, "output_tokens") for item in iterations])
        cache_read = _iteration_or_top_level(usage, iterations, "cache_read_input_tokens")
        cache_write = _iteration_or_top_level(usage, iterations, "cache_creation_input_tokens")
        kinds = [str(item.get("type")) for item in iterations]
        dimensions["iteration_types"] = kinds
        dimensions["compaction_iterations"] = kinds.count("compaction")
        semantics_in += "; summed over usage.iterations (compaction included)"
        semantics_out += "; summed over usage.iterations"
        top_in, top_out = _int(usage, "input_tokens"), _int(usage, "output_tokens")
        non_compaction = [it for it in iterations if it.get("type") != "compaction"]
        if (
            top_in is not None
            and _sum(*[_int(it, "input_tokens") for it in non_compaction]) != top_in
        ):
            issues.append("top_level_input_differs_from_non_compaction_iterations")
        if (
            top_out is not None
            and _sum(*[_int(it, "output_tokens") for it in non_compaction]) != top_out
        ):
            issues.append("top_level_output_differs_from_non_compaction_iterations")
        thinking = None  # per-iteration thinking is not reported
        iteration_count = len(iterations)
    else:
        uncached = _int(usage, "input_tokens")
        output_tokens = _int(usage, "output_tokens")
        cache_read = _int(usage, "cache_read_input_tokens")
        cache_write = _int(usage, "cache_creation_input_tokens")
        thinking = _int(usage.get("output_tokens_details"), "thinking_tokens")
        iteration_count = 1
    breakdown = usage.get("cache_creation") or {}
    if breakdown:
        five = _int(breakdown, "ephemeral_5m_input_tokens") or 0
        hour = _int(breakdown, "ephemeral_1h_input_tokens") or 0
        dimensions["cache_write_5m_tokens"] = five
        dimensions["cache_write_1h_tokens"] = hour
        if cache_write is not None and iteration_count == 1 and five + hour != cache_write:
            issues.append("cache_creation_breakdown_mismatch")
    server_tools = usage.get("server_tool_use") or {}
    if server_tools:
        dimensions["server_tool_use"] = {k: v for k, v in server_tools.items() if v}
    for key in ("service_tier", "inference_geo"):
        if usage.get(key) is not None:
            dimensions[key] = usage[key]
    total_input = None if uncached is None else uncached + (cache_read or 0) + (cache_write or 0)
    visible = None
    if thinking is not None and output_tokens is not None:
        if thinking > output_tokens:
            issues.append("thinking_tokens_exceed_output_tokens")
        else:
            visible = output_tokens - thinking
    normalized_total = _sum(total_input, output_tokens)
    result = NormalizedUsage(
        actual_input_tokens=total_input,
        actual_output_tokens=output_tokens,
        actual_reasoning_tokens=thinking,
        actual_total_tokens=normalized_total,
        actual_cached_input_tokens=cache_read,
        actual_cache_creation_tokens=cache_write,
        actual_uncached_input_tokens=uncached,
        actual_visible_output_tokens=visible,
        input_usage_semantics=semantics_in,
        output_usage_semantics=semantics_out,
        reasoning_included_in_output=True,
        provider_total_tokens=None,  # Anthropic reports no total
        normalized_total_tokens=normalized_total,
        usage_iteration_count=iteration_count,
        usage_observation_source="anthropic.messages.usage",
        billing_dimensions=dimensions,
        issues=issues,
    )
    result.issues.extend(_negative_issues(result))
    return result


def normalize_gemini(usage: dict[str, Any] | None) -> NormalizedUsage:
    """Gemini ``usage_metadata`` (google-genai ``GenerateContentResponseUsageMetadata``).

    ``prompt_token_count`` includes ``cached_content_token_count``.
    ``candidates_token_count`` excludes ``thoughts_token_count``; billed output is
    candidates + thoughts. ``total_token_count`` = prompt + candidates + thoughts.
    ``tool_use_prompt_token_count`` is kept separately: its inclusion in the total is
    not documented, so it is recorded (and the observed relationship noted) but never
    added to the normalized totals.
    """
    semantics_in = "gemini: prompt_token_count includes cached_content_token_count"
    semantics_out = (
        "gemini: actual_output = candidates_token_count + thoughts_token_count "
        "(provider candidates exclude thoughts)"
    )
    if not usage:
        return NormalizedUsage(
            input_usage_semantics=semantics_in,
            output_usage_semantics=semantics_out,
            usage_observation_source="gemini.usage_metadata (missing)",
        )
    prompt = _int(usage, "prompt_token_count")
    cached = _int(usage, "cached_content_token_count")
    candidates = _int(usage, "candidates_token_count")
    thoughts = _int(usage, "thoughts_token_count")
    tool_prompt = _int(usage, "tool_use_prompt_token_count")
    provider_total = _int(usage, "total_token_count")
    issues: list[str] = []
    dimensions: dict[str, Any] = {}
    uncached = None
    if prompt is not None and cached is not None:
        if cached > prompt:
            issues.append("cached_content_exceeds_prompt_tokens")
        else:
            uncached = prompt - cached
    elif prompt is not None:
        uncached = prompt  # no cache reported
    output_tokens = None if candidates is None else candidates + (thoughts or 0)
    normalized_total = _sum(prompt, output_tokens)
    if provider_total is not None and normalized_total is not None:
        if provider_total == normalized_total:
            dimensions["total_relationship"] = "prompt+candidates+thoughts"
        elif tool_prompt is not None and provider_total == normalized_total + tool_prompt:
            dimensions["total_relationship"] = "prompt+candidates+thoughts+tool_use_prompt"
        else:
            issues.append("total_token_count_mismatch")
    if usage.get("traffic_type") is not None:
        dimensions["traffic_type"] = usage["traffic_type"]
    result = NormalizedUsage(
        actual_input_tokens=prompt,
        actual_output_tokens=output_tokens,
        actual_reasoning_tokens=thoughts,
        actual_total_tokens=normalized_total,
        actual_cached_input_tokens=cached,
        actual_cache_creation_tokens=None,  # explicit cache creation is a separate API
        actual_uncached_input_tokens=uncached,
        actual_visible_output_tokens=candidates,
        actual_tool_input_tokens=tool_prompt,
        input_usage_semantics=semantics_in,
        output_usage_semantics=semantics_out,
        reasoning_included_in_output=False,
        provider_total_tokens=provider_total,
        normalized_total_tokens=normalized_total,
        usage_observation_source="gemini.usage_metadata",
        billing_dimensions=dimensions,
        issues=issues,
    )
    result.issues.extend(_negative_issues(result))
    return result


NORMALIZERS = {
    "openai": normalize_openai,
    "anthropic": normalize_anthropic,
    "gemini": normalize_gemini,
}
