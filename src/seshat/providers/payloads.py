"""Parse provider response payloads (SDK ``model_dump(mode="json")`` dicts).

Real adapters and the mock providers both go through these functions, so offline
tests exercise the same parsing used for live responses. Generated text is never
kept: only its character count, stop reason, tool-call counts and usage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from seshat.data.schemas.enums import ResponseStatus
from seshat.providers.base import ProviderResponse

_CONTENT_KEYS = {"content", "output", "candidates", "text", "output_text"}


def _metadata_only(payload: dict[str, Any]) -> dict[str, Any]:
    """Payload without generated content (kept: ids, model, status, usage, reasons)."""
    return {key: value for key, value in payload.items() if key not in _CONTENT_KEYS}


def parse_anthropic(payload: dict[str, Any], *, latency_ms: float | None) -> ProviderResponse:
    blocks = payload.get("content") or []
    text_chars = sum(
        len(block.get("text") or "") for block in blocks if block.get("type") == "text"
    )
    tool_calls = sum(1 for block in blocks if block.get("type") == "tool_use")
    server_calls = sum(1 for block in blocks if block.get("type") == "server_tool_use")
    stop = payload.get("stop_reason")
    status = {
        "refusal": ResponseStatus.REFUSED,
        "max_tokens": ResponseStatus.INCOMPLETE,
        "model_context_window_exceeded": ResponseStatus.INCOMPLETE,
        "pause_turn": ResponseStatus.INCOMPLETE,
    }.get(str(stop), ResponseStatus.COMPLETED)
    return ProviderResponse(
        response_id=payload.get("id"),
        resolved_model=payload.get("model"),
        status=status,
        stop_reason=stop,
        is_truncated=stop in {"max_tokens", "model_context_window_exceeded"},
        latency_ms=latency_ms,
        client_tool_call_count=tool_calls,
        server_tool_call_count=server_calls,
        output_char_count=text_chars,
        raw_usage=payload.get("usage") or {},
        raw_response=_metadata_only(payload),
        completed_at_utc=datetime.now(UTC),
    )


def parse_openai(payload: dict[str, Any], *, latency_ms: float | None) -> ProviderResponse:
    items = payload.get("output") or []
    text_chars = 0
    tool_calls = 0
    server_calls = 0
    for item in items:
        kind = item.get("type")
        if kind == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    text_chars += len(part.get("text") or "")
        elif kind == "function_call":
            tool_calls += 1
        elif kind and kind.endswith("_call"):
            server_calls += 1
    raw_status = payload.get("status")
    incomplete = payload.get("incomplete_details") or {}
    reason = incomplete.get("reason") if raw_status == "incomplete" else None
    status = {
        "completed": ResponseStatus.COMPLETED,
        "incomplete": ResponseStatus.INCOMPLETE,
        "failed": ResponseStatus.FAILED,
        "cancelled": ResponseStatus.FAILED,
    }.get(str(raw_status), ResponseStatus.COMPLETED)
    if reason == "content_filter":
        status = ResponseStatus.REFUSED
    return ProviderResponse(
        response_id=payload.get("id"),
        resolved_model=payload.get("model"),
        status=status,
        stop_reason=reason or raw_status,
        is_truncated=reason == "max_output_tokens",
        latency_ms=latency_ms,
        client_tool_call_count=tool_calls,
        server_tool_call_count=server_calls,
        output_char_count=text_chars,
        raw_usage=payload.get("usage") or {},
        raw_response=_metadata_only(payload),
        completed_at_utc=datetime.now(UTC),
    )


def parse_gemini(payload: dict[str, Any], *, latency_ms: float | None) -> ProviderResponse:
    candidates = payload.get("candidates") or []
    first = candidates[0] if candidates else {}
    parts = ((first.get("content") or {}).get("parts")) or []
    text_chars = sum(len(part.get("text") or "") for part in parts if not part.get("thought"))
    tool_calls = sum(1 for part in parts if part.get("function_call"))
    finish = first.get("finish_reason")
    status = ResponseStatus.COMPLETED
    if finish in {"MAX_TOKENS"}:
        status = ResponseStatus.INCOMPLETE
    elif finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}:
        status = ResponseStatus.REFUSED
    elif not candidates and (payload.get("prompt_feedback") or {}).get("block_reason"):
        status = ResponseStatus.REFUSED
    return ProviderResponse(
        response_id=payload.get("response_id"),
        resolved_model=payload.get("model_version"),
        model_version=payload.get("model_version"),
        status=status,
        stop_reason=finish,
        is_truncated=finish == "MAX_TOKENS",
        latency_ms=latency_ms,
        client_tool_call_count=tool_calls,
        server_tool_call_count=None,
        output_char_count=text_chars,
        raw_usage=payload.get("usage_metadata") or {},
        raw_response=_metadata_only(payload),
        completed_at_utc=datetime.now(UTC),
    )


PARSERS = {"anthropic": parse_anthropic, "openai": parse_openai, "gemini": parse_gemini}
