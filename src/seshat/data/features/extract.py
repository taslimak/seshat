"""Canonical request-time feature extraction.

``extract_request_features(request)`` is the single implementation used by the
collector, importers and telemetry, and is designed to be reused by the prediction
API. It reads only the request that will be (or was) submitted, never the response,
so every output is available before inference.

Definitions of every feature are in ``FEATURE_DEFINITIONS`` (rendered into
``docs/DATA_SCHEMA.md``). All functions are deterministic.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from seshat.data.privacy import hash_prompt
from seshat.data.schemas.records import AttachmentMetadata, ImageMetadata
from seshat.data.schemas.request import LLMRequest
from seshat.features import extract_prompt_features

FEATURE_SET_VERSION = "request_features/1.0.0"

_CODE_PATTERNS = re.compile(
    r"```|^\s*(def|class|import|from|function|const|let|var|public|private|#include|package)\b"
    r"|\b(SELECT|INSERT|UPDATE|DELETE)\b.+\b(FROM|INTO|SET|WHERE)\b"
    r"|[;{}]\s*$|=>|\breturn\b.*;",
    re.MULTILINE,
)
_MARKDOWN_PATTERNS = re.compile(
    r"^#{1,6}\s+\S|^\s*[-*+]\s+\S|^\s*\d+\.\s+\S|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\)|^\|.*\|\s*$|```",
    re.MULTILINE,
)
_MATH_PATTERNS = re.compile(
    r"\$[^$\n]+\$|\\(frac|sum|int|sqrt|alpha|beta|theta|pi|cdot|leq|geq|infty)\b"
    r"|\d\s*[\^×÷]\s*\d|[∑∫√≤≥≠∞π]|\b\d+\s*[+\-*/]\s*\d+\s*=|\b[a-z]\s*\^\s*\d"
    r"|\b(mod)\s+\d+",
    re.IGNORECASE,
)
_JSON_KEY = re.compile(r'"[A-Za-z_][\w\- ]*"\s*:')

_FORMAT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("json", re.compile(r"\bjson\b", re.IGNORECASE)),
    ("yaml", re.compile(r"\byaml\b", re.IGNORECASE)),
    ("csv", re.compile(r"\bcsv\b", re.IGNORECASE)),
    ("table", re.compile(r"\b(table|tabular)\b", re.IGNORECASE)),
    (
        "code",
        re.compile(
            r"\b(write|implement|refactor|fix)\b.*\b(function|class|script|code|query|program)\b",
            re.IGNORECASE,
        ),
    ),
    ("bullet_list", re.compile(r"\b(bullet|bulleted|list of|numbered list)\b", re.IGNORECASE)),
    ("markdown", re.compile(r"\bmarkdown\b", re.IGNORECASE)),
)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_LENGTH_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:under|at most|no more than|fewer than|less than|maximum of|max(?:imum)?)\s+(\d+)\s+(words|sentences|tokens|characters|lines|paragraphs)\b",
            re.IGNORECASE,
        ),
        "max_{unit}:{n}",
    ),
    (
        re.compile(
            r"\b(?:at least|minimum of|no fewer than)\s+(\d+)\s+(words|sentences|paragraphs|lines)\b",
            re.IGNORECASE,
        ),
        "min_{unit}:{n}",
    ),
    (
        re.compile(
            r"\b(?:in|exactly|about|around|approximately|roughly)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(words|sentences|bullet points|bullets|paragraphs|lines|items|examples|steps)\b",
            re.IGNORECASE,
        ),
        "count_{unit}:{n}",
    ),
    (
        re.compile(
            r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)[- ](sentence|paragraph|word|line|bullet|item|step)s?\b",
            re.IGNORECASE,
        ),
        "count_{unit}:{n}",
    ),
    (
        re.compile(
            r"\b(one sentence|single sentence|one word|single word|one line)\b", re.IGNORECASE
        ),
        "count_{phrase}",
    ),
    (
        re.compile(
            r"\b(brief(?:ly)?|concise(?:ly)?|short|succinct(?:ly)?|tl;?dr)\b", re.IGNORECASE
        ),
        "qualitative:brief",
    ),
    (
        re.compile(
            r"\b(detailed|in detail|comprehensive|thorough(?:ly)?|in depth|step[- ]by[- ]step)\b",
            re.IGNORECASE,
        ),
        "qualitative:detailed",
    ),
)

FEATURE_DEFINITIONS: dict[str, str] = {
    "prompt_char_count": "Unicode characters in all text sent: system prompt + every message, joined by blank lines (tool schemas excluded).",
    "prompt_word_count": "Whitespace-separated tokens in the same text.",
    "prompt_line_count": "Newline count + 1 in the same text.",
    "prompt_byte_count": "UTF-8 bytes of the same text.",
    "system_prompt_char_count": "Characters in the system prompt (0 when absent).",
    "message_count": "Messages in the request, excluding the system prompt.",
    "conversation_turn_count": "Number of user messages (turns initiated by the user).",
    "conversation_char_count": "Characters across all messages, excluding the system prompt.",
    "current_user_message_char_count": "Characters in the final (current) message.",
    "role_sequence": "Ordered roles: 'system' first when present, then each message role.",
    "contains_code": "Regex: code fences, language keywords at line start, SQL statements, or statement terminators.",
    "contains_json": 'A JSON object/array parses from the text, or a quoted-key pattern ("key":) appears.',
    "contains_markdown": "Regex: headings, list items, bold, links, tables or code fences.",
    "contains_mathematics": "Regex: LaTeX, math symbols, arithmetic equations, exponents or modular notation.",
    "requested_output_format": "response_format json/json_schema -> 'json'; else first matching keyword rule on the current message (json, yaml, csv, table, code, bullet_list, markdown); else 'unspecified'.",
    "explicit_output_length_hint": "First matching length instruction in the current message, normalized (e.g. 'max_words:120', 'count_bullet_points:3', 'qualitative:brief'); null if none.",
    "prompt_hash": "Hash (HMAC-SHA256 with SESHAT_PROMPT_HASH_KEY, else SHA-256) of the canonical JSON of system + messages (text and attachment metadata).",
    "conversation_hash": "Hash of the prior context (system + all but the last message); null for single-message requests.",
    "stop_sequence_count": "Number of stop sequences configured.",
    "tool_count": "Number of tools offered.",
    "tool_schema_bytes": "UTF-8 bytes of the canonical JSON of all tool definitions (0 without tools).",
    "tool_schema_hash": "SHA-256 of that canonical JSON; null without tools.",
    "enabled_tool_types": "Sorted distinct tool types (function, web_search, ...).",
    "caching_enabled": "Prompt caching requested.",
    "context_message_count": "Messages before the current one.",
    "context_character_count": "Characters in the system prompt plus all messages before the current one.",
    "context_preflight_tokens": "Preflight input-token count for the whole request when available (from the preflight table).",
    "image_count": "Image parts across all messages.",
    "image_total_bytes": "Sum of known image byte sizes; null when no size is known.",
    "document_count": "Document parts across all messages.",
    "document_page_count": "Sum of known document page counts; null when unknown.",
    "audio_duration_seconds": "Sum of known audio durations; null when no audio or unknown.",
    "video_duration_seconds": "Sum of known video durations; null when no video or unknown.",
    "modality_types": "Sorted distinct content part types present (text, image, document, audio, video).",
}


class RequestFeatures(BaseModel):
    """Feature fields of ``RequestRecord`` sections B (computed part), C and D."""

    model_config = ConfigDict(extra="forbid")

    prompt_char_count: int
    prompt_word_count: int
    prompt_line_count: int
    prompt_byte_count: int
    system_prompt_char_count: int
    message_count: int
    conversation_turn_count: int
    conversation_char_count: int
    current_user_message_char_count: int
    role_sequence: list[str]
    contains_code: bool
    contains_json: bool
    contains_markdown: bool
    contains_mathematics: bool
    requested_output_format: str
    explicit_output_length_hint: str | None
    prompt_hash: str
    conversation_hash: str | None
    max_output_tokens: int | None
    requested_effort: str | None
    resolved_effort: str | None
    thinking_mode: str | None
    thinking_budget_tokens: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    seed: int | None
    stop_sequence_count: int
    response_format: str | None
    verbosity: str | None
    tool_count: int
    tool_choice: str | None
    tool_schema_bytes: int
    tool_schema_hash: str | None
    enabled_tool_types: list[str]
    parallel_tool_calls: bool | None
    streaming_enabled: bool
    caching_enabled: bool
    requested_cache_ttl: str | None
    service_tier: str | None
    region: str | None
    routing_mode: str | None
    provider_settings: dict[str, Any]
    context_window_limit: int | None
    context_message_count: int
    context_character_count: int
    image_count: int
    image_total_bytes: int | None
    image_metadata: list[ImageMetadata]
    document_count: int
    document_page_count: int | None
    audio_duration_seconds: float | None
    video_duration_seconds: float | None
    modality_types: list[str]
    attachment_metadata: list[AttachmentMetadata]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def contains_json(text: str) -> bool:
    if _JSON_KEY.search(text):
        return True
    for start, _opener, closer in (
        (i, c, "}" if c == "{" else "]") for i, c in enumerate(text) if c in "{["
    ):
        end = text.rfind(closer)
        if end > start:
            try:
                json.loads(text[start : end + 1])
                return True
            except ValueError:
                pass
        if start > 2000:
            break
    return False


def requested_output_format(request: LLMRequest) -> str:
    if request.params.response_format in {"json", "json_schema"}:
        return "json"
    current = request.current_user_message().text()
    for name, pattern in _FORMAT_RULES:
        if pattern.search(current):
            return name
    return "unspecified"


def _plural_unit(unit: str) -> str:
    """'bullet point' / 'bullets' -> 'bullet_points'; 'sentence' -> 'sentences'."""
    unit = unit.strip().lower()
    if unit in {"bullet", "bullets"}:
        unit = "bullet points"
    words = unit.split()
    if not words[-1].endswith("s"):
        words[-1] += "s"
    return "_".join(words)


def explicit_length_hint(text: str) -> str | None:
    for pattern, template in _LENGTH_RULES:
        match = pattern.search(text)
        if not match:
            continue
        if template == "count_{phrase}":
            return "count_" + match.group(1).lower().replace(" ", "_")
        if template.startswith("qualitative"):
            return template
        raw_n, unit = match.group(1).lower(), match.group(2).lower()
        n = _NUMBER_WORDS.get(raw_n, raw_n)
        return template.format(unit=_plural_unit(unit), n=n)
    return None


def _sum_known(values: list[int | float | None], present: bool) -> Any:
    known = [value for value in values if value is not None]
    if not present or not known:
        return None
    return sum(known)


def extract_request_features(
    request: LLMRequest,
    *,
    context_window_limit: int | None = None,
    resolved_effort: str | None = None,
    hash_key: str | None = None,
) -> RequestFeatures:
    """Compute request-time features from the exact request to be submitted."""
    text = request.all_text()
    stats = extract_prompt_features(text)
    system_text = request.system or ""
    message_texts = [message.text() for message in request.messages]
    current = request.current_user_message()
    context_texts = [system_text, *message_texts[:-1]]

    content_payload: dict[str, Any] = {
        "system": request.system,
        "messages": [
            {
                "role": message.role,
                "parts": [part.model_dump(exclude={"source_path"}) for part in message.parts()],
            }
            for message in request.messages
        ],
    }
    prompt_hash, _ = hash_prompt(_canonical_json(content_payload), hash_key)
    conversation_hash = None
    if len(request.messages) > 1:
        context_payload = {**content_payload, "messages": content_payload["messages"][:-1]}
        conversation_hash, _ = hash_prompt(_canonical_json(context_payload), hash_key)

    tools_json = (
        _canonical_json([tool.model_dump() for tool in request.tools]) if request.tools else ""
    )
    parts = [part for message in request.messages for part in message.parts()]
    images = [part for part in parts if part.type == "image"]
    documents = [part for part in parts if part.type == "document"]
    audio = [part for part in parts if part.type == "audio"]
    video = [part for part in parts if part.type == "video"]
    params = request.params

    return RequestFeatures(
        prompt_char_count=stats.char_count,
        prompt_word_count=stats.word_count,
        prompt_line_count=stats.line_count,
        prompt_byte_count=len(text.encode("utf-8")),
        system_prompt_char_count=len(system_text),
        message_count=len(request.messages),
        conversation_turn_count=sum(1 for message in request.messages if message.role == "user"),
        conversation_char_count=sum(len(item) for item in message_texts)
        + 2 * (len(message_texts) - 1),
        current_user_message_char_count=len(current.text()),
        role_sequence=(["system"] if request.system else []) + [m.role for m in request.messages],
        contains_code=bool(_CODE_PATTERNS.search(text)),
        contains_json=contains_json(text),
        contains_markdown=bool(_MARKDOWN_PATTERNS.search(text)),
        contains_mathematics=bool(_MATH_PATTERNS.search(text)),
        requested_output_format=requested_output_format(request),
        explicit_output_length_hint=explicit_length_hint(current.text()),
        prompt_hash=prompt_hash,
        conversation_hash=conversation_hash,
        max_output_tokens=params.max_output_tokens,
        requested_effort=params.effort,
        resolved_effort=resolved_effort if resolved_effort is not None else params.effort,
        thinking_mode=params.thinking_mode,
        thinking_budget_tokens=params.thinking_budget_tokens,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        seed=params.seed,
        stop_sequence_count=len(params.stop_sequences),
        response_format=params.response_format,
        verbosity=params.verbosity,
        tool_count=len(request.tools),
        tool_choice=params.tool_choice,
        tool_schema_bytes=len(tools_json.encode("utf-8")),
        tool_schema_hash=hashlib.sha256(tools_json.encode("utf-8")).hexdigest()
        if tools_json
        else None,
        enabled_tool_types=sorted({tool.type for tool in request.tools}),
        parallel_tool_calls=params.parallel_tool_calls,
        streaming_enabled=params.stream,
        caching_enabled=params.cache.enabled,
        requested_cache_ttl=params.cache.ttl,
        service_tier=params.service_tier,
        region=params.region,
        routing_mode=params.routing_mode,
        provider_settings=dict(request.provider_options),
        context_window_limit=context_window_limit,
        context_message_count=len(request.messages) - 1,
        context_character_count=sum(len(item) for item in context_texts),
        image_count=len(images),
        image_total_bytes=_sum_known([p.byte_size for p in images], bool(images)),
        image_metadata=[
            ImageMetadata(
                mime_type=p.mime_type, byte_size=p.byte_size, width=p.width, height=p.height
            )
            for p in images
        ],
        document_count=len(documents),
        document_page_count=_sum_known([p.page_count for p in documents], bool(documents)),
        audio_duration_seconds=_sum_known([p.duration_seconds for p in audio], bool(audio)),
        video_duration_seconds=_sum_known([p.duration_seconds for p in video], bool(video)),
        modality_types=sorted({part.type for part in parts}),
        attachment_metadata=[
            AttachmentMetadata(
                type=p.type,
                mime_type=p.mime_type,
                byte_size=p.byte_size,
                page_count=p.page_count,
                duration_seconds=p.duration_seconds,
            )
            for p in parts
            if p.type != "text"
        ],
    )
