"""Pre-inference model features shared by training and the prediction API.

Every feature is computed from the exact request that will be submitted
(``LLMRequest``) plus provider/model identity. The same function runs on stored
``plan.jsonl`` requests during training and on API requests at serving time, so
there is no train/serve skew. Experimental design labels (``prompt_category``,
``design_cell_id``) and anything observed after dispatch are never features.

Task signals are generic, language-level cues parsed from the instruction text
(e.g. "write ... function", "translate", "in detail", "under 80 words"). They
are deliberately coarse so they transfer beyond the template generator; how well
they transfer to real user traffic is unverified until such data exists.
"""

from __future__ import annotations

import math
import re
from typing import Any

from seshat.data.features.extract import extract_request_features
from seshat.data.schemas.request import LLMRequest, Message

MODEL_FEATURE_SET_VERSION = "model_features/1.0.0"

_FLAGS = re.IGNORECASE
TASK_SIGNALS: dict[str, re.Pattern[str]] = {
    "write_code": re.compile(
        r"\b(write|implement|create|build)\b[^.\n]{0,60}\b(function|class|script|program|code|"
        r"module|endpoint)\b",
        _FLAGS,
    ),
    "review_code": re.compile(
        r"\b(review|critique|audit)\b[^.\n]{0,60}\b(code|function|implementation|diff)\b", _FLAGS
    ),
    "debug_code": re.compile(r"\b(bug|fix|debug|traceback|exception|off-by-one)\b", _FLAGS),
    "include_tests": re.compile(r"\b(unit tests?|test cases?|pytest)\b", _FLAGS),
    "translate": re.compile(r"\btranslat(e|ion)\b", _FLAGS),
    "summarize": re.compile(r"\bsummar(y|ise|ize)\b", _FLAGS),
    "classify": re.compile(r"\b(classif(y|ication)|sentiment|categori[sz]e)\b", _FLAGS),
    "extract": re.compile(r"\b(extract|entities)\b", _FLAGS),
    "json_output": re.compile(r"\bjson\b", _FLAGS),
    "math": re.compile(
        r"\b(solve|prove|compute|calculate|equation|integers?|divisible|how many)\b"
        r"|\d\s*[x*^]\s*\d|\b\d+\s*(liters?|litres?|km|miles|hours|minutes|percent|%)\b",
        _FLAGS,
    ),
    "explain": re.compile(r"\b(explain|why|how does|describe|misconceptions?)\b", _FLAGS),
    "creative": re.compile(r"\b(story|poem|dialogue|speech|creative|fiction)\b", _FLAGS),
    "long_form": re.compile(
        r"\b(comprehensive|in detail|detailed|step[- ]by[- ]step|thorough|essay|university|"
        r"examples)\b",
        _FLAGS,
    ),
    "brevity": re.compile(
        r"\b(concise|briefly|one sentence|one[- ]word|single word|short answer|"
        r"one label per line)\b",
        _FLAGS,
    ),
    "bullets": re.compile(r"\b(bullet|bullet points|list (every|all)|table)\b", _FLAGS),
    "puzzle": re.compile(r"\b(puzzle|deduction|deduce|logic)\b", _FLAGS),
    "document_qa": re.compile(
        r"\b(the document|which sections|according to|supporting quotes)\b", _FLAGS
    ),
}
# Mutually exclusive coarse task group for slicing/segment statistics: the first
# matching signal in this priority order (computable at inference time).
TASK_GROUP_ORDER = (
    "write_code",
    "review_code",
    "debug_code",
    "translate",
    "summarize",
    "classify",
    "extract",
    "json_output",
    "puzzle",
    "math",
    "creative",
    "long_form",
    "brevity",
    "explain",
    "document_qa",
)
_WORD_LIMIT = re.compile(
    r"\b(?:under|at most|no more than|fewer than|maximum of|max(?:imum)?|within)\s+(\d{1,5})"
    r"\s+words?\b",
    _FLAGS,
)
_N_WORDS = re.compile(r"\b(\d{1,5})\s+words?\b", _FLAGS)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10}
_N_ITEMS = re.compile(
    r"\b(one|two|three|four|five|six|ten|\d{1,2})\s+(bullet points?|bullets|paragraphs?|"
    r"sentences?|items|points|examples)\b",
    _FLAGS,
)
_LIST_LINE = re.compile(r"^\s*(?:[-*]|\d{1,3}[.)])\s+\S", re.MULTILINE)

NUMERIC_FEATURES = (
    "log_prompt_words",
    "log_current_message_chars",
    "log_context_chars",
    "log_list_items",
    "log_code_lines",
    "log_requested_word_limit",
    "has_word_limit",
    "requested_item_count",
    "message_count",
    "tool_count",
    "has_system_prompt",
    "contains_code",
    "contains_json",
    "contains_mathematics",
)
SIGNAL_FEATURES = tuple(f"signal_{name}" for name in TASK_SIGNALS)
CATEGORICAL_FEATURES = ("effort", "output_format", "provider_model", "task_group")


def _code_lines(text: str) -> int:
    blocks = re.findall(r"```.*?```", text, re.DOTALL)
    return sum(block.count("\n") for block in blocks)


def task_group(signals: dict[str, bool]) -> str:
    for name in TASK_GROUP_ORDER:
        if signals.get(name):
            return name
    return "general"


def model_features(request: LLMRequest) -> dict[str, Any]:
    """Model features for one request (deterministic; no network, no labels)."""
    base = extract_request_features(request)
    instruction = request.current_user_message().text()
    system = request.system or ""
    text = f"{system}\n{instruction}"
    signals = {name: bool(pattern.search(text)) for name, pattern in TASK_SIGNALS.items()}
    if request.tools:
        signals["tool_use"] = True
    limit_match = _WORD_LIMIT.search(instruction) or _N_WORDS.search(instruction)
    word_limit = int(limit_match.group(1)) if limit_match else 0
    item_match = _N_ITEMS.search(instruction)
    items = 0
    if item_match:
        token = item_match.group(1).lower()
        items = _COUNT_WORDS.get(token, int(token) if token.isdigit() else 0)
    effort = base.requested_effort or "unset"
    features: dict[str, Any] = {
        "log_prompt_words": math.log1p(base.prompt_word_count),
        "log_current_message_chars": math.log1p(base.current_user_message_char_count),
        "log_context_chars": math.log1p(base.context_character_count),
        "log_list_items": math.log1p(len(_LIST_LINE.findall(instruction))),
        "log_code_lines": math.log1p(_code_lines(instruction)),
        "log_requested_word_limit": math.log1p(word_limit),
        "has_word_limit": float(word_limit > 0),
        "requested_item_count": float(min(items, 20)),
        "message_count": float(base.message_count),
        "tool_count": float(base.tool_count),
        "has_system_prompt": float(bool(system)),
        "contains_code": float(base.contains_code),
        "contains_json": float(base.contains_json),
        "contains_mathematics": float(base.contains_mathematics),
        "effort": effort,
        "output_format": base.requested_output_format,
        "provider_model": f"{request.provider}/{request.model}",
        "task_group": "tool_use" if request.tools else task_group(signals),
    }
    for name in TASK_SIGNALS:
        features[f"signal_{name}"] = float(signals[name])
    return features


def request_from_prompt(
    prompt: str,
    *,
    provider: str,
    model: str,
    effort: str | None,
    system: str | None = None,
    max_output_tokens: int | None = None,
) -> LLMRequest:
    """The ``LLMRequest`` the API scores (single user turn)."""
    from seshat.data.schemas.request import GenerationParams

    return LLMRequest(
        provider=provider,
        model=model,
        api_surface="seshat.predict",
        system=system,
        messages=[Message(role="user", content=prompt)],
        params=GenerationParams(max_output_tokens=max_output_tokens, effort=effort),
    )
