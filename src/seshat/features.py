"""Deterministic feature extraction for prompt-level prediction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptFeatures:
    char_count: int
    word_count: int
    line_count: int
    sentence_count: int
    digit_count: int
    punctuation_count: int
    code_fence_count: int
    avg_word_length: float
    log_char_count: float


_SENTENCE_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\S+")


def extract_prompt_features(prompt: str) -> PromptFeatures:
    words = _WORD_RE.findall(prompt)
    word_lengths = [len(word) for word in words]

    return PromptFeatures(
        char_count=len(prompt),
        word_count=len(words),
        line_count=prompt.count("\n") + 1,
        sentence_count=len(_SENTENCE_RE.findall(prompt)),
        digit_count=sum(char.isdigit() for char in prompt),
        punctuation_count=sum(not char.isalnum() and not char.isspace() for char in prompt),
        code_fence_count=prompt.count("```"),
        avg_word_length=(sum(word_lengths) / len(word_lengths)) if word_lengths else 0.0,
        log_char_count=math.log1p(len(prompt)),
    )
