"""Provider-neutral execution contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from seshat.data.models import EffortLevel, ProviderResult


class ProviderAdapter(ABC):
    name: str

    @abstractmethod
    def execute(
        self,
        *,
        prompt: str,
        model: str,
        effort: EffortLevel,
        max_output_tokens: int,
        temperature: float | None,
    ) -> ProviderResult:
        """Execute one visible request and return normalized + raw provider usage."""
        raise NotImplementedError


def is_retriable_exception(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = ("timeout", "timed out", "rate limit", "connection", "temporar", "overload")
    return any(marker in name or marker in text for marker in markers)
