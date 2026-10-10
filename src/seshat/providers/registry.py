"""Provider registry and the paid-call authorization gate."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from seshat.data.schemas.enums import SourceType
from seshat.providers.anthropic import AnthropicAdapter
from seshat.providers.base import ProviderAdapter
from seshat.providers.gemini import GeminiAdapter
from seshat.providers.mock import MockAnthropicAdapter, MockGeminiAdapter, MockOpenAIAdapter
from seshat.providers.openai import OpenAIAdapter

AdapterFactory = Callable[[], ProviderAdapter]

PROVIDERS: dict[str, type[ProviderAdapter]] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "mock-anthropic": MockAnthropicAdapter,
    "mock-openai": MockOpenAIAdapter,
    "mock-gemini": MockGeminiAdapter,
}


class ProviderCallsNotAllowedError(RuntimeError):
    """Raised before any request when live (paid) calls were not explicitly authorized."""


def adapter_class(provider: str) -> type[ProviderAdapter]:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider {provider!r}; known: {sorted(PROVIDERS)}") from exc


def live_providers(providers: Iterable[str]) -> list[str]:
    return sorted({name for name in providers if adapter_class(name).makes_external_calls})


def ensure_live_calls_authorized(providers: Iterable[str], execute_live: bool) -> None:
    """Fail fast unless live calls were authorized explicitly (CLI ``--execute-live``)."""
    live = live_providers(providers)
    if live and not execute_live:
        raise ProviderCallsNotAllowedError(
            f"The plan includes live providers ({', '.join(live)}). Live API calls are off "
            "by default; re-run with --execute-live and a --max-cost-usd budget, or use "
            "--dry-run / mock providers."
        )


def source_type_for(
    provider: str, live_source: SourceType = SourceType.CONTROLLED_EXPERIMENT
) -> SourceType:
    """Mock providers always produce test fixtures; real providers produce ``live_source``."""
    return (
        SourceType.TEST_FIXTURE if not adapter_class(provider).makes_external_calls else live_source
    )
