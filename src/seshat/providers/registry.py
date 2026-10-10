"""Provider registry, provenance mapping and the paid-call safety gate."""

from __future__ import annotations

import os
from collections.abc import Iterable

from seshat.data.provenance import DataOrigin, DataSource, Provenance
from seshat.providers.anthropic import AnthropicAdapter
from seshat.providers.base import ProviderAdapter
from seshat.providers.mock import MockAdapter
from seshat.providers.openai import OpenAIAdapter

ALLOW_PROVIDER_CALLS_ENV = "SESHAT_ALLOW_PROVIDER_CALLS"

PROVIDERS: dict[str, type[ProviderAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "mock": MockAdapter,
}


class ProviderCallsNotAllowedError(RuntimeError):
    """Raised before any request when real provider calls have not been enabled."""


def adapter_class(provider: str) -> type[ProviderAdapter]:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider {provider!r}; known: {sorted(PROVIDERS)}") from exc


def provider_calls_allowed(explicit: bool | None = None) -> bool:
    """Explicit flag wins; otherwise ``SESHAT_ALLOW_PROVIDER_CALLS`` must be truthy."""
    if explicit is not None:
        return explicit
    return os.environ.get(ALLOW_PROVIDER_CALLS_ENV, "").strip().lower() in {"1", "true", "yes"}


def ensure_provider_calls_allowed(providers: Iterable[str], explicit: bool | None = None) -> None:
    """Fail fast if any provider would make external (potentially paid) calls."""
    external = sorted(
        {name for name in providers if adapter_class(name).makes_external_calls}
    )
    if external and not provider_calls_allowed(explicit):
        raise ProviderCallsNotAllowedError(
            f"Manifest requires real provider calls ({', '.join(external)}), which are "
            f"disabled by default. Re-run with --allow-provider-calls or set "
            f"{ALLOW_PROVIDER_CALLS_ENV}=1 after reviewing the expected spend."
        )


def provenance_for_provider(provider: str, experiment_id: str | None = None) -> Provenance:
    """Provenance of results produced by a provider adapter in a controlled experiment."""
    cls = adapter_class(provider)
    if cls.data_origin == DataOrigin.SYNTHETIC:
        source = DataSource.MOCK_ADAPTER
    else:
        source = DataSource.CONTROLLED_EXPERIMENT
    return Provenance(
        data_origin=cls.data_origin,
        source=source,
        generator=f"{cls.__module__}.{cls.__name__}",
        experiment_id=experiment_id,
    )
