"""Model catalog: supported models and their capabilities (configs/model_catalog.json)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from seshat.data.io import load_json
from seshat.providers.base import ModelCapabilities

DEFAULT_CATALOG = Path("configs") / "model_catalog.json"


@dataclass(frozen=True)
class Catalog:
    version: str
    providers: dict[str, dict[str, Any]]

    def family(self, provider: str) -> str:
        return str(self._provider(provider)["family"])

    def models_for(self, provider: str) -> list[str]:
        return list(self._provider(provider)["models"])

    def capabilities(self, provider: str, model: str) -> ModelCapabilities:
        models = self._provider(provider)["models"]
        if model not in models:
            raise ValueError(
                f"Model {model!r} is not in the catalog for {provider!r} "
                f"(known: {sorted(models)}). Add it to configs/model_catalog.json with "
                "documented capabilities before collecting."
            )
        spec = models[model]

        def as_set(key: str) -> frozenset[str] | None:
            value = spec.get(key)
            return None if value is None else frozenset(value)

        return ModelCapabilities(
            provider=provider,
            model=model,
            verified=bool(spec.get("verified", False)),
            context_window=spec.get("context_window"),
            max_output_tokens=spec.get("max_output_tokens"),
            efforts=as_set("efforts"),
            thinking_modes=as_set("thinking_modes"),
            supports_temperature=bool(spec.get("supports_temperature")),
            supports_top_p=bool(spec.get("supports_top_p")),
            supports_top_k=bool(spec.get("supports_top_k")),
            supports_seed=bool(spec.get("supports_seed")),
            supports_stop_sequences=bool(spec.get("supports_stop_sequences")),
            supports_tools=bool(spec.get("supports_tools", True)),
            supports_parallel_tool_calls=bool(spec.get("supports_parallel_tool_calls")),
            supports_verbosity=bool(spec.get("supports_verbosity")),
            supports_response_format=bool(spec.get("supports_response_format")),
            supports_caching=bool(spec.get("supports_caching")),
            cache_ttls=frozenset(spec.get("cache_ttls") or []),
            native_token_counting=bool(spec.get("native_token_counting")),
            notes=spec.get("notes"),
        )

    def _provider(self, provider: str) -> dict[str, Any]:
        if provider not in self.providers:
            raise ValueError(f"Unknown provider {provider!r}; known: {sorted(self.providers)}")
        return self.providers[provider]


@lru_cache(maxsize=8)
def _load(path: str) -> Catalog:
    raw = load_json(path)
    return Catalog(version=str(raw["catalog_version"]), providers=raw["providers"])


def load_catalog(path: str | Path | None = None) -> Catalog:
    return _load(str(path or DEFAULT_CATALOG))
