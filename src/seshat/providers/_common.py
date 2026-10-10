"""Helpers shared by live adapters."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from seshat.data.schemas.request import ContentPart, LLMRequest
from seshat.providers.base import ErrorCategory, ProviderError

PRIVATE_OPTION_PREFIX = "_"


def public_options(request: LLMRequest) -> dict[str, Any]:
    """Provider options to forward (keys starting with '_' are Seshat-internal)."""
    return {
        k: v for k, v in request.provider_options.items() if not k.startswith(PRIVATE_OPTION_PREFIX)
    }


def read_attachment(part: ContentPart) -> tuple[str, str]:
    """(mime_type, base64 data) for a non-text part, read at dispatch time only."""
    if not part.source_path:
        raise ProviderError(
            ErrorCategory.UNSUPPORTED,
            f"{part.type} part has no source_path; attachment bytes are never stored in datasets",
        )
    if not part.mime_type:
        raise ProviderError(ErrorCategory.UNSUPPORTED, f"{part.type} part needs mime_type")
    data = Path(part.source_path).read_bytes()
    return part.mime_type, base64.standard_b64encode(data).decode("ascii")


_CREDENTIAL_MARKERS = (
    "api key",
    "api_key",
    "credential",
    "authentication",
    "auth_token",
    "unauthorized",
)


def classify_client_side(exc: Exception) -> ProviderError:
    """Errors raised while *building* a request (before any network I/O).

    Nothing reached the provider, so the outcome is certain (not executed). Missing or
    invalid credentials are fatal for the whole run.
    """
    text = f"{type(exc).__name__}: {exc}"
    if any(marker in text.lower() for marker in _CREDENTIAL_MARKERS):
        return ProviderError(ErrorCategory.AUTHENTICATION, text[:500])
    return ProviderError(ErrorCategory.INVALID_REQUEST, text[:500])


def require_max_output(request: LLMRequest) -> int:
    if request.params.max_output_tokens is None:
        raise ProviderError(
            ErrorCategory.INVALID_REQUEST,
            "max_output_tokens is required for live requests (budget enforcement)",
        )
    return request.params.max_output_tokens
