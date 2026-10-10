"""Privacy-safe retention: prompt hashing, free-text redaction and expiry policy.

Defaults are conservative:

* Prompt text is **not** stored. Records keep a hash (keyed HMAC-SHA256 when
  ``SESHAT_PROMPT_HASH_KEY`` is set, plain SHA-256 otherwise) plus request-time
  features computed from the text before it is discarded.
* Free-text fields (error messages, string metadata) are scrubbed of e-mail
  addresses, bearer tokens and API-key-like strings, and truncated.
* Telemetry and log-import records get an expiry date (``retain_until``);
  ``purge_expired`` removes them once it passes. Controlled-experiment and fixture
  records use project-authored prompts and do not expire.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from seshat.data.schemas.enums import SourceType

PROMPT_HASH_KEY_ENV = "SESHAT_PROMPT_HASH_KEY"
MAX_FREE_TEXT_LENGTH = 500

_SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED_TOKEN]"),
    ("api_key", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}\b"), "[REDACTED_KEY]"),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_KEY]"),
    ("long_number", re.compile(r"\b\d{12,19}\b"), "[REDACTED_NUMBER]"),
)
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (pattern, replacement) for _, pattern, replacement in _SENSITIVE_PATTERNS
)


def detect_sensitive(text: str) -> list[str]:
    """Kinds of secrets/PII found in ``text`` (same patterns ``redact_text`` removes)."""
    return [kind for kind, pattern, _ in _SENSITIVE_PATTERNS if pattern.search(text)]


class PromptHashScheme(StrEnum):
    HMAC_SHA256 = "hmac-sha256"
    SHA256 = "sha256"


def hash_prompt(text: str, key: str | None = None) -> tuple[str, PromptHashScheme]:
    """Hash prompt text. Uses ``key`` or ``SESHAT_PROMPT_HASH_KEY`` for HMAC if set.

    An unkeyed SHA-256 of a short prompt can be reversed by guessing, so production
    deployments should configure a key. The scheme is recorded so hashes made under
    different schemes are never compared as equal.
    """
    secret = key if key is not None else os.environ.get(PROMPT_HASH_KEY_ENV) or None
    payload = text.encode("utf-8")
    if secret:
        digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return digest, PromptHashScheme.HMAC_SHA256
    return hashlib.sha256(payload).hexdigest(), PromptHashScheme.SHA256


def redact_text(text: str | None, *, max_length: int = MAX_FREE_TEXT_LENGTH) -> str | None:
    """Scrub common secrets/PII patterns from free text and cap its length."""
    if text is None:
        return None
    redacted = text
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    if len(redacted) > max_length:
        redacted = redacted[: max_length - 1] + "…"
    return redacted


@dataclass(frozen=True)
class RetentionPolicy:
    """How long records from user-derived sources are kept, and whether raw text may be.

    Consented telemetry and authorized log imports can contain end-user content, so
    they expire after a configurable number of days. Controlled experiments use
    project-authored or licensed prompts and test fixtures are generated, so they do
    not expire. Raw prompt/response text is stored only when explicitly enabled *and*
    consent allows it.
    """

    telemetry_days: int | None = 90
    log_import_days: int | None = 90
    store_prompt_text: bool = False
    store_response_text: bool = False

    def __post_init__(self) -> None:
        for name in ("telemetry_days", "log_import_days"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")

    def retain_until(
        self, source_type: SourceType | str, reference_time: datetime
    ) -> datetime | None:
        """Expiry for a record, or ``None`` when it is kept indefinitely."""
        source = SourceType(source_type)
        days = {
            SourceType.CONSENTED_API_TELEMETRY: self.telemetry_days,
            SourceType.AUTHORIZED_LOG_IMPORT: self.log_import_days,
        }.get(source)
        return None if days is None else reference_time + timedelta(days=days)
