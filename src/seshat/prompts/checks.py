"""Per-prompt policy checks shared by import and bank-wide validation."""

from __future__ import annotations

from dataclasses import dataclass

from seshat.data.models import PromptRecord
from seshat.data.privacy import detect_sensitive
from seshat.prompts.policy import CollectionPolicy

SYNTHETIC_SOURCES = frozenset({"synthetic", "ai_drafted", "generated"})


@dataclass(frozen=True)
class Issue:
    code: str
    message: str


def check_prompt(prompt: PromptRecord, policy: CollectionPolicy) -> list[Issue]:
    """Policy violations that make a prompt ineligible for the bank."""
    issues: list[Issue] = []
    words = len(prompt.text.split())
    if words < policy.min_words:
        issues.append(Issue("text_too_short", f"{words} words < min_words={policy.min_words}"))
    if len(prompt.text) > policy.max_chars:
        issues.append(
            Issue("text_too_long", f"{len(prompt.text)} chars > max_chars={policy.max_chars}")
        )
    texts = [prompt.text, *(tool.description for tool in prompt.tools)]
    found = sorted({kind for text in texts for kind in detect_sensitive(text)})
    if found:
        issues.append(Issue("sensitive_content", f"possible secrets/PII: {', '.join(found)}"))
    terms = policy.licenses.get(prompt.license_id or "")
    if terms is None:
        issues.append(
            Issue(
                "license_not_allowed",
                f"license_id={prompt.license_id!r} is not in the collection allowlist",
            )
        )
    else:
        if prompt.commercial_use_allowed is None:
            issues.append(
                Issue("commercial_use_unknown", "commercial_use_allowed must be set explicitly")
            )
        elif prompt.commercial_use_allowed and not terms.commercial_use:
            issues.append(
                Issue(
                    "commercial_use_conflict",
                    f"{prompt.license_id} does not permit commercial use",
                )
            )
        if prompt.commercial_use_allowed is False and not policy.allow_noncommercial:
            issues.append(
                Issue("noncommercial_not_allowed", "non-commercial prompts are disabled by policy")
            )
        if terms.attribution_required and not prompt.source_detail:
            issues.append(
                Issue(
                    "attribution_missing",
                    f"{prompt.license_id} requires attribution in source_detail",
                )
            )
    if prompt.source in SYNTHETIC_SOURCES and not prompt.is_synthetic:
        issues.append(
            Issue("synthetic_flag_missing", f"source={prompt.source!r} must set is_synthetic")
        )
    if prompt.is_synthetic and not prompt.source_detail:
        issues.append(
            Issue(
                "synthetic_provenance_missing",
                "synthetic prompts must name their generator in source_detail",
            )
        )
    return issues
