"""Collection policy: license allowlist and bank-wide limits (configs/prompt_collection.json)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from seshat.data.io import load_json

DEFAULT_POLICY_PATH = Path("configs") / "prompt_collection.json"


class LicenseTerms(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commercial_use: bool
    attribution_required: bool = False


class CollectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_version: str = "1.0"
    licenses: dict[str, LicenseTerms] = Field(min_length=1)
    allow_noncommercial: bool = False
    min_words: int = Field(default=3, ge=1)
    max_chars: int = Field(default=20_000, ge=1)
    max_synthetic_share: float = Field(default=0.5, ge=0.0, le=1.0)
    near_duplicate_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    min_approved_per_category: int = Field(default=3, ge=0)
    notes: str | None = None


def load_policy(path: str | Path | None = None) -> CollectionPolicy:
    """Load the policy from ``path`` or ``configs/prompt_collection.json`` in the CWD."""
    return CollectionPolicy.model_validate(load_json(path or DEFAULT_POLICY_PATH))
