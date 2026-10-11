"""Collection configuration (``configs/data_collection.yaml``).

YAML or JSON. Every field has a conservative default: offline, small budgets, no
raw text retention. Paid live calls additionally require ``--execute-live`` on the
command line; nothing in the config alone can enable them.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _unset_str() -> list[str | None]:
    return [None]


def _unset_int() -> list[int | None]:
    return [None]


def _unset_float() -> list[float | None]:
    return [None]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GeneratePromptsConfig(_Strict):
    count: int = Field(default=120, ge=1)
    seed: int = 42
    templates: list[str] | None = None


class PromptSourcesConfig(_Strict):
    generate: GeneratePromptsConfig | None = Field(default_factory=GeneratePromptsConfig)
    bank_files: list[str] = Field(default_factory=list)
    include_categories: list[str] | None = None
    # Relative sampling weight per category (unlisted categories weigh 1.0; 0 drops a
    # category). Weights shape the round-robin order, so ``budget.max_requests``
    # trims the design in these proportions.
    category_weights: dict[str, float] | None = None
    require_approved_for_live: bool = True

    @model_validator(mode="after")
    def _non_negative_weights(self) -> PromptSourcesConfig:
        if self.category_weights and any(w < 0 for w in self.category_weights.values()):
            raise ValueError("category_weights must be >= 0")
        return self


class StrataConfig(_Strict):
    short_max_words: int = Field(default=80, ge=1)
    medium_max_words: int = Field(default=600, ge=2)

    @model_validator(mode="after")
    def _ordered(self) -> StrataConfig:
        if self.medium_max_words <= self.short_max_words:
            raise ValueError("medium_max_words must exceed short_max_words")
        return self


class TargetConfig(_Strict):
    """One provider/model and the setting levels to explore on it.

    Each list is a factor; ``null`` means "leave unset". Combinations are assigned to
    prompts by the design (never a full Cartesian product with prompts).
    """

    provider: str
    model: str
    efforts: list[str | None] = Field(default_factory=_unset_str)
    thinking_modes: list[str | None] = Field(default_factory=_unset_str)
    thinking_budget_tokens: list[int | None] = Field(default_factory=_unset_int)
    max_output_tokens: list[int] = Field(min_length=1)
    caching: list[bool] = Field(default_factory=lambda: [False])
    cache_ttl: str | None = None
    verbosity: list[str | None] = Field(default_factory=_unset_str)
    temperature: list[float | None] = Field(default_factory=_unset_float)
    structured_output: list[bool] = Field(default_factory=lambda: [False])
    service_tier: str | None = None
    # Restrict this target's configurations to prompts of these categories (e.g. a
    # larger output cap only where shorter caps truncate). ``None`` = all categories.
    categories: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _positive_limits(self) -> TargetConfig:
        if any(value < 1 for value in self.max_output_tokens):
            raise ValueError("max_output_tokens levels must be >= 1")
        return self


class SamplingConfig(_Strict):
    method: Literal["balanced_stratified"] = "balanced_stratified"
    configs_per_prompt: int = Field(default=2, ge=1)


class RepetitionConfig(_Strict):
    fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    count: int = Field(default=3, ge=1)


class BudgetConfig(_Strict):
    max_requests: int = Field(default=400, ge=1)
    max_cost_usd: Decimal | None = Field(default=None, ge=0)


class RateLimitConfig(_Strict):
    requests_per_minute: float = Field(default=60.0, gt=0)


class ExecutionConfig(_Strict):
    concurrency: int = Field(default=4, ge=1)
    per_provider_concurrency: int = Field(default=2, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    backoff_max_seconds: float = Field(default=60.0, ge=0)
    rate_limits: dict[str, RateLimitConfig] = Field(default_factory=dict)
    preflight: Literal["native", "approximate", "none"] = "native"
    preflight_discrepancy_tolerance: float = Field(default=0.1, ge=0)


class PrivacyConfig(_Strict):
    store_prompt_text: bool = False
    store_response_text: bool = False


class StorageConfig(_Strict):
    root: str = "data"


class CollectionConfig(_Strict):
    experiment_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    description: str | None = None
    seed: int = 42
    prompts: PromptSourcesConfig = Field(default_factory=PromptSourcesConfig)
    strata: StrataConfig = Field(default_factory=StrataConfig)
    targets: list[TargetConfig] = Field(min_length=1)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    repetitions: RepetitionConfig = Field(default_factory=RepetitionConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    def config_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def design_hash(self) -> str:
        """Hash of the settings that define *which* samples exist (not how they run).

        Budget, execution and privacy settings may change between resumed runs of one
        experiment; prompts, targets, sampling, repetitions, strata and seed may not.
        """
        design = self.model_dump(
            mode="json", include={"seed", "prompts", "strata", "targets", "sampling", "repetitions"}
        )
        canonical = json.dumps(design, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_collection_config(path: str | Path) -> CollectionConfig:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    raw: Any
    if source.suffix in {".yaml", ".yml"}:
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    return CollectionConfig.model_validate(raw)
