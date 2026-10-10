"""Prompt-bank schema (PromptRecord v2.0) and data-quality decision types.

Canonical request/usage records live in ``seshat.data.schemas``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class QualityDecision(StrEnum):
    PASS = "PASS"
    WARNING = "WARNING"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"


PromptRole = Literal["system", "user", "assistant"]
ReviewStatus = Literal["unreviewed", "approved", "rejected"]
PROMPT_SCHEMA_VERSION: Literal["2.0"] = "2.0"


class PromptMessage(BaseModel):
    """One turn of a structured (multi-turn) prompt."""

    model_config = ConfigDict(extra="forbid")

    role: PromptRole
    content: str = Field(min_length=1)


class ToolDefinition(BaseModel):
    """Provider-neutral tool/function definition offered to the model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})


def render_messages(messages: list[PromptMessage]) -> str:
    """Canonical flat text for a structured prompt (used for features and hashing)."""
    return "\n\n".join(f"{message.role}: {message.content}" for message in messages)


class PromptRecord(BaseModel):
    """A prompt in the bank (schema 2.0; 1.0 lines without the new fields still load).

    Single-turn prompts set ``text``. Structured prompts set ``messages`` (and
    optionally ``tools``); ``text`` is then derived with ``render_messages`` so
    features and hashes stay comparable across both forms.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = PROMPT_SCHEMA_VERSION
    prompt_id: str = Field(min_length=1)
    prompt_family_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    subcategory: str | None = None
    prompt_template_id: str | None = None
    generation_seed: int | None = None
    variation_id: str | None = None
    text: str = Field(min_length=1)
    messages: list[PromptMessage] | None = None
    tools: list[ToolDefinition] = Field(default_factory=list)
    source: str = Field(min_length=1)
    source_detail: str | None = None
    language: str = Field(default="en", pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
    complexity: Literal["low", "medium", "high"] = "medium"
    is_synthetic: bool = False
    license_id: str | None = None
    commercial_use_allowed: bool | None = None
    tags: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = "unreviewed"
    reviewed_by: str | None = None
    reviewed_at: AwareDatetime | None = None
    review_notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _derive_text_from_messages(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not data.get("messages"):
            return data
        messages = [PromptMessage.model_validate(item) for item in data["messages"]]
        rendered = render_messages(messages)
        if data.get("text") not in (None, "", rendered):
            raise ValueError("text is derived from messages; omit it for structured prompts")
        return {**data, "text": rendered}

    @model_validator(mode="after")
    def _check_structure_and_review(self) -> PromptRecord:
        if self.messages is not None:
            if not self.messages:
                raise ValueError("messages must not be empty")
            if self.messages[-1].role != "user":
                raise ValueError("the last message of a prompt must have role 'user'")
            if any(message.role == "system" for message in self.messages[1:]):
                raise ValueError("a system message may only appear first")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        if self.review_status != "unreviewed" and not self.reviewed_by:
            raise ValueError("reviewed_by is required once a prompt is approved or rejected")
        return self

    @property
    def message_count(self) -> int:
        return len(self.messages) if self.messages else 1

    def provider_messages(self) -> list[dict[str, str]] | None:
        """Messages as plain dicts for provider adapters (``None`` for single-turn)."""
        if self.messages is None:
            return None
        return [message.model_dump() for message in self.messages]

    def provider_tools(self) -> list[dict[str, Any]] | None:
        return [tool.model_dump() for tool in self.tools] or None


class QualityFlag(BaseModel):
    code: str
    message: str
    severity: QualityDecision
