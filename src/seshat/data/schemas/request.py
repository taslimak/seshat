"""Provider-neutral description of the exact request sent to a model.

``LLMRequest`` is what the collector dispatches and what request-time features are
computed from, so features always describe the request that was actually submitted.
Unsupported settings stay ``None``; adapters translate only the settings a model
supports (see ``seshat.providers.base.ModelCapabilities``) and reject the rest.

Attachments carry metadata plus an optional local ``source_path``. Their bytes are
read only at dispatch time and are never written to datasets.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]
PartType = Literal["text", "image", "document", "audio", "video"]


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: PartType
    text: str | None = None
    mime_type: str | None = None
    byte_size: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    page_count: int | None = Field(default=None, ge=0)
    duration_seconds: float | None = Field(default=None, ge=0)
    source_path: str | None = None

    @model_validator(mode="after")
    def _text_parts_need_text(self) -> ContentPart:
        if self.type == "text" and not self.text:
            raise ValueError("text parts require text")
        if self.type != "text" and self.text is not None:
            raise ValueError("only text parts carry text")
        return self


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str | list[ContentPart]
    name: str | None = None
    tool_call_id: str | None = None

    def parts(self) -> list[ContentPart]:
        if isinstance(self.content, str):
            return [ContentPart(type="text", text=self.content)]
        return self.content

    def text(self) -> str:
        return "\n".join(part.text or "" for part in self.parts() if part.type == "text")


class ToolSpec(BaseModel):
    """A tool offered to the model. ``type="function"`` for client tools;
    other values name provider server tools (e.g. ``web_search``)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    type: str = "function"


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    ttl: str | None = None  # e.g. "5m", "1h", "24h"; provider-specific
    key: str | None = None


class GenerationParams(BaseModel):
    """Execution settings. ``None`` means "not set / not supported" — never a default."""

    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int | None = Field(default=None, ge=1)
    effort: str | None = None
    thinking_mode: str | None = None  # adaptive | enabled | disabled | level name
    thinking_budget_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    seed: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    response_format: Literal["text", "json", "json_schema"] | None = None
    response_schema: dict[str, Any] | None = None
    verbosity: str | None = None
    tool_choice: str | None = None  # auto | any | none | <tool name>
    parallel_tool_calls: bool | None = None
    stream: bool = False
    cache: CacheConfig = Field(default_factory=CacheConfig)
    service_tier: str | None = None
    region: str | None = None
    routing_mode: str | None = None


class LLMRequest(BaseModel):
    """The exact request sent to one provider/model (one billable invocation)."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_surface: str | None = None
    system: str | None = None
    messages: list[Message] = Field(min_length=1)
    tools: list[ToolSpec] = Field(default_factory=list)
    params: GenerationParams = Field(default_factory=GenerationParams)
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _structure(self) -> LLMRequest:
        if any(message.role == "system" for message in self.messages):
            raise ValueError("put the system prompt in LLMRequest.system, not in messages")
        if self.messages[-1].role not in {"user", "tool"}:
            raise ValueError("the last message must be from the user (or a tool result)")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        return self

    def current_user_message(self) -> Message:
        return self.messages[-1]

    def all_text(self) -> str:
        """System + message text in order (the text actually sent, excluding tools)."""
        chunks = [self.system] if self.system else []
        chunks.extend(message.text() for message in self.messages)
        return "\n\n".join(chunk for chunk in chunks if chunk)
