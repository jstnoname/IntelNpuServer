from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class StreamEvent(BaseModel):
    """A single engine event emitted while generating a reply."""

    kind: Literal["text", "tool_calls", "finish", "error"]
    text: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None


class GenerationRequest(BaseModel):
    """A composed prompt together with the resolved generation configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    prompt_tokens: int
    generation_config: Any


class GenerationOutcome(BaseModel):
    """A finalized generation result in neutral (HTTP-free) form."""

    model_config = ConfigDict(extra="forbid")

    content_text: str | None
    tool_calls: list[dict[str, Any]]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
