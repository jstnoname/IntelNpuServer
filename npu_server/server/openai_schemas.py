from __future__ import annotations

import json
import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatCompletionChunkBuilder:
    """Build OpenAI streaming chunk payloads and SSE frames."""

    def __init__(self, model: str) -> None:
        self._id = f"chatcmpl-{uuid.uuid4().hex}"
        self._created = int(time.time())
        self._model = model

    def chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    def event(self, delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = json.dumps(self.chunk(delta, finish_reason), ensure_ascii=False)
        return f"data: {payload}\n\n"

    def done(self) -> str:
        return "data: [DONE]\n\n"


class CompletionResult(BaseModel):
    """A finalized chat completion ready to be serialized for the client."""

    model_config = ConfigDict(extra="forbid")

    id: str
    created: int
    model: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int

    def response_body(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        }
