from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException


class ChatMessageNormalizer:
    """Normalize OpenAI chat messages into chat-template input format."""

    @staticmethod
    def content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="This server currently supports text-only message content.",
                    )
            return "\n".join(parts)
        return str(content)

    @classmethod
    def normalize(cls, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            if not isinstance(role, str):
                raise HTTPException(status_code=400, detail="Every message must have a string role")

            item: dict[str, Any] = {"role": role}
            if "content" in message:
                item["content"] = cls.content_to_text(message.get("content"))

            if role == "assistant" and message.get("tool_calls"):
                item["tool_calls"] = cls._hf_tool_calls(message["tool_calls"])

            if role == "tool":
                if message.get("tool_call_id") is not None:
                    item["tool_call_id"] = message["tool_call_id"]
                if message.get("name") is not None:
                    item["name"] = message["name"]

            if role == "assistant" and message.get("name") is not None:
                item["name"] = message["name"]

            normalized.append(item)

        return normalized

    @staticmethod
    def _hf_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hf_tool_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            function = call.get("function", {})
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    raw_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    raw_arguments = {"_raw_arguments": raw_arguments}
            if not isinstance(raw_arguments, dict):
                raw_arguments = {"_value": raw_arguments}

            hf_tool_calls.append(
                {
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "arguments": raw_arguments,
                    },
                }
            )
        return hf_tool_calls
