import json
import re
import uuid
from typing import Any

_LFM_OPEN = chr(60) + "tool_call" + chr(62)
_LFM_CLOSE = chr(60) + "/tool_call" + chr(62)

_HERMES_OPEN = chr(60) + "tool_call" + chr(62)
_HERMES_CLOSE = chr(60) + "/tool_call" + chr(62)


class Lfm3ToolCallParser:
    """Parse the XML-ish tool-call markup used by LFM2/LFM3 checkpoints."""

    name = "lfm3"
    marker_opens: tuple[str, ...] = (_LFM_OPEN,)

    TOOL_BLOCK_RE = re.compile(
        rf"{re.escape(_LFM_OPEN)}\s*<function=([^>\r\n]+)>(.*?)</function>\s*{re.escape(_LFM_CLOSE)}",
        flags=re.DOTALL,
    )
    PARAMETER_RE = re.compile(
        r"<parameter=([^>\r\n]+)>\r?\n?(.*?)\r?\n?</parameter>",
        flags=re.DOTALL,
    )

    @staticmethod
    def _parse_parameter_value(raw: str) -> Any:
        # Strip template separator newlines without changing meaningful user text.
        if raw.startswith("\r\n"):
            raw = raw[2:]
        elif raw.startswith("\n"):
            raw = raw[1:]
        if raw.endswith("\r\n"):
            raw = raw[:-2]
        elif raw.endswith("\n"):
            raw = raw[:-1]

        stripped = raw.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return raw

    @classmethod
    def parse(cls, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Split generated text into visible content and OpenAI tool_calls."""
        tool_calls: list[dict[str, Any]] = []

        for match in cls.TOOL_BLOCK_RE.finditer(text):
            function_name = match.group(1).strip()
            arguments: dict[str, Any] = {}

            for parameter_match in cls.PARAMETER_RE.finditer(match.group(2)):
                parameter_name = parameter_match.group(1).strip()
                arguments[parameter_name] = cls._parse_parameter_value(parameter_match.group(2))

            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )

        visible_text = cls.TOOL_BLOCK_RE.sub("", text).strip()
        return visible_text, tool_calls


class HermesToolCallParser:
    """Parse JSON tool-call blocks used by Qwen/Hermes-style models."""

    name = "hermes"
    marker_opens: tuple[str, ...] = (_HERMES_OPEN,)

    TOOL_BLOCK_RE = re.compile(
        rf"{re.escape(_HERMES_OPEN)}\s*(.*?)\s*{re.escape(_HERMES_CLOSE)}",
        flags=re.DOTALL,
    )

    @staticmethod
    def _parse_payload(payload: str) -> dict[str, Any]:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("tool call payload must be a JSON object")
        return data

    @classmethod
    def parse(cls, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Split generated text into visible content and OpenAI tool_calls."""
        tool_calls: list[dict[str, Any]] = []
        visible_parts: list[str] = []
        cursor = 0

        for match in cls.TOOL_BLOCK_RE.finditer(text):
            visible_parts.append(text[cursor : match.start()])
            cursor = match.end()
            try:
                payload = cls._parse_payload(match.group(1))
            except (json.JSONDecodeError, ValueError):
                visible_parts.append(match.group(0))
                continue

            name = str(payload.get("name", ""))
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw_arguments": arguments}
            if not isinstance(arguments, dict):
                arguments = {"_value": arguments}

            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )

        visible_parts.append(text[cursor:])
        visible_text = "".join(visible_parts).strip()
        return visible_text, tool_calls


class NoneToolCallParser:
    """Passthrough parser for models without tool-call markup."""

    name = "none"
    marker_opens: tuple[str, ...] = ()

    @staticmethod
    def parse(text: str) -> tuple[str, list[dict[str, Any]]]:
        return text, []


_PARSERS = {
    Lfm3ToolCallParser.name: Lfm3ToolCallParser,
    HermesToolCallParser.name: HermesToolCallParser,
    NoneToolCallParser.name: NoneToolCallParser,
}

TOOL_PARSER_NAMES = frozenset(_PARSERS) | {"auto"}


def resolve_tool_call_parser(value: str | None, model_hint: str = "") -> Any:
    """Return a parser instance by explicit name or guess one from the model hint."""
    name = (value or "auto").strip().lower()
    if name != "auto":
        parser_cls = _PARSERS.get(name)
        if parser_cls is None:
            raise ValueError(f"Unknown tool_parser '{value}'. Expected one of: {sorted(TOOL_PARSER_NAMES)}")
        return parser_cls()

    hint = model_hint.lower()
    if "lfm" in hint:
        return Lfm3ToolCallParser()
    if "qwen" in hint or "hermes" in hint or "glm" in hint:
        return HermesToolCallParser()
    return NoneToolCallParser()
