from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from npu_server.tool_call_parsers import TOOL_PARSER_NAMES

CONFIG_FILENAME = "model_configuration.json"
MANIFEST_FILENAME = "model_manifest.json"


class ModelConfig(BaseModel):
    """Runtime configuration persisted in model_configuration.json."""

    model_config = ConfigDict(extra="ignore")

    model_id: str = Field(description="Hugging Face repository ID or local model directory (required)")
    model_dir: Path = Field(
        default=(Path.cwd().parent / "model").resolve(),
        description="Directory containing the exported OpenVINO IR and tokenizer",
    )
    model_name: str | None = Field(
        default=None,
        description="Model ID exposed by the /v1/models endpoint; defaults to model_id",
    )
    device: str = Field(
        description="OpenVINO device (NPU, CPU, AUTO, ...) (required)",
    )
    host: str = Field(default="127.0.0.1", description="HTTP server bind address")
    port: int = Field(default=8000, ge=1, le=65535)

    tool_parser: str = Field(
        default="auto",
        description="Tool call markup format: auto, lfm3, hermes, none",
    )
    enable_thinking: bool = Field(
        default=False,
        description="Pass enable_thinking to the chat template (thinking-capable models)",
    )
    full_attention_only: bool = Field(
        default=False,
        description=(
            "Convert sliding_attention layers to full_attention before export; "
            "required for hybrid sliding/full LFM checkpoints on Intel NPU (NPUW "
            "conflicting attention mask assertion)"
        ),
    )

    max_prompt_len: int | Literal["auto"] = Field(
        default="auto",
        description="Maximum prompt length for the NPU pipeline, or 'auto'",
    )
    min_response_len: int | Literal["auto"] = Field(
        default="auto",
        description="Minimum response length reserved by the NPU pipeline, or 'auto'",
    )
    max_output_tokens: int | Literal["auto"] = Field(
        default="auto",
        description="Maximum generated tokens per request, or 'auto'",
    )

    @field_validator("model_id", "device", "host")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value must not be empty")
        return value

    @field_validator("model_dir")
    @classmethod
    def normalize_model_dir(cls, value: Path) -> Path:
        value = Path(value).expanduser()
        value.mkdir(parents=True, exist_ok=True)
        return value.resolve()

    @field_validator("tool_parser")
    @classmethod
    def validate_tool_parser(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in TOOL_PARSER_NAMES:
            raise ValueError(f"tool_parser must be one of: {sorted(TOOL_PARSER_NAMES)}")
        return value

    @field_validator("max_prompt_len", "min_response_len", "max_output_tokens")
    @classmethod
    def validate_positive_lengths(cls, value: int | Literal["auto"]) -> int | Literal["auto"]:
        if (isinstance(value, int) and value <= 0) or (isinstance(value, str) and value != "auto"):
            raise ValueError("Length values must be positive integers or 'auto'")
        return value

    def save_config(self, path: str | Path) -> None:
        config_path = Path(path).expanduser().resolve()
        if config_path.suffix == ".json" and config_path.name == CONFIG_FILENAME:
            config_file = config_path
        else:
            config_file = config_path / CONFIG_FILENAME
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )

    @model_validator(mode="after")
    def derive_model_name(self) -> ModelConfig:
        if self.model_name is None:
            self.model_name = self.model_id
        return self
