from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from npu_server.cli.model_config_cli import CONFIG_FILENAME, ModelConfig


class ModelArgumentsParser:
    """Read model_configuration.json and apply command-line overrides onto ModelConfig."""

    @staticmethod
    def _load_config() -> tuple[dict[str, Any] | None, Path]:
        """Load config from cwd, then its parent; otherwise return the cwd target path."""
        cwd = Path.cwd().resolve()
        candidates = (cwd / CONFIG_FILENAME, cwd.parent / CONFIG_FILENAME)

        for config_path in candidates:
            if config_path.is_file():
                try:
                    raw_config: Any = json.loads(config_path.read_text(encoding="utf-8-sig"))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in configuration file {config_path}: {exc}") from exc

                if not isinstance(raw_config, dict):
                    raise ValueError(f"Configuration root must be a JSON object: {config_path}")
                return cast(dict[str, Any], raw_config), config_path

        return None, candidates[0]

    @staticmethod
    def _parse_int_or_auto(value: str) -> int | Literal["auto"]:
        if value.strip().lower() == "auto":
            return "auto"
        try:
            return int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Expected a positive integer or 'auto'") from exc

    @staticmethod
    def get_model_params(argv: Sequence[str] | None = None) -> ModelConfig:
        config_data, config_path = ModelArgumentsParser._load_config()
        usable_config = {key: value for key, value in (config_data or {}).items() if value is not None}
        defaults = ModelConfig.model_validate(usable_config)

        parser = argparse.ArgumentParser(description="Configuration options for the local OpenVINO model server")

        parser.add_argument(
            "--model-id",
            "--model",
            "-m",
            dest="model_id",
            type=str,
            default=defaults.model_id,
            help="Hugging Face repository ID or local model directory",
        )
        parser.add_argument("--model-dir", type=Path, default=defaults.model_dir, help="Path to save/load model")
        parser.add_argument(
            "--model-name",
            dest="model_name",
            type=str,
            default=defaults.model_name,
            help="Model name exposed by the /v1/models endpoint",
        )
        parser.add_argument("--device", default=defaults.device)
        parser.add_argument(
            "--tool-parser",
            dest="tool_parser",
            default=defaults.tool_parser,
            help="Tool call markup converter: auto, lfm3, hermes, none",
        )
        parser.add_argument(
            "--enable-thinking",
            dest="enable_thinking",
            action=argparse.BooleanOptionalAction,
            default=defaults.enable_thinking,
            help="Pass enable_thinking to the chat template",
        )
        parser.add_argument(
            "--full-attention-only",
            dest="full_attention_only",
            action=argparse.BooleanOptionalAction,
            default=defaults.full_attention_only,
            help="Unify all attention layers to full_attention before export (NPU/NPUW fix)",
        )
        parser.add_argument("--host", default=defaults.host)
        parser.add_argument("--port", type=int, default=defaults.port)
        parser.add_argument(
            "--max-prompt-len",
            type=ModelArgumentsParser._parse_int_or_auto,
            default=defaults.max_prompt_len,
        )
        parser.add_argument(
            "--min-response-len",
            type=ModelArgumentsParser._parse_int_or_auto,
            default=defaults.min_response_len,
        )
        parser.add_argument(
            "--max-output-tokens",
            type=ModelArgumentsParser._parse_int_or_auto,
            default=defaults.max_output_tokens,
        )

        args = parser.parse_args(argv)
        validated_config = ModelConfig.model_validate(vars(args))
        validated_config.save_config(config_path)

        return validated_config
