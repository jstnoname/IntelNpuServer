from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from npu_server.cli.model_config_cli import ModelConfig


class GenerationError(ValueError):
    """Raised when a generation request cannot be served by the configured pipeline."""


class ModelRuntime(BaseModel):
    """Loaded OpenVINO GenAI pipeline together with its Hugging Face tokenizer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Any
    pipeline: Any


class RuntimeLengths:
    """Resolve length options; 'auto' on NPU applies safe defaults with a fallback ladder."""

    NPU_LENGTH_LADDER: tuple[tuple[int, int], ...] = (
        (8192, 1024),
        (4096, 1024),
        (2048, 512),
        (1024, 256),
    )

    def __init__(self, config: ModelConfig) -> None:
        device = config.device.upper()
        on_npu = device == "NPU" or device.startswith("NPU.")
        self.auto_applied = on_npu and config.max_prompt_len == "auto" and config.min_response_len == "auto"
        if self.auto_applied:
            self.max_prompt_len = self.NPU_LENGTH_LADDER[0][0]
            self.min_response_len = self.NPU_LENGTH_LADDER[0][1]
        else:
            self.max_prompt_len = self._strict_or_none(config.max_prompt_len)
            self.min_response_len = self._strict_or_none(config.min_response_len)
        self.max_output_tokens = self._strict_or_none(config.max_output_tokens)

    @staticmethod
    def _strict_or_none(value: int | Literal["auto"]) -> int | None:
        return None if value == "auto" else int(value)

    def compile_lengths(self) -> tuple[tuple[int | None, int | None], ...]:
        """Length rungs to try during pipeline compilation (prompt len, response len)."""
        if self.auto_applied:
            return self.NPU_LENGTH_LADDER
        return ((self.max_prompt_len, self.min_response_len),)

    def pipeline_properties(
        self,
        cache_dir: Path,
        *,
        max_prompt_len: int | None = None,
        min_response_len: int | None = None,
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {"CACHE_DIR": str(cache_dir.resolve())}
        prompt_len = self.max_prompt_len if max_prompt_len is None else max_prompt_len
        response_len = self.min_response_len if min_response_len is None else min_response_len
        if prompt_len is not None:
            properties["MAX_PROMPT_LEN"] = prompt_len
        if response_len is not None:
            properties["MIN_RESPONSE_LEN"] = response_len
        return properties

    def verify_prompt_length(self, prompt_tokens: int) -> None:
        if self.max_prompt_len is None or prompt_tokens <= self.max_prompt_len:
            return
        raise GenerationError(
            f"Prompt has {prompt_tokens} tokens, but the pipeline was compiled for "
            f"MAX_PROMPT_LEN={self.max_prompt_len}. Increase --max-prompt-len and restart."
        )

    def cap_new_tokens(self, requested: int | None) -> int | None:
        """None input keeps the library default generation length."""
        if requested is None or self.max_output_tokens is None:
            return requested
        return min(requested, self.max_output_tokens)
