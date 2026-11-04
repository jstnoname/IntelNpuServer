from __future__ import annotations

from pathlib import Path

import openvino as ov
import openvino_genai as ov_genai
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from npu_server.cli.model_config_cli import ModelConfig
from npu_server.load_model.preparer import OpenvinoModelPreparer
from npu_server.load_model.runtime import ModelRuntime, RuntimeLengths


class OpenvinoModelLoader:
    """Load an exported model into an OpenVINO GenAI pipeline on the configured device."""

    def __init__(self, preparer: OpenvinoModelPreparer | None = None) -> None:
        self._preparer = preparer if preparer is not None else OpenvinoModelPreparer()

    def load(
        self,
        config: ModelConfig,
        *,
        prepare_if_missing: bool = True,
        force_export: bool = False,
    ) -> ModelRuntime:
        """Ensure assets exist, verify the device and compile the GenAI pipeline."""
        exported_dir = self._preparer.exported_dir(config)
        if prepare_if_missing:
            self._preparer.ensure(config, force=force_export)
        elif not self._preparer.has_openvino_ir(exported_dir):
            raise FileNotFoundError(
                f"OpenVINO IR not found in {exported_dir}. Run `uv run python -m npu_server prepare` first."
            )

        tokenizer = self._ensure_tokenizer(config)
        lengths = RuntimeLengths(config)
        self._verify_device(config)

        cache_dir = config.model_dir / ".openvino_npu_cache" / self._preparer.model_folder_name(config)
        cache_dir.mkdir(parents=True, exist_ok=True)

        pipeline = self._compile(config, exported_dir, lengths, cache_dir)
        return ModelRuntime(tokenizer=tokenizer, pipeline=pipeline)

    @classmethod
    def _compile(
        cls,
        config: ModelConfig,
        exported_dir: Path,
        lengths: RuntimeLengths,
        cache_dir: Path,
    ):
        """Compile the pipeline, walking the fallback ladder when auto defaults are used."""
        device = config.device.upper()
        last_error: Exception | None = None
        for attempt, (max_prompt_len, min_response_len) in enumerate(lengths.compile_lengths()):
            if attempt:
                print(
                    f"[OpenVINO] Retrying with a smaller context: "
                    f"MAX_PROMPT_LEN={max_prompt_len}, MIN_RESPONSE_LEN={min_response_len}",
                    flush=True,
                )
            print(
                f"[OpenVINO] Compiling/loading {config.model_name} on {device} "
                f"(MAX_PROMPT_LEN={max_prompt_len or 'auto'}, MIN_RESPONSE_LEN={min_response_len or 'auto'}) ...",
                flush=True,
            )
            properties = lengths.pipeline_properties(
                cache_dir,
                max_prompt_len=max_prompt_len,
                min_response_len=min_response_len,
            )
            try:
                pipeline = ov_genai.LLMPipeline(str(exported_dir), device, properties)
                print("[OpenVINO] Model is ready.", flush=True)
                return pipeline
            except Exception as exc:
                last_error = exc
                print(f"[OpenVINO] Compilation failed: {exc}", flush=True)
                if not lengths.auto_applied:
                    break
        raise RuntimeError(f"Could not compile the model on {device}: {last_error}") from last_error

    @staticmethod
    def _verify_device(config: ModelConfig) -> None:
        available_devices = list(ov.Core().available_devices)
        print(f"[OpenVINO] Available devices: {available_devices}", flush=True)
        wanted = config.device.upper()
        if not any(device == wanted or device.startswith(wanted + ".") for device in available_devices):
            raise RuntimeError(
                f"OpenVINO device '{config.device}' is not available: {available_devices}. "
                "Check the drivers and `ov.Core().available_devices` in this environment."
            )

    @staticmethod
    def _load_tokenizer_checked(path: str) -> PreTrainedTokenizerBase:
        """Load a tokenizer, raising instead of returning the None allowed by transformers."""
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if tokenizer is None:
            raise RuntimeError(f"Tokenizer could not be loaded from '{path}'")
        return tokenizer

    @classmethod
    def _ensure_tokenizer(cls, config: ModelConfig) -> PreTrainedTokenizerBase:
        """Load the tokenizer from the exported dir, falling back to the local checkpoint."""
        exported_dir = OpenvinoModelPreparer.exported_dir(config)
        try:
            return OpenvinoModelLoader._load_tokenizer_checked(str(exported_dir))
        except (OSError, RuntimeError):
            source_dir = OpenvinoModelPreparer._resolve_hf_model(
                config.model_id, OpenvinoModelPreparer.checkpoint_dir(config)
            )
            tokenizer = OpenvinoModelLoader._load_tokenizer_checked(str(source_dir))
            tokenizer.save_pretrained(exported_dir)
            return tokenizer
