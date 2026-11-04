from __future__ import annotations

import gc
import queue
import threading
from collections.abc import Iterator
from typing import Any

import openvino_genai as ov_genai

from npu_server.cli.model_config_cli import ModelConfig
from npu_server.generation.events import GenerationOutcome, GenerationRequest, StreamEvent
from npu_server.generation.streaming import TokenStreamCallback, ToolSafeStreamBuffer
from npu_server.load_model.runtime import GenerationError, ModelRuntime, RuntimeLengths
from npu_server.tool_call_parsers import resolve_tool_call_parser


class GenerationEngine:
    """Single-request generation engine over an OpenVINO GenAI pipeline."""

    def __init__(self, config: ModelConfig, runtime: ModelRuntime) -> None:
        self._config = config
        self._runtime = runtime
        self._inference_lock = threading.Lock()
        self._lengths = RuntimeLengths(config)
        self._parser = resolve_tool_call_parser(
            config.tool_parser,
            model_hint=f"{config.model_id} {config.model_name}",
        )

    @property
    def tool_parser_name(self) -> str:
        return self._parser.name

    @property
    def lengths(self) -> RuntimeLengths:
        return self._lengths

    @property
    def inference_lock(self) -> threading.Lock:
        return self._inference_lock

    def release(self) -> str:
        """Drop the pipeline reference so the device context and its memory are freed."""
        with self._inference_lock:
            if self._runtime.pipeline is None:
                return "already-released"
            self._runtime.pipeline = None
        gc.collect()
        return "released"

    def prepare(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> GenerationRequest:
        """Validate limits, apply the chat template and build the generation config."""
        prompt = self._build_prompt(messages, tools)
        prompt_tokens = self._count_tokens(prompt)
        self._lengths.verify_prompt_length(prompt_tokens)

        generation_config = ov_genai.GenerationConfig()
        capped = self._resolve_max_new_tokens(max_tokens)
        if capped is not None:
            generation_config.max_new_tokens = int(capped)
        generation_config.apply_chat_template = False
        self._apply_sampling(generation_config, temperature, top_p, top_k)

        return GenerationRequest(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            generation_config=generation_config,
        )

    def generate(self, request: GenerationRequest) -> GenerationOutcome:
        """Run blocking generation and return the neutral outcome."""
        with self._inference_lock:
            results = self._runtime.pipeline.generate(request.prompt, request.generation_config)
        full_text = str(results.texts[0]) if hasattr(results, "texts") else str(results)
        visible_text, tool_calls = self._parser.parse(full_text)
        return GenerationOutcome(
            content_text=visible_text if visible_text else None,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            prompt_tokens=request.prompt_tokens,
            completion_tokens=self._count_tokens(full_text),
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        """Generate on a worker thread while yielding neutral stream events."""
        fragments: queue.SimpleQueue[str | None] = queue.SimpleQueue()
        stop_event = threading.Event()
        stream_buffer = ToolSafeStreamBuffer(self._parser.marker_opens)
        decoder = lambda token_ids: self._runtime.tokenizer.decode(token_ids, skip_special_tokens=True)
        streamer = TokenStreamCallback(stream_buffer, stop_event, fragments, decoder)
        outcome: dict[str, Any] = {}

        worker = threading.Thread(
            target=self._stream_worker,
            args=(request, streamer, outcome, fragments),
            name="ov-genai-stream",
            daemon=True,
        )
        worker.start()

        while True:
            fragment = fragments.get()
            if fragment is None:
                break
            yield StreamEvent(kind="text", text=fragment)

        stop_event.set()
        worker.join(timeout=30)

        results = outcome.get("results")
        if results is None:
            yield StreamEvent(kind="error", text=str(outcome.get("error", "unknown failure")))
            yield StreamEvent(kind="finish", finish_reason="stop")
            return

        held_visible, tool_calls = self._parser.parse(stream_buffer.held_text())
        if held_visible:
            yield StreamEvent(kind="text", text=held_visible)
        if tool_calls:
            yield StreamEvent(kind="tool_calls", tool_calls=tool_calls)
        yield StreamEvent(kind="finish", finish_reason="tool_calls" if tool_calls else "stop")

    def _stream_worker(
        self,
        request: GenerationRequest,
        streamer: TokenStreamCallback,
        outcome: dict[str, Any],
        fragments: queue.SimpleQueue,
    ) -> None:
        try:
            with self._inference_lock:
                results = self._runtime.pipeline.generate(request.prompt, request.generation_config, streamer=streamer)
            outcome["results"] = results
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            fragments.put(None)

    def _build_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": self._config.enable_thinking,
        }
        if tools:
            template_kwargs["tools"] = tools

        try:
            prompt_value = self._runtime.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
            )
            if not isinstance(prompt_value, str):
                raise TypeError("The tokenizer chat template did not return a string prompt")
        except Exception as exc:
            raise GenerationError(f"Could not apply the chat template: {exc}") from exc
        return prompt_value

    def _resolve_max_new_tokens(self, max_tokens: Any) -> int | None:
        if max_tokens is None:
            return None
        try:
            requested = int(max_tokens)
        except (TypeError, ValueError) as exc:
            raise GenerationError("max_tokens must be an integer") from exc
        if requested < 1:
            raise GenerationError("max_tokens must be positive")
        return self._lengths.cap_new_tokens(requested)

    def _apply_sampling(
        self,
        generation_config: ov_genai.GenerationConfig,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
    ) -> None:
        try:
            temperature = 0.2 if temperature is None else float(temperature)
            top_p = 0.95 if top_p is None else float(top_p)
        except (TypeError, ValueError) as exc:
            raise GenerationError("temperature and top_p must be numeric") from exc

        if temperature > 0.0:
            generation_config.do_sample = True
            generation_config.temperature = temperature
            generation_config.top_p = min(max(top_p, 0.0), 1.0)
            if top_k is not None:
                try:
                    top_k_int = int(top_k)
                except (TypeError, ValueError) as exc:
                    raise GenerationError("top_k must be an integer") from exc
                if top_k_int > 0:
                    generation_config.top_k = top_k_int

    def _count_tokens(self, text: str) -> int:
        return len(self._runtime.tokenizer.encode(text, add_special_tokens=False))
