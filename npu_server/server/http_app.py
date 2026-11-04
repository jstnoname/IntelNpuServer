from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Iterator
from typing import Any

import openvino as ov
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from npu_server.cli.model_config_cli import ModelConfig
from npu_server.generation.engine import GenerationEngine
from npu_server.generation.events import GenerationOutcome, GenerationRequest, StreamEvent
from npu_server.load_model.loader import OpenvinoModelLoader
from npu_server.load_model.preparer import OpenvinoModelPreparer
from npu_server.load_model.runtime import GenerationError, ModelRuntime
from npu_server.server.message_normalizer import ChatMessageNormalizer
from npu_server.server.openai_schemas import ChatCompletionChunkBuilder, CompletionResult

SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive"}


class NpuServer:
    """FastAPI application plus the uvicorn runner for the OpenAI-compatible API."""

    def __init__(
        self,
        config: ModelConfig,
        runtime: ModelRuntime | None = None,
        *,
        prepare_if_missing: bool = True,
        force_export: bool = False,
    ) -> None:
        self.config = config
        if runtime is None:
            runtime = OpenvinoModelLoader().load(
                config,
                prepare_if_missing=prepare_if_missing,
                force_export=force_export,
            )
        self.engine = GenerationEngine(config, runtime)

        @contextlib.asynccontextmanager
        async def _lifespan(_app: FastAPI):
            yield
            self._release_model()

        self.app = FastAPI(title="OpenVINO GenAI NPU API", version="0.6.0", lifespan=_lifespan)
        self._register_routes()

    def run(self) -> None:
        """Serve the API until the process is interrupted, releasing the device on exit."""
        print(
            f"[server] {self.config.model_name} is listening at http://{self.config.host}:{self.config.port}",
            flush=True,
        )
        print("[server] Press Ctrl+C to stop.", flush=True)
        try:
            uvicorn.run(self.app, host=self.config.host, port=self.config.port, log_level="info", workers=1)
        finally:
            self._release_model()

    def _release_model(self) -> None:
        """Unload the model from the device during server shutdown."""
        try:
            status = self.engine.release()
            if status == "released":
                print("[OpenVINO] Model released from the device.", flush=True)
        except Exception as exc:
            print(f"[OpenVINO] Model release warning: {exc}", flush=True)

    def _register_routes(self) -> None:
        engine = self.engine

        @self.app.get("/health")
        def health() -> dict[str, Any]:
            return self._health_payload()

        @self.app.get("/v1/models")
        def list_models() -> dict[str, Any]:
            return self._models_payload()

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            payload = await self._read_body(request)
            messages = self._validated_messages(payload)
            tools = self._endpoint_tools(payload)
            options = self._endpoint_sampling(payload)

            try:
                prepared: GenerationRequest = await run_in_threadpool(engine.prepare, messages, tools, **options)
                if payload.get("stream", False):
                    return StreamingResponse(
                        self._sse_frames(engine.stream(prepared)),
                        media_type="text/event-stream",
                        headers=SSE_HEADERS,
                    )
                outcome = await run_in_threadpool(engine.generate, prepared)
            except HTTPException:
                raise
            except GenerationError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc
            return JSONResponse(self._response_body(outcome))

    def _health_payload(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.config.model_name,
            "hf_model": self.config.model_id,
            "model_dir": str(OpenvinoModelPreparer.exported_dir(self.config)),
            "device": self.config.device.upper(),
            "tool_parser": self.engine.tool_parser_name,
            "available_devices": list(ov.Core().available_devices),
            "max_prompt_len": self.engine.lengths.max_prompt_len,
            "min_response_len": self.engine.lengths.min_response_len,
            "max_output_tokens": self.engine.lengths.max_output_tokens,
        }

    def _models_payload(self) -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": self.config.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @staticmethod
    async def _read_body(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        if payload.get("n", 1) not in (1, None):
            raise HTTPException(status_code=400, detail="This server supports n=1 only")
        return payload

    @staticmethod
    def _validated_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="`messages` must be a non-empty array")
        return ChatMessageNormalizer.normalize(messages)

    @staticmethod
    def _endpoint_tools(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
        if payload.get("tool_choice") == "none":
            return None
        return payload.get("tools") or None

    @staticmethod
    def _endpoint_sampling(payload: dict[str, Any]) -> dict[str, Any]:
        max_tokens = payload.get("max_tokens")
        if max_tokens is None:
            max_tokens = payload.get("max_completion_tokens")
        return {
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "top_k": payload.get("top_k"),
            "max_tokens": max_tokens,
        }

    def _sse_frames(self, events: Iterator[StreamEvent]) -> Iterator[str]:
        builder = ChatCompletionChunkBuilder(self.config.model_name or "model")
        for event in events:
            if event.kind == "text":
                yield builder.event({"content": event.text})
            elif event.kind == "tool_calls":
                for index, tool_call in enumerate(event.tool_calls or []):
                    yield builder.event({"tool_calls": [{"index": index, **tool_call}]})
            elif event.kind == "finish":
                yield builder.event({}, finish_reason=event.finish_reason or "stop")
            elif event.kind == "error":
                yield builder.event({"content": f"\n[generation failed: {event.text}]"})
        yield builder.done()

    def _response_body(self, outcome: GenerationOutcome) -> dict[str, Any]:
        result = CompletionResult(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=self.config.model_name or "model",
            content=outcome.content_text,
            tool_calls=outcome.tool_calls,
            finish_reason=outcome.finish_reason,
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
        )
        return result.response_body()
