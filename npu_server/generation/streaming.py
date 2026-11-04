from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence

import openvino_genai as ov_genai


class ToolSafeStreamBuffer:
    """Withhold streamed fragments that could belong to a tool-call block."""

    def __init__(self, marker_opens: tuple[str, ...] = ()) -> None:
        self._marker_opens = marker_opens
        self._max_marker_len = max((len(marker) for marker in marker_opens), default=0)
        self._buffer = ""
        self._holding = False

    def feed(self, fragment: str) -> list[str]:
        """Accept a token piece and return the fragments safe to emit now."""
        self._buffer += fragment
        if self._holding:
            return []

        if any(marker in self._buffer for marker in self._marker_opens):
            self._holding = True
            return []

        keep = self._marker_prefix_length()
        if keep == 0:
            released, self._buffer = self._buffer, ""
        else:
            released, self._buffer = self._buffer[:-keep], self._buffer[-keep:]
        return [released] if released else []

    def held_text(self) -> str:
        """Return everything still withheld (potential tool-call block tail)."""
        return self._buffer

    def _marker_prefix_length(self) -> int:
        for length in range(min(len(self._buffer), self._max_marker_len), 0, -1):
            suffix = self._buffer[-length:]
            if any(marker.startswith(suffix) for marker in self._marker_opens):
                return length
        return 0


class TokenStreamCallback(ov_genai.StreamerBase):
    """Bridge OpenVINO streamed token ids into a queue through a tool-safe buffer."""

    def __init__(
        self,
        stream_buffer: ToolSafeStreamBuffer,
        stop_event: threading.Event,
        sink: queue.SimpleQueue,
        decoder: Callable[[list[int]], str],
    ) -> None:
        super().__init__()
        self._buffer = stream_buffer
        self._stop_event = stop_event
        self._sink = sink
        self._decoder = decoder
        self._token_history: list[int] = []
        self._previous_text = ""

    @staticmethod
    def _as_id_list(token: int | Sequence[int]) -> list[int]:
        if isinstance(token, (list, tuple)):
            return [int(value) for value in token]
        return [int(token)]

    def write(self, token: int | Sequence[int]) -> ov_genai.StreamingStatus:
        if self._stop_event.is_set():
            return ov_genai.StreamingStatus.STOP
        self._token_history.extend(self._as_id_list(token))
        full_text = self._decoder(self._token_history)
        delta = full_text[len(self._previous_text) :] if full_text.startswith(self._previous_text) else full_text
        self._previous_text = full_text
        if delta:
            for fragment in self._buffer.feed(delta):
                self._sink.put(fragment)
        return ov_genai.StreamingStatus.RUNNING

    def end(self) -> None:
        return None
