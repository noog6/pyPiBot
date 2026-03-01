from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


class FunctionCallAccumulator:
    """Owns function-call event accumulation and delegates execution decisions."""

    def __init__(
        self,
        *,
        on_function_call_item: Callable[[dict[str, Any]], None],
        on_assistant_message_item: Callable[[dict[str, Any]], None],
        on_arguments_done: Callable[[dict[str, Any], Any], Awaitable[None]],
    ) -> None:
        self._on_function_call_item = on_function_call_item
        self._on_assistant_message_item = on_assistant_message_item
        self._on_arguments_done = on_arguments_done
        self._arguments_buffer = ""

    @property
    def arguments_buffer(self) -> str:
        return self._arguments_buffer

    def reset_arguments_buffer(self) -> None:
        self._arguments_buffer = ""

    async def handle_output_item_added(self, event: dict[str, Any]) -> None:
        item = event.get("item", {})
        if isinstance(item, dict) and item.get("type") == "function_call":
            self._on_function_call_item(item)
            self.reset_arguments_buffer()
            return
        if isinstance(item, dict):
            self._on_assistant_message_item(item)

    def handle_function_call_arguments_delta(self, event: dict[str, Any]) -> None:
        self._arguments_buffer += str(event.get("delta", ""))

    async def handle_function_call_arguments_done(self, event: dict[str, Any], websocket: Any) -> None:
        await self._on_arguments_done(event, websocket)
