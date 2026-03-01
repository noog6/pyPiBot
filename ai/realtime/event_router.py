"""Event routing helper for realtime API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

EventHandler = Callable[[dict[str, Any], Any], Awaitable[bool]]
FallbackHandler = Callable[[dict[str, Any], Any], Awaitable[None]]


class RealtimeEventRouter:
    """Maps event type -> handler and applies fallback when no handler matches."""

    def __init__(self, *, fallback: FallbackHandler) -> None:
        self._fallback = fallback
        self._handlers: dict[str, EventHandler] = {}

    def register(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type] = handler

    async def dispatch(self, event: dict[str, Any], websocket: Any) -> bool:
        event_type = str(event.get("type") or "")
        handler = self._handlers.get(event_type)
        if handler is None:
            await self._fallback(event, websocket)
            return False
        handled = await handler(event, websocket)
        if not handled:
            await self._fallback(event, websocket)
        return handled
