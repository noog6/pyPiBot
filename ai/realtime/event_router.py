"""Event routing helper for realtime API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

EventHandler = Callable[[dict[str, Any], Any], Awaitable[None]]
FallbackHandler = Callable[[dict[str, Any], Any], Awaitable[None]]
ExceptionHandler = Callable[[str, Exception], None]


class EventRouter:
    """Maps event type -> handler and applies fallback when no handler matches."""

    def __init__(
        self,
        *,
        fallback: FallbackHandler,
        on_exception: ExceptionHandler | None = None,
    ) -> None:
        self._fallback = fallback
        self._on_exception = on_exception
        self._handlers: dict[str, EventHandler] = {}

    def register(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type] = handler

    async def dispatch(self, event: dict[str, Any], websocket: Any) -> None:
        event_type = str(event.get("type") or "")
        handler = self._handlers.get(event_type)
        if handler is None:
            await self._fallback(event, websocket)
            return
        try:
            await handler(event, websocket)
        except Exception as exc:
            if self._on_exception is not None:
                self._on_exception(event_type, exc)
            raise


# Backward-compatible alias while migration lands in the larger realtime module.
RealtimeEventRouter = EventRouter
