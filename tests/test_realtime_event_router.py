"""Unit tests for realtime inbound event dispatch."""

from __future__ import annotations

import asyncio
from typing import Any

from ai.realtime.event_router import EventRouter


def test_dispatch_invokes_registered_handler() -> None:
    calls: list[tuple[str, dict[str, Any], Any]] = []

    async def _fallback(event: dict[str, Any], websocket: Any) -> None:
        calls.append(("fallback", event, websocket))

    async def _handler(event: dict[str, Any], websocket: Any) -> None:
        calls.append(("handler", event, websocket))

    router = EventRouter(fallback=_fallback)
    router.register("response.created", _handler)

    async def _run() -> None:
        await router.dispatch({"type": "response.created", "id": "evt-1"}, websocket="ws")

    asyncio.run(_run())

    assert calls == [("handler", {"type": "response.created", "id": "evt-1"}, "ws")]


def test_unknown_event_uses_fallback_stably() -> None:
    fallback_calls: list[str] = []

    async def _fallback(event: dict[str, Any], websocket: Any) -> None:
        fallback_calls.append(str(event.get("type") or "unknown"))

    router = EventRouter(fallback=_fallback)

    async def _run() -> None:
        await router.dispatch({"type": "unregistered.one"}, websocket=None)
        await router.dispatch({"type": "unregistered.two"}, websocket=None)

    asyncio.run(_run())

    assert fallback_calls == ["unregistered.one", "unregistered.two"]


def test_handler_exception_triggers_exception_callback_once() -> None:
    exception_calls: list[tuple[str, str]] = []

    async def _fallback(event: dict[str, Any], websocket: Any) -> None:
        raise AssertionError("fallback should not execute for registered handler")

    def _on_exception(event_type: str, exc: Exception) -> None:
        exception_calls.append((event_type, str(exc)))

    async def _handler(event: dict[str, Any], websocket: Any) -> None:
        raise RuntimeError("boom")

    router = EventRouter(fallback=_fallback, on_exception=_on_exception)
    router.register("response.done", _handler)

    async def _run() -> None:
        try:
            await router.dispatch({"type": "response.done"}, websocket=None)
        except RuntimeError:
            return
        raise AssertionError("dispatch should re-raise handler exceptions")

    asyncio.run(_run())

    assert exception_calls == [("response.done", "boom")]
