"""Unit tests for realtime inbound event dispatch."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

from ai.realtime.event_router import EventRouter
from ai.realtime_api import RealtimeAPI


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


def test_realtime_api_handle_event_dispatches_via_event_router_once() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    event = {"type": "response.created", "id": "evt-123"}
    websocket = object()

    async def _fallback(_event: dict[str, Any], _websocket: Any) -> None:
        return None

    router = EventRouter(fallback=_fallback)
    router.dispatch = AsyncMock(return_value=None)
    api._event_router = router
    api._maybe_handle_confirmation_decision_timeout = AsyncMock(return_value=None)

    asyncio.run(api.handle_event(event, websocket))

    router.dispatch.assert_awaited_once_with(event, websocket)


def test_realtime_api_handle_event_triggers_recovery_when_router_dispatch_raises() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    event = {"type": "response.done", "id": "evt-err"}
    websocket = object()

    async def _fallback(_event: dict[str, Any], _websocket: Any) -> None:
        return None

    router = EventRouter(fallback=_fallback)
    router.dispatch = AsyncMock(side_effect=RuntimeError("router exploded"))
    api._event_router = router
    api._maybe_handle_confirmation_decision_timeout = AsyncMock(return_value=None)
    api._recover_from_event_handler_error = AsyncMock(return_value=None)

    with (
        patch("ai.realtime_api.logger.exception") as mock_exception,
        patch("ai.realtime_api.logger.error") as mock_error,
    ):
        asyncio.run(api.handle_event(event, websocket))

    router.dispatch.assert_awaited_once_with(event, websocket)
    api._recover_from_event_handler_error.assert_awaited_once_with("response.done", websocket)
    assert mock_exception.call_count == 1
    mock_error.assert_called_once_with("EVENT_HANDLER_ERROR event=%s", "response.done")
