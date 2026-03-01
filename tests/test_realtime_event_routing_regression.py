from __future__ import annotations

import asyncio
from typing import Any

from ai.realtime.event_router import EventRouter
from ai.realtime_api import RealtimeAPI


TARGET_EVENT_TYPES = {
    "response.created",
    "response.done",
    "response.output_audio.delta",
    "conversation.item.input_audio_transcription.completed",
}


def _make_router_only_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._event_router = EventRouter(
        fallback=api._handle_unknown_event,
        on_exception=api._on_event_handler_exception,
    )
    return api


def test_configure_event_router_registers_explicit_handlers_for_high_frequency_events() -> None:
    api = _make_router_only_api()

    api._configure_event_router()

    assert api._event_router._handlers["response.created"] == api._handle_response_created_event
    assert api._event_router._handlers["response.done"] == api._handle_response_done_event
    assert api._event_router._handlers["response.output_audio.delta"] == api._handle_response_output_audio_delta_event
    assert (
        api._event_router._handlers["conversation.item.input_audio_transcription.completed"]
        == api._handle_input_audio_transcription_completed_event
    )


def test_high_frequency_events_do_not_route_through_legacy_handler() -> None:
    api = _make_router_only_api()
    calls: list[str] = []

    async def _legacy_handler(_event: dict[str, Any], _websocket: Any) -> None:
        raise AssertionError("legacy handler should not run for explicit high-frequency events")

    async def _handle_response_created_event(event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        calls.append(str(event.get("type") or ""))

    async def _handle_response_done_event(event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        calls.append(str(event.get("type") or ""))

    async def _handle_response_output_audio_delta_event(event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        calls.append(str(event.get("type") or ""))

    async def _handle_input_audio_transcription_completed_event(event: dict[str, Any], websocket: Any) -> None:
        _ = websocket
        calls.append(str(event.get("type") or ""))

    api._handle_event_legacy = _legacy_handler
    api._handle_response_created_event = _handle_response_created_event
    api._handle_response_done_event = _handle_response_done_event
    api._handle_response_output_audio_delta_event = _handle_response_output_audio_delta_event
    api._handle_input_audio_transcription_completed_event = _handle_input_audio_transcription_completed_event

    api._configure_event_router()

    async def _run() -> None:
        for event_type in TARGET_EVENT_TYPES:
            await api.handle_event({"type": event_type}, websocket=None)

    asyncio.run(_run())

    assert sorted(calls) == sorted(TARGET_EVENT_TYPES)
