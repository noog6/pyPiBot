from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

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
    stub = AsyncMock()
    api._maybe_handle_confirmation_decision_timeout = AsyncMock(return_value=False)
    api._input_audio_events = SimpleNamespace(
        handle_input_audio_buffer_speech_started=stub,
        handle_input_audio_buffer_speech_stopped=stub,
        handle_input_audio_buffer_committed=stub,
        handle_input_audio_buffer_timeout_triggered=stub,
        handle_input_audio_transcription_partial=stub,
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


def test_cancelled_response_logs_deliverable_selected_once_for_terminal_events() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._cancelled_response_ids = {"resp-cancelled"}
    api._cancelled_deliverable_logged_ids = set()
    api.handle_response_done = AsyncMock()

    with patch("ai.realtime_api.logger.info") as info_log:
        asyncio.run(
            api._handle_event_legacy(
                {"type": "response.output_text.delta", "response_id": "resp-cancelled"},
                None,
            )
        )
        assert "resp-cancelled" in api._cancelled_response_ids
        asyncio.run(api._handle_response_done_event({"type": "response.done", "response_id": "resp-cancelled"}, None))
        asyncio.run(api._handle_event_legacy({"type": "response.completed", "response_id": "resp-cancelled"}, None))

    cancelled_calls = [
        call
        for call in info_log.call_args_list
        if call.args == (
            "deliverable_selected response_id=%s selected=false reason=cancelled",
            "resp-cancelled",
        )
    ]
    assert len(cancelled_calls) == 1
    assert "resp-cancelled" not in api._cancelled_response_ids


def test_response_id_from_event_prefers_explicit_over_nested_response_object() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    explicit = api._response_id_from_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-explicit",
            "response": {"id": "resp-nested"},
        }
    )
    fallback = api._response_id_from_event(
        {
            "type": "response.output_audio.delta",
            "response": {"id": "resp-nested-only"},
        }
    )

    assert explicit == "resp-explicit"
    assert fallback == "resp-nested-only"


def test_cancelled_audio_delta_is_suppressed_without_side_effects() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._cancelled_response_ids = {"resp-cancelled"}
    api._superseded_response_ids = set()
    api._current_run_id = lambda: "run-123"
    api._audio_accum = bytearray(b"seed")
    api._mic_receive_on_first_audio = True
    api._audio_playback_busy = False
    api._speaking_started = False
    api._is_active_response_guarded = lambda: False
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._cancel_micro_ack = lambda **_kwargs: None
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._canonical_response_state_mutate = lambda **_kwargs: None
    api._canonical_lifecycle_state = lambda _key: {}
    api._lifecycle_controller = lambda: SimpleNamespace(on_audio_delta=lambda _key: None)
    api._active_response_canonical_key = ""
    api.audio_player = SimpleNamespace(play_audio=AsyncMock())
    api.mic = SimpleNamespace(is_receiving=False, start_receiving=AsyncMock())
    api.state_manager = SimpleNamespace(update_state=AsyncMock())

    with patch("ai.realtime_api.logger.debug") as debug_log:
        asyncio.run(
            api._handle_response_output_audio_delta_event(
                {
                    "type": "response.output_audio.delta",
                    "response": {"id": "resp-cancelled"},
                    "delta": "c29tZV9hdWRpbw==",
                },
                None,
            )
        )

    assert api._audio_accum == bytearray(b"seed")
    assert api._mic_receive_on_first_audio is True
    assert api._audio_playback_busy is False
    assert api._speaking_started is False
    api.mic.start_receiving.assert_not_called()
    api.state_manager.update_state.assert_not_called()
    api.audio_player.play_audio.assert_not_called()
    debug_log.assert_any_call(
        "audio_delta_suppressed run_id=%s response_id=%s event_type=response.output_audio.delta",
        "run-123",
        "resp-cancelled",
    )


def test_cancelled_audio_events_capture_timing_and_emit_race_log_once() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._cancelled_response_ids = {"resp-cancelled"}
    api._superseded_response_ids = set()
    api._cancelled_response_timing_by_id = {
        "resp-cancelled": {
            "cancel_issued_at": 100.0,
            "first_audio_delta_seen_at": None,
            "output_audio_done_at": None,
            "race_logged": False,
        }
    }
    api._current_run_id = lambda: "run-123"
    api._audio_accum = bytearray()
    api._mic_receive_on_first_audio = True
    api._audio_playback_busy = False
    api._speaking_started = False
    api._is_active_response_guarded = lambda: False
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._cancel_micro_ack = lambda **_kwargs: None
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._canonical_response_state_mutate = lambda **_kwargs: None
    api._canonical_lifecycle_state = lambda _key: {}
    api._lifecycle_controller = lambda: SimpleNamespace(on_audio_delta=lambda _key: None)
    api._active_response_canonical_key = ""
    api.audio_player = SimpleNamespace(play_audio=AsyncMock())
    api.mic = SimpleNamespace(is_receiving=False, start_receiving=AsyncMock())
    api.state_manager = SimpleNamespace(update_state=AsyncMock())

    with patch("ai.realtime_api.time.time", side_effect=[100.25, 100.5]), patch("ai.realtime_api.logger.info") as info_log:
        asyncio.run(
            api._handle_response_output_audio_delta_event(
                {
                    "type": "response.output_audio.delta",
                    "response": {"id": "resp-cancelled"},
                    "delta": "c29tZV9hdWRpbw==",
                },
                None,
            )
        )
        asyncio.run(
            api._handle_event_legacy(
                {
                    "type": "response.output_audio.done",
                    "response": {"id": "resp-cancelled"},
                },
                None,
            )
        )

    timing = api._cancelled_response_timing_by_id["resp-cancelled"]
    assert timing["cancel_issued_at"] == 100.0
    assert timing["first_audio_delta_seen_at"] == 100.25
    assert timing["output_audio_done_at"] == 100.5
    race_logs = [
        call
        for call in info_log.call_args_list
        if call.args and isinstance(call.args[0], str) and call.args[0].startswith("cancel_audio_race_observed")
    ]
    assert len(race_logs) == 1


def test_cancelled_audio_done_without_delta_still_logs_race_with_na_delta() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._cancelled_response_ids = {"resp-cancelled"}
    api._superseded_response_ids = set()
    api._cancelled_response_timing_by_id = {
        "resp-cancelled": {
            "cancel_issued_at": 10.0,
            "first_audio_delta_seen_at": None,
            "output_audio_done_at": None,
            "race_logged": False,
        }
    }

    with patch("ai.realtime_api.time.time", return_value=10.4), patch("ai.realtime_api.logger.info") as info_log:
        asyncio.run(
            api._handle_event_legacy(
                {
                    "type": "response.output_audio.done",
                    "response_id": "resp-cancelled",
                },
                None,
            )
        )

    timing = api._cancelled_response_timing_by_id["resp-cancelled"]
    assert timing["output_audio_done_at"] == 10.4
    assert timing["first_audio_delta_seen_at"] is None
    race_logs = [
        call
        for call in info_log.call_args_list
        if call.args and isinstance(call.args[0], str) and call.args[0].startswith("cancel_audio_race_observed")
    ]
    assert len(race_logs) == 1
    assert race_logs[0].args[-1] == "na"
