from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
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
    api._suppressed_audio_response_ids = set()
    api._superseded_response_ids = set()
    api._current_run_id = lambda: "run-123"
    api._audio_accum = bytearray(b"seed")
    api._audio_accum_response_id = "resp-other"
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
    api._suppressed_audio_response_ids = set()
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
    api._suppressed_audio_response_ids = set()
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


def test_suppress_cancelled_response_audio_clears_accumulator_and_flushes_player() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._suppressed_audio_response_ids = set()
    api._audio_accum = bytearray(b"stale")
    api._audio_accum_response_id = "resp-x"
    player = SimpleNamespace(cancel_current_response=AsyncMock())
    player.cancel_current_response = lambda: setattr(player, "cancelled", True)
    player.cancelled = False
    api.audio_player = player

    api._suppress_cancelled_response_audio("resp-x")

    assert "resp-x" in api._suppressed_audio_response_ids
    assert api._audio_accum == bytearray()
    assert api._audio_accum_response_id is None
    assert player.cancelled is True


def test_suppressed_response_audio_delta_is_dropped_without_enqueue() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._cancelled_response_ids = set()
    api._suppressed_audio_response_ids = {"resp-x"}
    api._superseded_response_ids = set()
    api._current_run_id = lambda: "run-123"
    api._audio_accum = bytearray()
    api._audio_accum_response_id = None
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

    asyncio.run(
        api._handle_response_output_audio_delta_event(
            {
                "type": "response.output_audio.delta",
                "response": {"id": "resp-x"},
                "delta": "c29tZV9hdWRpbw==",
            },
            None,
        )
    )

    assert api._audio_playback_busy is False
    assert api._audio_accum == bytearray()
    api.audio_player.play_audio.assert_not_called()


def test_stale_response_done_event_is_dropped() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._stale_response_ids_set = {"resp-stale"}
    api._cancelled_response_ids = set()
    api._suppressed_audio_response_ids = set()
    api._superseded_response_ids = set()
    api.handle_response_done = AsyncMock()

    with patch("ai.realtime_api.logger.debug") as debug_log:
        asyncio.run(api._handle_response_done_event({"type": "response.done", "response_id": "resp-stale"}, None))

    api.handle_response_done.assert_not_called()
    debug_log.assert_any_call(
        "dropped_stale_response_event response_id=%s event_type=%s",
        "resp-stale",
        "response.done",
    )


def test_stale_response_audio_transcript_delta_is_dropped_without_merge() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._stale_response_ids_set = {"resp-stale"}
    api._cancelled_response_ids = set()
    api._suppressed_audio_response_ids = set()
    api._superseded_response_ids = set()
    api._is_active_response_guarded = lambda: False
    api._mark_utterance_info_summary = AsyncMock()
    api._mark_first_assistant_utterance_observed_if_needed = AsyncMock()
    api._append_assistant_reply_text = AsyncMock()

    with patch("ai.realtime_api.logger.debug") as debug_log:
        asyncio.run(
            api._handle_event_legacy(
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": "resp-stale",
                    "delta": "hello",
                },
                None,
            )
        )

    api._mark_utterance_info_summary.assert_not_called()
    api._mark_first_assistant_utterance_observed_if_needed.assert_not_called()
    api._append_assistant_reply_text.assert_not_called()
    debug_log.assert_any_call(
        "dropped_stale_response_event response_id=%s event_type=%s",
        "resp-stale",
        "response.output_audio_transcript.delta",
    )


def test_stale_response_output_audio_done_is_dropped_without_playback_completion() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._stale_response_ids_set = {"resp-stale"}
    api._cancelled_response_ids = set()
    api._suppressed_audio_response_ids = set()
    api._superseded_response_ids = set()
    api.handle_audio_response_done = AsyncMock()
    api.state_manager = SimpleNamespace(update_state=AsyncMock())

    with patch("ai.realtime_api.logger.debug") as debug_log:
        asyncio.run(
            api._handle_event_legacy(
                {
                    "type": "response.output_audio.done",
                    "response_id": "resp-stale",
                },
                None,
            )
        )

    api.handle_audio_response_done.assert_not_called()
    api.state_manager.update_state.assert_not_called()
    debug_log.assert_any_call(
        "dropped_stale_response_event response_id=%s event_type=%s",
        "resp-stale",
        "response.output_audio.done",
    )


def test_dropped_stale_response_event_is_rate_limited() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._stale_response_ids_set = {"resp-stale"}
    api._cancelled_response_ids = set()
    api._suppressed_audio_response_ids = set()
    api._superseded_response_ids = set()
    api._stale_response_drop_window_by_id = {}
    api._stale_response_drop_window_s = 3.0

    with patch("ai.realtime_api.logger.info") as info_log:
        assert api._should_drop_stale_response_event({"type": "response.done", "response_id": "resp-stale"}) is True
        assert api._should_drop_stale_response_event({"type": "response.done", "response_id": "resp-stale"}) is True
        assert api._should_drop_stale_response_event({"type": "response.done", "response_id": "resp-stale"}) is True

    assert info_log.call_count == 1
    info_log.assert_called_with(
        "dropped_stale_response_event_summary response_id=%s counts=%s window_s=%s",
        "resp-stale",
        '{"response.done": 2}',
        3,
    )


def test_transcript_final_skips_verify_on_risk_when_confirmation_active() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    verify_gate = AsyncMock(return_value=True)

    @contextmanager
    def _scope(*, turn_id: str, input_event_key: str):
        yield SimpleNamespace(turn_id=turn_id, input_event_key=input_event_key)

    api._extract_transcript = lambda _event: "yes"
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._log_user_transcript = lambda *_args, **_kwargs: None
    api._resolve_input_event_key = lambda _event: "evt-1"
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._utterance_context_scope = _scope
    api._lifecycle_controller = lambda: SimpleNamespace(on_transcript_final=lambda _key: None)
    api._canonical_utterance_key = lambda **_kwargs: "turn-1:evt-1"
    api._rebind_active_response_correlation_key = lambda **_kwargs: None
    api._clear_stale_pending_server_auto_for_turn = lambda **_kwargs: None
    api._is_memory_intent = lambda _text: False
    api._current_run_id = lambda: "run-1"
    api._start_transcript_response_watchdog = lambda **_kwargs: None
    api._has_active_confirmation_token = lambda: True
    api._is_awaiting_confirmation_phase = lambda: False
    api._mark_confirmation_activity = lambda **_kwargs: None
    api._active_utterance = None
    api._log_utterance_trust_snapshot = lambda **_kwargs: {
        "run_id": "run-1",
        "turn_id": "turn-1",
        "input_event_key": "evt-1",
        "transcript_text": "yes",
    }
    api._maybe_verify_on_risk_clarify = verify_gate
    api._maybe_schedule_micro_ack = lambda **_kwargs: None
    api._micro_ack_category_for_reason = lambda _reason: "latency_mask"
    api._record_user_input = lambda *_args, **_kwargs: None
    api._maybe_handle_approval_response = AsyncMock(return_value=True)
    api._active_response_origin = ""

    asyncio.run(
        api._handle_input_audio_transcription_completed_event(
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "yes"},
            websocket=None,
        )
    )

    verify_gate.assert_not_awaited()
    api._maybe_handle_approval_response.assert_awaited_once_with("yes", None)
