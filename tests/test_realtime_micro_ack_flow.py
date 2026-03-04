from __future__ import annotations

import asyncio
import base64
from collections import deque
import time

from ai.orchestration import OrchestrationPhase
from ai.realtime_api import PendingResponseCreate, RealtimeAPI
from interaction import InteractionState


class _Manager:
    def __init__(self) -> None:
        self.scheduled: list[tuple[str, str, str, int | None]] = []
        self.cancelled: list[tuple[str, str]] = []
        self.speech_started = 0
        self.speech_ended = 0
        self.suppression_reason = None

    def maybe_schedule(self, *, context, reason: str, loop, expected_delay_ms=None) -> None:
        if callable(self.suppression_reason) and self.suppression_reason():
            return
        self.scheduled.append((context.turn_id, context.category, reason, expected_delay_ms))

    def cancel(self, *, turn_id: str, reason: str) -> None:
        self.cancelled.append((turn_id, reason))
        self.scheduled = [item for item in self.scheduled if item[0] != turn_id]

    def cancel_matching(self, *, turn_id: str, reason: str, matcher) -> None:
        self.cancelled.append((turn_id, reason))
        kept = []
        for item in self.scheduled:
            category = item[1]
            context = type("Ctx", (), {"category": category, "turn_id": item[0], "channel": "voice"})()
            if item[0] == turn_id and matcher(context):
                continue
            kept.append(item)
        self.scheduled = kept

    def cancel_all(self, *, reason: str) -> None:
        self.cancelled.append(("*", reason))

    def on_user_speech_started(self) -> None:
        self.speech_started += 1

    def on_user_speech_ended(self) -> None:
        self.speech_ended += 1

    def mark_talk_over_incident(self) -> None:
        return None

    def suppression_baseline_reason(self):
        return None


class _Ws:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


class _Mic:
    is_receiving = False

    def start_receiving(self) -> None:
        self.is_receiving = True


def _api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.loop = asyncio.new_event_loop()
    api.websocket = _Ws()
    api._micro_ack_manager = _Manager()
    api._pending_micro_ack_by_turn_channel = {}
    api._confirmation_speech_active = False
    api._audio_playback_busy = False
    api._active_utterance = None
    api._utterance_counter = 0
    api._assistant_reply_accum = ""
    api.assistant_reply = ""
    api._audio_accum = bytearray()
    api._audio_accum_bytes_target = 9600
    api._mic_receive_on_first_audio = False
    api._speaking_started = False
    api._active_response_confirmation_guarded = False
    api._active_response_preference_guarded = False
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._is_active_response_guarded = lambda: False
    api._has_active_confirmation_token = lambda: False
    api._is_awaiting_confirmation_phase = lambda: False
    api._mark_confirmation_activity = lambda **_kwargs: None
    api.handle_speech_stopped = lambda _ws: asyncio.sleep(0)
    api._response_in_flight = False
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._active_server_auto_input_event_key = None
    api._active_response_origin = "unknown"
    api._pending_server_auto_input_event_keys = deque()
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._clear_pending_response_contenders = lambda **_kwargs: None
    api._extract_response_create_metadata = lambda *_args, **_kwargs: {}
    api._extract_response_create_trigger = lambda *_args, **_kwargs: ""
    api._can_release_queued_response_create = lambda *_args, **_kwargs: True
    api._response_create_queue = deque()
    api._pending_response_create = None
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._response_create_turn_counter = 0
    api._drain_response_create_queue = lambda: asyncio.sleep(0)
    api._flush_pending_image_stimulus = lambda *_args, **_kwargs: asyncio.sleep(0)
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False
    api._current_run_id = lambda: "run-test"
    api._consume_response_origin = lambda: "server_auto"
    api._mark_transcript_response_outcome = lambda **_kwargs: None
    api._is_user_approved_interrupt_response = lambda *_args, **_kwargs: False
    api.mic = _Mic()
    api.state_manager = type(
        "State",
        (),
        {"state": InteractionState.IDLE, "update_state": lambda *_args, **_kwargs: None},
    )()
    api.orchestration_state = type(
        "State",
        (),
        {"phase": OrchestrationPhase.IDLE, "transition": lambda *_args, **_kwargs: None},
    )()
    api._maybe_handle_confirmation_decision_timeout = lambda *_args, **_kwargs: asyncio.sleep(0)
    api._last_response_create_ts = None
    api._response_create_debug_trace = False
    api._debug_vad = False

    api._micro_ack_channel_mode = "text_and_audio"
    api._micro_ack_runtime_mode = "normal"
    api._micro_ack_channel_policy = {
        "voice": {"enabled": True, "cooldown_ms": 10000, "speak": True},
        "text": {"enabled": True, "cooldown_ms": 1000, "speak": False},
    }
    return api


def test_speech_stopped_schedules_micro_ack() -> None:
    api = _api_stub()
    asyncio.run(api.handle_event({"type": "input_audio_buffer.speech_stopped"}, api.websocket))
    assert ("turn-1", "start_of_work", "speech_stopped", 700) in api._micro_ack_manager.scheduled
    api.loop.close()


def test_response_audio_delta_cancels_pending_micro_ack() -> None:
    api = _api_stub()
    event = {"type": "response.output_audio.delta", "delta": base64.b64encode(b"abc").decode("ascii")}
    asyncio.run(api.handle_event(event, api.websocket))
    assert ("turn-1", "response_started") in api._micro_ack_manager.cancelled


def test_append_assistant_reply_text_inserts_separator_between_ack_and_answer() -> None:
    api = _api_stub()

    api._append_assistant_reply_text("Let me think.")
    api._append_assistant_reply_text("Here are your saved preferences.")

    assert api.assistant_reply == "Let me think. Here are your saved preferences."
    assert api._assistant_reply_accum == "Let me think. Here are your saved preferences."
    api.loop.close()


def test_append_assistant_reply_text_can_preserve_streaming_subword_chunks() -> None:
    api = _api_stub()

    api._append_assistant_reply_text("Hel", allow_separator=False)
    api._append_assistant_reply_text("lo", allow_separator=False)

    assert api.assistant_reply == "Hello"
    assert api._assistant_reply_accum == "Hello"
    api.loop.close()


def test_send_response_create_does_not_schedule_micro_ack_while_deferred() -> None:
    api = _api_stub()
    api._audio_playback_busy = True
    api._resolve_response_create_turn_id = lambda **_kwargs: "turn-1"
    api._sync_pending_response_create_queue = lambda: None
    api._extract_confirmation_reminder_dedupe_key = lambda *_args, **_kwargs: None
    api._response_schedule_logged_turn_ids = set()
    api._turn_diagnostic_timestamps = {}
    api._response_done_serial = 0

    result = asyncio.run(
        api._send_response_create(
            api.websocket,
            {"type": "response.create", "response": {"metadata": {"origin": "assistant_message"}}},
            origin="assistant_message",
        )
    )

    assert result is False
    assert api._micro_ack_manager.scheduled == []
    api.loop.close()


def test_maybe_schedule_micro_ack_suppressed_during_pending_confirmation() -> None:
    api = _api_stub()
    api._micro_ack_manager.suppression_reason = api._micro_ack_suppression_reason
    api._has_active_confirmation_token = lambda: True
    api._is_awaiting_confirmation_phase = lambda: False

    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category="speech",
        channel="voice",
        reason="speech_stopped",
        expected_delay_ms=900,
    )

    assert api._micro_ack_manager.scheduled == []
    api.loop.close()


def test_micro_ack_suppression_reason_blocks_non_safety_reason_while_confirmation_pending() -> None:
    api = _api_stub()
    api._has_active_confirmation_token = lambda: True
    api._pending_micro_ack_reason = "speech_stopped"

    assert api._micro_ack_suppression_reason() == "confirmation_pending"
    api.loop.close()


def test_micro_ack_suppression_reason_allows_allowlisted_safety_gate_reason() -> None:
    api = _api_stub()
    api._has_active_confirmation_token = lambda: True
    api._pending_micro_ack_reason = "watchdog_confirmation_pending"

    assert api._micro_ack_suppression_reason() is None
    api.loop.close()


def test_allowlisted_confirmation_safety_gate_does_not_enable_duplicate_generic_micro_ack() -> None:
    api = _api_stub()
    api._micro_ack_manager.suppression_reason = api._micro_ack_suppression_reason
    api._has_active_confirmation_token = lambda: True

    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category="safety_gate",
        channel="voice",
        reason="watchdog_confirmation_pending",
        expected_delay_ms=900,
    )
    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category="start_of_work",
        channel="voice",
        reason="speech_stopped",
        expected_delay_ms=900,
    )

    assert api._micro_ack_manager.scheduled == [
        ("turn-1", "safety_gate", "watchdog_confirmation_pending", 900)
    ]
    api.loop.close()


def test_talk_over_aborts_active_response_and_clears_pending() -> None:
    api = _api_stub()
    api._response_in_flight = True
    api._audio_playback_busy = True
    api.state_manager.state = InteractionState.SPEAKING
    cleared: list[dict[str, str | None]] = []

    def _record_clear(*, turn_id: str, input_event_key: str | None, reason: str) -> None:
        cleared.append({"turn_id": turn_id, "input_event_key": input_event_key, "reason": reason})

    api._clear_pending_response_contenders = _record_clear
    asyncio.run(api.handle_event({"type": "input_audio_buffer.speech_started"}, api.websocket))

    assert cleared == [{"turn_id": "turn-1", "input_event_key": "", "reason": "talk_over_abort"}]
    assert '{"type": "response.cancel"}' in api.websocket.sent
    api.loop.close()


def test_emit_micro_ack_respects_quiet_mode() -> None:
    api = _api_stub()
    calls: list[bool] = []

    async def _capture_send(_message, _websocket, *, speak=True, response_metadata=None, utterance_context=None):
        calls.append(speak)

    api.send_assistant_message = _capture_send
    api._micro_ack_runtime_mode = "quiet"
    api._emit_micro_ack(type("Ctx", (), {"turn_id": "turn-1", "channel": "voice", "category": "speech", "intent": None, "action": None, "tool_call_id": None})(), "p1", "One sec")
    api.loop.run_until_complete(asyncio.sleep(0))

    assert calls == [False]
    api.loop.close()


def test_emit_micro_ack_respects_text_only_channel_mode() -> None:
    api = _api_stub()
    calls: list[bool] = []

    async def _capture_send(_message, _websocket, *, speak=True, response_metadata=None, utterance_context=None):
        calls.append(speak)

    api.send_assistant_message = _capture_send
    api._micro_ack_channel_mode = "text_only"
    api._emit_micro_ack(type("Ctx", (), {"turn_id": "turn-1", "channel": "voice", "category": "speech", "intent": None, "action": None, "tool_call_id": None})(), "p1", "One sec")
    api.loop.run_until_complete(asyncio.sleep(0))

    assert calls == [False]
    api.loop.close()


def test_micro_ack_reason_maps_to_expected_category() -> None:
    api = _api_stub()

    assert api._micro_ack_category_for_reason("speech_stopped") == "start_of_work"
    assert api._micro_ack_category_for_reason("transcript_finalized") == "latency_mask"
    assert api._micro_ack_category_for_reason("watchdog_confirmation_pending") == "safety_gate"
    assert api._micro_ack_category_for_reason("watchdog_timeout") == "failure_fallback"

    api.loop.close()


def test_log_micro_ack_event_includes_dedupe_fingerprint_and_suppression_source(monkeypatch) -> None:
    api = _api_stub()
    info_messages: list[str] = []
    debug_messages: list[str] = []

    def _capture_info(message, *args, **kwargs):
        _ = kwargs
        info_messages.append(message % args)
        return None

    def _capture_debug(message, *args, **kwargs):
        _ = kwargs
        debug_messages.append(message % args)
        return None

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture_info)
    monkeypatch.setattr("ai.realtime_api.logger.debug", _capture_debug)

    api._log_micro_ack_event(
        "scheduled",
        "turn-1",
        "speech_stopped",
        700,
        "start_of_work",
        "voice",
        None,
        None,
        None,
        "a1b2c3d4",
        None,
    )
    api._log_micro_ack_event(
        "emitted",
        "turn-1",
        "start_of_work_on_it",
        None,
        "start_of_work",
        "voice",
        None,
        None,
        None,
        "a1b2c3d4",
        None,
    )
    api._log_micro_ack_event(
        "suppressed",
        "turn-1",
        "confirmation_pending",
        None,
        "start_of_work",
        "voice",
        None,
        None,
        None,
        "a1b2c3d4",
        "confirmation",
    )

    assert any("micro_ack_scheduled" in msg and "dedupe_fp=a1b2c3d4" in msg for msg in info_messages)
    assert any("micro_ack_emitted" in msg and "dedupe_fp=a1b2c3d4" in msg for msg in info_messages)
    assert any(
        "micro_ack_suppressed" in msg
        and "dedupe_fp=a1b2c3d4" in msg
        and "suppression_source=confirmation" in msg
        for msg in debug_messages
    )
    api.loop.close()


def test_response_audio_delta_allow_logs_are_debug(monkeypatch) -> None:
    api = _api_stub()
    api._active_response_canonical_key = "run-1:turn-1:item-1"
    api._active_response_origin = "assistant_message"
    api._active_response_id = "resp-1"
    api._active_response_input_event_key = "item-1"
    event = {"type": "response.output_audio.delta", "delta": base64.b64encode(b"abc").decode("ascii")}

    lifecycle_logs: list[tuple[int, str]] = []

    def _capture_log(level, message, *args, **kwargs):
        _ = kwargs
        if "lifecycle_event" not in message:
            return None
        lifecycle_logs.append((level, str(args[-1])))
        return None

    monkeypatch.setattr("ai.realtime_api.logger.log", _capture_log)

    asyncio.run(api.handle_event(event, api.websocket))
    asyncio.run(api.handle_event(event, api.websocket))
    asyncio.run(api.handle_event(event, api.websocket))

    transition_logs = [entry for entry in lifecycle_logs if "transitioned=audio_started" in entry[1]]
    steady_state_logs = [entry for entry in lifecycle_logs if "state=audio_started" in entry[1]]

    assert len(transition_logs) == 1
    assert transition_logs[0][0] == 10
    assert len(steady_state_logs) == 2
    assert all(entry[0] == 10 for entry in steady_state_logs)
    api.loop.close()


def test_same_turn_speech_stopped_then_transcript_finalized_schedules_once() -> None:
    api = _api_stub()

    api._micro_ack_manager.suppression_reason = api._micro_ack_suppression_reason
    api._active_input_event_key_by_turn_id = {"turn-1": "ie-1"}

    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category=api._micro_ack_category_for_reason("speech_stopped"),
        channel="voice",
        reason="speech_stopped",
        expected_delay_ms=700,
    )
    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category=api._micro_ack_category_for_reason("transcript_finalized"),
        channel="voice",
        reason="transcript_finalized",
        expected_delay_ms=700,
    )

    assert api._micro_ack_manager.scheduled == [
        ("turn-1", "latency_mask", "transcript_finalized", 700)
    ]
    api.loop.close()


def test_near_ready_suppresses_latency_mask_and_keeps_safety_gate(monkeypatch) -> None:
    api = _api_stub()
    api._micro_ack_near_ready_suppress_ms = 0
    api._micro_ack_manager.suppression_reason = api._micro_ack_suppression_reason

    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category="safety_gate",
        channel="voice",
        reason="watchdog_confirmation_pending",
        expected_delay_ms=700,
    )
    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category="latency_mask",
        channel="voice",
        reason="transcript_finalized",
        expected_delay_ms=700,
    )

    info_messages: list[str] = []

    def _capture_info(message, *args, **kwargs):
        _ = kwargs
        info_messages.append(message % args)
        return None

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture_info)

    api._micro_ack_near_ready_suppress_ms = 200
    api._pending_response_create = PendingResponseCreate(
        websocket=api.websocket,
        event={"type": "response.create"},
        origin="assistant_message",
        turn_id="turn-1",
        created_at=time.monotonic() - 0.25,
        reason="test",
    )

    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category="latency_mask",
        channel="voice",
        reason="transcript_finalized",
        expected_delay_ms=700,
    )

    assert api._micro_ack_manager.scheduled == [
        ("turn-1", "safety_gate", "watchdog_confirmation_pending", 700)
    ]
    assert ("turn-1", "near_ready") in api._micro_ack_manager.cancelled
    assert any("micro_ack_suppressed_near_ready" in msg and "turn_id=turn-1" in msg for msg in info_messages)
    api.loop.close()


def test_near_ready_does_not_suppress_safety_gate() -> None:
    api = _api_stub()
    api._micro_ack_near_ready_suppress_ms = 200
    api._pending_response_create = PendingResponseCreate(
        websocket=api.websocket,
        event={"type": "response.create"},
        origin="assistant_message",
        turn_id="turn-1",
        created_at=time.monotonic() - 0.5,
        reason="test",
    )

    api._maybe_schedule_micro_ack(
        turn_id="turn-1",
        category="safety_gate",
        channel="voice",
        reason="watchdog_confirmation_pending",
        expected_delay_ms=700,
    )

    assert api._micro_ack_manager.scheduled == [
        ("turn-1", "safety_gate", "watchdog_confirmation_pending", 700)
    ]
    assert api._micro_ack_manager.cancelled == []
    api.loop.close()
