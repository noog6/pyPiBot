"""Regression coverage for turn-level response guards."""

from __future__ import annotations

import asyncio
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from collections import deque

from ai.realtime_api import PendingResponseCreate, RealtimeAPI
from core.logging import logger
from interaction import InteractionState


class _FakeStateManager:
    state = None


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._response_create_turn_counter = 0
    api._current_response_turn_id = "turn_1"
    api._current_input_event_key = None
    api._synthetic_input_event_counter = 0
    api._response_created_canonical_keys = set()
    api._empty_response_retry_canonical_keys = set()
    api._response_delivery_ledger = {}
    api._empty_response_retry_counts = {}
    api._empty_response_retry_fallback_emitted = set()
    api._empty_response_retry_max_attempts = 2
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._pending_response_create = None
    api._response_create_queue = deque()
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._response_schedule_logged_turn_ids = set()
    api._conversation_efficiency_by_turn = {}
    api._silent_turn_incident_count = 0
    api._turn_diagnostic_timestamps = {}
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_locked_input_event_keys = set()
    api._current_run_id = lambda: "run-395"
    api._extract_confirmation_reminder_dedupe_key = lambda event: None
    api._sync_pending_response_create_queue = lambda: None
    api._mark_transcript_response_outcome = lambda **kwargs: None
    api._can_release_queued_response_create = lambda trigger, metadata: True
    api.state_manager = _FakeStateManager()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    return api


def test_empty_response_done_schedules_single_retry_for_prompt_origin() -> None:
    api = _make_api()
    sent_events: list[dict[str, object]] = []
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_1")

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_1",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )
    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_1",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    assert len(sent_events) == 1
    metadata = sent_events[0]["event"]["response"]["metadata"]
    assert metadata["retry_reason"] == "empty_response_done"
    assert metadata["idempotency_key"].startswith("empty_response_done:")



def test_empty_response_retry_uses_origin_canonical_key_and_exhausts_with_terminal_fallback() -> None:
    api = _make_api()
    api._empty_response_retry_max_attempts = 1
    sent_events: list[dict[str, object]] = []
    fallback_messages: list[tuple[str, bool]] = []
    terminal_markers: list[str] = []

    origin_canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_1")
    retry_canonical_key = api._canonical_utterance_key(
        turn_id="turn_1",
        input_event_key="input_evt_1__empty_retry",
    )

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    async def _capture_assistant_message(message, _websocket, *, speak=True, **kwargs):
        fallback_messages.append((message, speak))

    def _capture_log_lifecycle_event(**kwargs):
        terminal_markers.append(str(kwargs.get("decision") or ""))

    api._send_response_create = _capture_send_response_create
    api.send_assistant_message = _capture_assistant_message
    api._log_lifecycle_event = _capture_log_lifecycle_event
    api._debug_dump_canonical_key_timeline = lambda **kwargs: None

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=origin_canonical_key,
            input_event_key="input_evt_1",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )
    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=retry_canonical_key,
            input_event_key="input_evt_1__empty_retry",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    assert len(sent_events) == 1
    assert api._empty_response_retry_counts[origin_canonical_key] == 1
    assert fallback_messages == [("Sorry—I’m having trouble generating a response right now.", False)]
    assert "transition_terminal:empty_response_retry_exhausted" in terminal_markers
    assert api._response_delivery_state(turn_id="turn_1", input_event_key="input_evt_1") == "done"

def test_empty_response_done_retry_skipped_for_cancelled_terminal_state() -> None:
    api = _make_api()
    sent_events: list[dict[str, object]] = []
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_1")

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_1",
            origin="prompt",
            delivery_state_before_done="cancelled",
        )
    )

    assert sent_events == []


def test_response_create_guard_blocks_duplicate_for_same_canonical_key() -> None:
    api = _make_api()
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_prompt_1")
    api._response_created_canonical_keys.add(canonical_key)

    blocked = api._schedule_pending_response_create(
        websocket=None,
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "synthetic_prompt_1"}}},
        origin="injection",
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert blocked is False
    assert api._pending_response_create is None


def test_response_create_correlation_adds_non_unknown_input_event_key() -> None:
    api = _make_api()
    event = {"type": "response.create", "response": {"metadata": {}}}

    input_event_key = api._ensure_response_create_correlation(
        response_create_event=event,
        origin="prompt",
        turn_id="turn_1",
    )

    assert input_event_key.startswith("synthetic_prompt_")
    assert event["response"]["metadata"]["input_event_key"] == input_event_key


def test_clear_all_pending_response_creates_clears_pending_and_queue() -> None:
    api = _make_api()
    api._pending_response_create = object()
    api._response_create_queue.append({"event": {"type": "response.create"}})

    api._clear_all_pending_response_creates(reason="talk_over_abort")

    assert api._pending_response_create is None
    assert list(api._response_create_queue) == []


def test_response_create_guard_blocks_default_response_while_preference_lock_active() -> None:
    api = _make_api()
    api._preference_recall_locked_input_event_keys.add("input_evt_1")

    blocked = api._schedule_pending_response_create(
        websocket=None,
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_1"}}},
        origin="assistant_message",
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert blocked is False
    assert api._pending_response_create is None


def test_audio_playback_busy_retry_dropped_after_delivery() -> None:
    api = _make_api()
    api._audio_playback_busy = True
    api._set_response_delivery_state(
        turn_id="turn_1",
        input_event_key="input_evt_1",
        state="delivered",
    )

    scheduled = api._schedule_pending_response_create(
        websocket=None,
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_1"}}},
        origin="assistant_message",
        reason="audio_playback_busy",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert scheduled is False
    assert api._pending_response_create is None


def test_audio_playback_busy_retry_enqueues_once_before_delivery() -> None:
    api = _make_api()
    api._audio_playback_busy = True

    api._schedule_pending_response_create(
        websocket=None,
        response_create_event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_1"}}},
        origin="assistant_message",
        reason="audio_playback_busy",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    send_attempts: list[str] = []

    async def _fake_send_response_create(*args, **kwargs):
        send_attempts.append("sent")
        return True

    api._send_response_create = _fake_send_response_create
    api._audio_playback_busy = False
    asyncio.run(api._drain_response_create_queue())

    assert send_attempts == ["sent"]
    assert api._pending_response_create is None


def test_run_411_watchdog_skips_audio_busy_retry_after_delivered_answer(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    emitted_response_creates: list[dict[str, object]] = []
    scheduled_micro_acks: list[dict[str, object]] = []

    api._audio_playback_busy = True
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_1", state="delivered")

    async def _capture_send_response_create(*_args, **kwargs):
        emitted_response_creates.append(kwargs)
        return True

    api._send_response_create = _capture_send_response_create

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))

    asyncio.run(
        api._watch_transcript_response_outcome(
            turn_id="turn_1",
            input_event_key="input_evt_1",
            timeout_s=0.01,
        )
    )

    assert emitted_response_creates == []
    assert scheduled_micro_acks == []
    assert any(
        "response_not_scheduled" in entry
        and "input_event_key=input_evt_1" in entry
        and "reason=already_handled" in entry
        for entry in info_logs
    )
    assert not any(
        "response_create_scheduled" in entry and "reason=audio_playback_busy" in entry
        for entry in info_logs
    )


def test_watchdog_audio_busy_after_done_does_not_emit_duplicate_create_or_micro_ack(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    emitted_response_creates: list[dict[str, object]] = []
    scheduled_micro_acks: list[dict[str, object]] = []

    api._audio_playback_busy = True
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)

    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_done")
    api._response_created_canonical_keys.add(canonical_key)
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_done", state="done")
    initial_assistant_response_count = len(api._response_created_canonical_keys)

    async def _capture_send_response_create(*_args, **kwargs):
        emitted_response_creates.append(kwargs)
        return True

    api._send_response_create = _capture_send_response_create

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))

    asyncio.run(
        api._watch_transcript_response_outcome(
            turn_id="turn_1",
            input_event_key="input_evt_done",
            timeout_s=0.01,
        )
    )

    assert emitted_response_creates == []
    assert scheduled_micro_acks == []
    assert not any("micro_ack_emitted" in entry for entry in info_logs)
    assert len(api._response_created_canonical_keys) == initial_assistant_response_count
    assert any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_done" in entry
        for entry in info_logs
    )


def test_watchdog_audio_busy_terminal_state_ignores_pending_for_other_canonical_key(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    emitted_response_creates: list[dict[str, object]] = []
    scheduled_micro_acks: list[dict[str, object]] = []

    api._audio_playback_busy = True
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_target", state="done")
    api._pending_response_create = PendingResponseCreate(
        websocket=None,
        event={"type": "response.create", "response": {"metadata": {"input_event_key": "input_evt_other"}}},
        origin="assistant_message",
        turn_id="turn_1",
        created_at=0.0,
        reason="audio_playback_busy",
    )

    async def _capture_send_response_create(*_args, **kwargs):
        emitted_response_creates.append(kwargs)
        return True

    api._send_response_create = _capture_send_response_create

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))

    asyncio.run(
        api._watch_transcript_response_outcome(
            turn_id="turn_1",
            input_event_key="input_evt_target",
            timeout_s=0.01,
        )
    )

    assert emitted_response_creates == []
    assert scheduled_micro_acks == []
    assert api._pending_response_create is not None
    assert any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_target" in entry
        for entry in info_logs
    )


def test_drain_response_create_queue_drops_terminal_canonical_key(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    transcript_outcomes: list[dict[str, object]] = []
    send_attempts: list[str] = []

    api._mark_transcript_response_outcome = lambda **kwargs: transcript_outcomes.append(kwargs)
    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))

    turn_id = "turn_1"
    input_event_key = "input_evt_terminal"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._lifecycle_controller().on_response_done(canonical_key)

    api._response_create_queue.append(
        {
            "websocket": None,
            "event": {"type": "response.create", "response": {"metadata": {"input_event_key": input_event_key}}},
            "origin": "assistant_message",
            "turn_id": turn_id,
            "record_ai_call": False,
        }
    )

    async def _fake_send_response_create(*_args, **_kwargs):
        send_attempts.append("sent")
        return True

    api._send_response_create = _fake_send_response_create
    asyncio.run(api._drain_response_create_queue())

    assert send_attempts == []
    assert list(api._response_create_queue) == []
    assert any("response_dropped_terminal_state" in entry for entry in info_logs)
    assert transcript_outcomes == [
        {
            "input_event_key": input_event_key,
            "turn_id": turn_id,
            "outcome": "response_not_scheduled",
            "reason": "canonical_delivery_terminal_state",
            "details": "canonical delivery terminal state origin=assistant_message prior_state=done canonical_key=run-395:turn_1:input_evt_terminal origin_canonical_key=run-395:turn_1:input_evt_terminal",
        }
    ]


def test_empty_response_done_retry_terminal_guard_uses_retry_canonical_key() -> None:
    api = _make_api()
    turn_id = "turn_1"
    origin_input_event_key = "input_evt_1"
    retry_input_event_key = f"{origin_input_event_key}__empty_retry"
    metadata = {"input_event_key": retry_input_event_key, "retry_reason": "empty_response_done"}

    origin_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=origin_input_event_key)
    retry_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=retry_input_event_key)
    api._lifecycle_controller().on_response_done(origin_canonical_key)

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key=retry_input_event_key,
        origin="prompt",
        response_metadata=metadata,
    )

    assert dropped is False
    assert api._lifecycle_controller().state_for(origin_canonical_key).value == "done"
    assert api._lifecycle_controller().state_for(retry_canonical_key).value == "new"


def test_empty_response_done_retry_terminal_guard_drops_when_retry_canonical_terminal() -> None:
    api = _make_api()
    turn_id = "turn_1"
    origin_input_event_key = "input_evt_1"
    retry_input_event_key = f"{origin_input_event_key}__empty_retry"
    metadata = {"input_event_key": retry_input_event_key, "retry_reason": "empty_response_done"}
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)

    retry_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=retry_input_event_key)
    api._lifecycle_controller().on_response_done(retry_canonical_key)

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key=retry_input_event_key,
        origin="prompt",
        response_metadata=metadata,
    )

    assert dropped is True
    assert outcomes
    assert outcomes[0]["reason"] == "canonical_delivery_terminal_state"
    assert "origin_canonical_key=run-395:turn_1:input_evt_1" in outcomes[0]["details"]


def test_watchdog_micro_ack_not_scheduled_when_terminal_and_idle() -> None:
    api = _make_api()
    scheduled: list[dict[str, object]] = []

    class _Manager:
        def maybe_schedule(self, **kwargs):
            scheduled.append(kwargs)

        def cancel(self, **_kwargs):
            return None

    api._micro_ack_manager = _Manager()
    api.loop = object()
    api._pending_micro_ack_by_turn_channel = {}
    api._micro_ack_near_ready_suppress_ms = 0
    api._micro_ack_correlation_metadata = lambda: {}
    api._active_input_event_key_by_turn = {"turn_1": "input_evt_terminal"}
    api.state_manager.state = InteractionState.IDLE
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_terminal", state="done")

    api._maybe_schedule_micro_ack(
        turn_id="turn_1",
        category=api._micro_ack_category_for_reason("watchdog_audio_playback_busy"),
        channel="voice",
        reason="watchdog_audio_playback_busy",
    )

    assert scheduled == []


def test_response_text_delta_without_bound_response_does_not_enter_speaking() -> None:
    api = _make_api()
    transitions: list[tuple[InteractionState, str]] = []

    class _StateManager:
        def __init__(self) -> None:
            self.state = InteractionState.IDLE

        def update_state(self, state, reason):
            transitions.append((state, reason))
            self.state = state

    api.state_manager = _StateManager()
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._cancel_micro_ack = lambda **_kwargs: None
    api._mark_first_assistant_utterance_observed_if_needed = lambda _text: None
    api._append_assistant_reply_text = lambda *args, **kwargs: None
    api._response_status_by_id = {}
    api._active_response_id = None

    asyncio.run(
        api._handle_event_legacy(
            {"type": "response.text.delta", "response_id": "resp_terminal", "delta": "late delta"},
            websocket=None,
        )
    )

    assert transitions == []
    assert api.state_manager.state == InteractionState.IDLE


def test_empty_transcript_blocked_state_rejects_followup_response_scheduling() -> None:
    api = _make_api()
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)

    api._set_response_delivery_state(
        turn_id="turn_1",
        input_event_key="input_evt_empty",
        state="blocked_empty_transcript",
    )

    dropped = api._drop_response_create_for_terminal_state(
        turn_id="turn_1",
        input_event_key="input_evt_empty",
        origin="upgraded_response",
        response_metadata={"input_event_key": "input_evt_empty"},
    )

    assert dropped is True
    assert outcomes
    assert outcomes[0]["reason"] == "empty_transcript_blocked"


def test_empty_transcript_blocked_state_rejects_injection_response_attempts() -> None:
    api = _make_api()
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)
    api._set_response_delivery_state(
        turn_id="turn_1",
        input_event_key="input_evt_empty",
        state="blocked_empty_transcript",
    )

    for origin in ("imu_injection", "battery_injection", "system_context"):
        dropped = api._drop_response_create_for_terminal_state(
            turn_id="turn_1",
            input_event_key="input_evt_empty",
            origin=origin,
            response_metadata={
                "input_event_key": "input_evt_empty",
                "create_response": "true",
            },
        )
        assert dropped is True

    assert len(outcomes) == 3
    assert all(entry["reason"] == "empty_transcript_blocked" for entry in outcomes)


def test_tool_followup_done_clears_stale_pending_assistant_create() -> None:
    api = _make_api()
    input_event_key = "tool:call_123"
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key=input_event_key)
    api._pending_response_create = PendingResponseCreate(
        websocket=None,
        event={
            "type": "response.create",
            "response": {"metadata": {"turn_id": "turn_1", "input_event_key": input_event_key}},
        },
        origin="assistant_message",
        turn_id="turn_1",
        created_at=0.0,
        reason="audio_playback_busy",
    )

    api._set_tool_followup_state(canonical_key=canonical_key, state="done", reason="response_done")

    assert api._pending_response_create is None


def test_tool_followup_done_clears_stale_assistant_queue_residue_only() -> None:
    api = _make_api()
    input_event_key = "tool:call_123"
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key=input_event_key)
    api._response_create_queue.append(
        {
            "websocket": None,
            "event": {
                "type": "response.create",
                "response": {"metadata": {"input_event_key": input_event_key}},
            },
            "origin": "assistant_message",
            "turn_id": "turn_1",
            "record_ai_call": False,
        }
    )
    api._response_create_queue.append(
        {
            "websocket": None,
            "event": {
                "type": "response.create",
                "response": {"metadata": {"input_event_key": input_event_key}},
            },
            "origin": "tool_output",
            "turn_id": "turn_1",
            "record_ai_call": False,
        }
    )

    api._set_tool_followup_state(canonical_key=canonical_key, state="done", reason="response_done")

    assert len(api._response_create_queue) == 1
    assert api._response_create_queue[0]["origin"] == "tool_output"


def test_tool_followup_cleanup_ignores_non_tool_canonical_keys() -> None:
    api = _make_api()
    input_event_key = "input_evt_1"
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key=input_event_key)
    api._pending_response_create = PendingResponseCreate(
        websocket=None,
        event={
            "type": "response.create",
            "response": {"metadata": {"turn_id": "turn_1", "input_event_key": input_event_key}},
        },
        origin="assistant_message",
        turn_id="turn_1",
        created_at=0.0,
        reason="audio_playback_busy",
    )

    api._set_tool_followup_state(canonical_key=canonical_key, state="done", reason="response_done")

    assert api._pending_response_create is not None
