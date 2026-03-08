"""Regression coverage for turn-level response guards."""

from __future__ import annotations

import asyncio
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from collections import deque

from ai.realtime.response_create_runtime import ResponseCreateRuntime
from ai.realtime.types import CanonicalResponseState
from ai.realtime_api import PendingResponseCreate, RealtimeAPI
from core.logging import logger
from interaction import InteractionState


class _FakeStateManager:
    state = None

    def update_state(self, state, _reason: str) -> None:
        self.state = state


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
    api._conversation_efficiency_logged_turns = set()
    api._silent_turn_incident_count = 0
    api._turn_diagnostic_timestamps = {}
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._preference_recall_locked_input_event_keys = set()
    api._active_response_confirmation_guarded = False
    api._active_response_preference_guarded = False
    api._active_server_auto_input_event_key = None
    api._active_input_event_key_by_turn_id = {}
    api._response_create_runtime = ResponseCreateRuntime(api)
    api._current_run_id = lambda: "run-395"
    api._extract_confirmation_reminder_dedupe_key = lambda event: None
    api._sync_pending_response_create_queue = lambda: None
    api._mark_transcript_response_outcome = lambda **kwargs: None
    api._can_release_queued_response_create = lambda trigger, metadata: True
    api.websocket = None

    async def _noop_async(*_args, **_kwargs):
        return None

    api._enqueue_response_done_reflection = _noop_async
    api._emit_preference_recall_skip_trace_if_needed = lambda **kwargs: None
    api.state_manager = _FakeStateManager()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._turn_contracts_by_turn_id = {}
    api._turn_contract_fallback = None
    api._last_interaction_state = InteractionState.IDLE
    api._last_gesture_time = 0.0
    api._gesture_global_cooldown_s = 0.0
    api._gesture_last_fired = {}
    api._gesture_cooldowns_s = {}
    api.rate_limits = {}
    api._last_response_metadata = {}
    return api


def test_turn_contract_parse_detects_exact_phrase_and_no_gesture() -> None:
    api = _make_api()

    contract = api._parse_turn_contract_from_text(
        'Say exactly "Ready. Run 556 online." Do not call tools or gesture.'
    )

    assert contract["exact_phrase"] == "Ready. Run 556 online."
    assert contract["no_tools"] is True
    assert contract["no_gesture"] is True


def test_turn_contract_blocks_state_gesture_cues() -> None:
    api = _make_api()
    api._last_interaction_state = InteractionState.SPEAKING
    api._update_turn_contract_from_input("Say exactly 'Ready.' Do not call tools or gesture.", source="startup_prompt")

    api._handle_state_gesture(InteractionState.IDLE)

    assert api._last_interaction_state == InteractionState.IDLE


def test_turn_contract_blocks_gesture_tool_calls() -> None:
    api = _make_api()
    api._update_turn_contract_from_input("Do not gesture.", source="input_audio_transcription")

    assert api._turn_contract_blocks_tool_call(tool_name="gesture_attention_snap") is True
    assert api._turn_contract_blocks_tool_call(tool_name="perform_research") is False


def test_turn_contract_allows_explicit_requested_gesture_even_when_no_gesture_present() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Do one attention snap, then do not gesture.",
        source="input_audio_transcription",
    )

    assert api._turn_contract_blocks_tool_call(tool_name="gesture_attention_snap") is False
    assert api._turn_contract_blocks_tool_call(tool_name="gesture_nod") is True


def test_turn_contract_exact_phrase_repair_scheduled_when_parent_missing_phrase() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Theo, do one attention snap, then say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = ""
    sent_events: list[dict[str, object]] = []

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    scheduled = asyncio.run(
        api._schedule_turn_contract_exact_phrase_repair_response(
            turn_id="turn_1",
            input_event_key="item_1",
            websocket=object(),
        )
    )

    assert scheduled is True
    assert len(sent_events) == 1
    metadata = sent_events[0]["event"]["response"]["metadata"]
    assert metadata["turn_contract_exact_phrase_repair"] == "true"
    assert metadata["input_event_key"] == "item_1:exact_phrase_repair"


def test_turn_contract_exact_phrase_repair_schedules_at_most_once() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = ""
    sent_events: list[dict[str, object]] = []

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    first = asyncio.run(
        api._schedule_turn_contract_exact_phrase_repair_response(
            turn_id="turn_1",
            input_event_key="item_1",
            websocket=object(),
        )
    )
    second = asyncio.run(
        api._schedule_turn_contract_exact_phrase_repair_response(
            turn_id="turn_1",
            input_event_key="item_1",
            websocket=object(),
        )
    )

    assert first is True
    assert second is False
    assert len(sent_events) == 1


def test_turn_contract_exact_phrase_not_open_when_already_spoken() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = "Sentinel Theo online."

    assert api._turn_contract_exact_phrase_open(turn_id="turn_1") is False
    assert api._turn_contract_exact_phrase(turn_id="turn_1") == ""


def test_response_done_decision_holds_terminal_selection_when_exact_phrase_open() -> None:
    api = _make_api()
    api._update_turn_contract_from_input(
        "Do one attention snap, then say exactly: Sentinel Theo online.",
        source="input_audio_transcription",
    )
    api._assistant_reply_accum = ""

    selected, reason = api._response_done_deliverable_decision(
        turn_id="turn_1",
        origin="server_auto",
        delivery_state_before_done="done",
        active_response_was_provisional=False,
        done_canonical_key=api._canonical_utterance_key(turn_id="turn_1", input_event_key="item_1"),
    )

    assert selected is False
    assert reason == "exact_phrase_obligation_open"


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
    debug_logs: list[str] = []
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
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg))

    asyncio.run(
        api._watch_transcript_response_outcome(
            turn_id="turn_1",
            input_event_key="input_evt_1",
            timeout_s=0.01,
        )
    )

    assert emitted_response_creates == []
    assert scheduled_micro_acks == []
    assert not any(
        "response_not_scheduled" in entry
        and "input_event_key=input_evt_1" in entry
        and "reason=already_handled" in entry
        for entry in info_logs
    )
    assert any(
        "response_not_scheduled" in entry
        and "input_event_key=input_evt_1" in entry
        and "reason=already_handled" in entry
        for entry in debug_logs
    )
    assert not any(
        "response_create_scheduled" in entry and "reason=audio_playback_busy" in entry
        for entry in info_logs
    )


def test_watchdog_audio_busy_after_done_does_not_emit_duplicate_create_or_micro_ack(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    debug_logs: list[str] = []
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
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg))

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
    assert not any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_done" in entry
        for entry in info_logs
    )
    assert any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_done" in entry
        for entry in debug_logs
    )


def test_watchdog_audio_busy_terminal_state_ignores_pending_for_other_canonical_key(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    debug_logs: list[str] = []
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
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg))

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
    assert not any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_target" in entry
        for entry in info_logs
    )
    assert any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_target" in entry
        for entry in debug_logs
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


def test_empty_retry_create_dropped_after_same_turn_final_deliverable() -> None:
    api = _make_api()
    outcomes: list[dict[str, str]] = []
    api._mark_transcript_response_outcome = lambda **kwargs: outcomes.append(kwargs)
    turn_id = "turn_1"
    final_input_event_key = "tool:call_123"
    final_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=final_input_event_key)

    api._canonical_response_state_store()[final_canonical_key] = CanonicalResponseState(
        turn_id=turn_id,
        input_event_key=final_input_event_key,
        origin="tool_output",
        deliverable_class="final",
    )

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key="synthetic_server_auto_3__empty_retry",
        origin="server_auto",
        response_metadata={
            "input_event_key": "synthetic_server_auto_3__empty_retry",
            "retry_reason": "empty_response_done",
        },
    )

    assert dropped is True
    assert outcomes
    assert outcomes[0]["reason"] == "canonical_delivery_terminal_state"


def test_empty_retry_create_not_dropped_without_same_turn_final_deliverable() -> None:
    api = _make_api()
    turn_id = "turn_1"

    dropped = api._drop_response_create_for_terminal_state(
        turn_id=turn_id,
        input_event_key="synthetic_server_auto_3__empty_retry",
        origin="server_auto",
        response_metadata={
            "input_event_key": "synthetic_server_auto_3__empty_retry",
            "retry_reason": "empty_response_done",
        },
    )

    assert dropped is False



def test_empty_retry_not_scheduled_after_same_turn_final_deliverable() -> None:
    api = _make_api()
    turn_id = "turn_1"
    final_input_event_key = "tool:call_123"
    final_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=final_input_event_key)
    sent_events: list[dict[str, object]] = []

    api._canonical_response_state_store()[final_canonical_key] = CanonicalResponseState(
        turn_id=turn_id,
        input_event_key=final_input_event_key,
        origin="tool_output",
        deliverable_class="final",
    )

    async def _capture_send_response_create(_websocket, event, **kwargs):
        sent_events.append({"event": event, **kwargs})
        return True

    api._send_response_create = _capture_send_response_create

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id=turn_id,
            canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key="synthetic_server_auto_3"),
            input_event_key="synthetic_server_auto_3",
            origin="server_auto",
            delivery_state_before_done="done",
        )
    )

    assert sent_events == []


def test_playback_complete_drain_drops_same_turn_empty_retry_after_final_deliverable() -> None:
    api = _make_api()
    ws = object()
    sent: list[dict[str, object]] = []

    async def _capture_send(*_args, **_kwargs):
        sent.append({"sent": True})
        return True

    api._send_response_create = _capture_send
    api._active_response_id = None
    api._active_response_origin = "unknown"

    turn_id = "turn_1"
    final_input_event_key = "tool:call_123"
    final_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=final_input_event_key)
    api._canonical_response_state_store()[final_canonical_key] = CanonicalResponseState(
        turn_id=turn_id,
        input_event_key=final_input_event_key,
        origin="tool_output",
        deliverable_class="final",
    )

    api._pending_response_create = PendingResponseCreate(
        websocket=ws,
        event={
            "type": "response.create",
            "response": {
                "metadata": {
                    "turn_id": turn_id,
                    "input_event_key": "synthetic_server_auto_3__empty_retry",
                    "retry_reason": "empty_response_done",
                }
            },
        },
        origin="server_auto",
        turn_id=turn_id,
        created_at=0.0,
        reason="awaiting_transcript_final",
    )

    asyncio.run(api._drain_response_create_queue(source_trigger="playback_complete"))

    assert sent == []
    assert api._pending_response_create is None


def test_mark_transcript_response_outcome_demotes_intentional_non_action_reasons(monkeypatch) -> None:
    api = _make_api()
    info_logs: list[str] = []
    debug_logs: list[str] = []

    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))
    monkeypatch.setattr(logger, "debug", lambda msg, *args, **kwargs: debug_logs.append(msg % args if args else msg))

    api._mark_transcript_response_outcome(
        input_event_key="input_evt_same_turn",
        turn_id="turn_1",
        outcome="response_not_scheduled",
        reason="same_turn_already_owned",
        details="owner=tool_followup_owned",
    )
    api._mark_transcript_response_outcome(
        input_event_key="input_evt_active",
        turn_id="turn_1",
        outcome="response_not_scheduled",
        reason="active_response_in_flight",
        details="active_origin=tool_output",
    )
    api._mark_transcript_response_outcome(
        input_event_key="input_evt_anomaly",
        turn_id="turn_1",
        outcome="response_not_scheduled",
        reason="already_handled",
        details="canonical_delivery_state=done",
    )

    assert not any("reason=same_turn_already_owned" in entry for entry in info_logs)
    assert not any("reason=active_response_in_flight" in entry for entry in info_logs)
    assert not any("reason=already_handled" in entry for entry in info_logs)
    assert any("reason=same_turn_already_owned" in entry for entry in debug_logs)
    assert any("reason=active_response_in_flight" in entry for entry in debug_logs)
    assert any("reason=already_handled" in entry for entry in debug_logs)
