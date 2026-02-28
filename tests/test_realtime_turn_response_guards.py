"""Regression coverage for turn-level response guards."""

from __future__ import annotations

import asyncio
from collections import deque

from ai.realtime_api import RealtimeAPI
from core.logging import logger


class _FakeStateManager:
    state = None


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._response_create_turn_counter = 0
    api._current_response_turn_id = "turn_1"
    api._current_input_event_key = None
    api._synthetic_input_event_counter = 0
    api._response_created_canonical_keys = set()
    api._response_delivery_ledger = {}
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._pending_response_create = None
    api._response_create_queue = deque()
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._response_schedule_logged_turn_ids = set()
    api._turn_diagnostic_timestamps = {}
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_locked_input_event_keys = set()
    api._current_run_id = lambda: "run-395"
    api._extract_confirmation_reminder_dedupe_key = lambda event: None
    api._sync_pending_response_create_queue = lambda: None
    api._mark_transcript_response_outcome = lambda **kwargs: None
    api._can_release_queued_response_create = lambda trigger, metadata: True
    api.state_manager = _FakeStateManager()
    return api


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
    scheduled_micro_acks: list[dict[str, object]] = []

    api._audio_playback_busy = True
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None
    api._maybe_schedule_micro_ack = lambda **kwargs: scheduled_micro_acks.append(kwargs)
    api._set_response_delivery_state(turn_id="turn_1", input_event_key="input_evt_1", state="delivered")

    monkeypatch.setattr(logger, "info", lambda msg, *args, **kwargs: info_logs.append(msg % args if args else msg))

    asyncio.run(
        api._watch_transcript_response_outcome(
            turn_id="turn_1",
            input_event_key="input_evt_1",
            timeout_s=0.01,
        )
    )

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

    api._audio_playback_busy = True
    api._transcript_response_watchdog_tasks = {}
    api._transcript_response_outcome_logged_keys = set()
    api._mark_transcript_response_outcome = RealtimeAPI._mark_transcript_response_outcome.__get__(api, RealtimeAPI)
    api._log_response_site_debug = lambda **_kwargs: None
    api._maybe_schedule_micro_ack = lambda **_kwargs: None

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
    assert not any("micro_ack_emitted" in entry for entry in info_logs)
    assert len(api._response_created_canonical_keys) == initial_assistant_response_count
    assert any(
        "response_not_scheduled" in entry
        and "reason=already_handled" in entry
        and "input_event_key=input_evt_done" in entry
        for entry in info_logs
    )
