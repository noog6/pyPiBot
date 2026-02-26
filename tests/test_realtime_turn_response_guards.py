"""Regression coverage for turn-level response guards."""

from __future__ import annotations

from collections import deque

from ai.realtime_api import RealtimeAPI


class _FakeStateManager:
    state = None


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._response_create_turn_counter = 0
    api._current_response_turn_id = "turn_1"
    api._current_input_event_key = None
    api._synthetic_input_event_counter = 0
    api._response_created_canonical_keys = set()
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._pending_response_create = None
    api._response_create_queue = deque()
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._response_schedule_logged_turn_ids = set()
    api._turn_diagnostic_timestamps = {}
    api._preference_recall_suppressed_turns = set()
    api._current_run_id = lambda: "run-395"
    api._extract_confirmation_reminder_dedupe_key = lambda event: None
    api._sync_pending_response_create_queue = lambda: None
    api._mark_transcript_response_outcome = lambda **kwargs: None
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
