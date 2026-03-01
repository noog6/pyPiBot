"""Focused tests for deterministic realtime response lifecycle retry policy."""

from __future__ import annotations

import asyncio
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

import logging
from unittest.mock import ANY, AsyncMock

from ai.realtime.response_lifecycle import (
    EmptyResponseDecisionAction,
    ResponseLifecycleTracker,
    decide_empty_response_done_action,
)
from ai.realtime_api import RealtimeAPI


class _StubCanonicalState:
    audio_started = False


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._empty_response_retry_canonical_keys = set()
    api._empty_response_retry_counts = {}
    api._empty_response_retry_max_attempts = 2
    api._empty_response_retry_fallback_emitted = set()
    api._response_created_canonical_keys = set()
    api._canonical_response_state_by_key = {}
    api._response_delivery_ledger = {}
    api._current_run_id = lambda: "run-test"
    api._canonical_first_audio_started = lambda canonical_key: False
    api._canonical_response_state = lambda canonical_key: _StubCanonicalState()
    api._log_lifecycle_event = lambda **kwargs: None
    api._debug_dump_canonical_key_timeline = lambda **kwargs: None
    api._lifecycle_controller = lambda: type("_Lifecycle", (), {"on_response_done": lambda self, *_: None})()
    return api


def test_created_done_empty_response_schedules_retry() -> None:
    api = _make_api()
    tracker = ResponseLifecycleTracker(api)
    sent_events: list[dict] = []

    async def _capture_send_response_create(_websocket, event, **_kwargs):
        sent_events.append(event)
        return True

    api._send_response_create = _capture_send_response_create

    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_1")
    tracker.mark_created(canonical_key=canonical_key)
    tracker.mark_done(canonical_key=canonical_key)

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
    assert sent_events[0]["response"]["metadata"]["retry_reason"] == "empty_response_done"


def test_created_done_with_assistant_content_does_not_retry() -> None:
    api = _make_api()
    api.assistant_reply = "has content"
    sent_events: list[dict] = []

    async def _capture_send_response_create(_websocket, event, **_kwargs):
        sent_events.append(event)
        return True

    api._send_response_create = _capture_send_response_create
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_2")

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_2",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    assert sent_events == []


def test_terminal_cancelled_state_skips_retry_with_reason() -> None:
    decision = decide_empty_response_done_action(
        origin="prompt",
        delivery_state_before_done="cancelled",
        assistant_text_present=False,
        audio_started=False,
        attempt_count=0,
        max_attempts=2,
        websocket_available=True,
    )

    assert decision.action == EmptyResponseDecisionAction.NOOP
    assert decision.reason_code == "delivery_state_terminal"


def test_empty_response_retry_exhausted_emits_fallback_without_scheduling_retry(caplog) -> None:
    api = _make_api()
    caplog.set_level(logging.INFO, logger="ai.realtime.response_lifecycle")

    origin_input_event_key = "input_evt_3"
    retry_input_event_key = f"{origin_input_event_key}__empty_retry"
    origin_canonical_key = api._canonical_utterance_key(
        turn_id="turn_1",
        input_event_key=origin_input_event_key,
    )
    retry_canonical_key = api._canonical_utterance_key(
        turn_id="turn_1",
        input_event_key=retry_input_event_key,
    )

    max_attempts = api._empty_response_retry_max_attempts
    api._empty_response_retry_counts[origin_canonical_key] = max_attempts

    api._emit_empty_response_retry_exhausted_fallback = AsyncMock()
    api._send_response_create = AsyncMock()

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=retry_canonical_key,
            input_event_key=retry_input_event_key,
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    api._emit_empty_response_retry_exhausted_fallback.assert_awaited_once_with(
        websocket=ANY,
        turn_id="turn_1",
        input_event_key=origin_input_event_key,
        canonical_key=origin_canonical_key,
        origin="prompt",
    )
    api._send_response_create.assert_not_awaited()


def test_maybe_schedule_empty_response_retry_delegates_to_tracker() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    tracker = ResponseLifecycleTracker(api)
    tracker.maybe_schedule_empty_response_retry = AsyncMock()
    api._response_lifecycle = tracker

    websocket = object()
    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=websocket,
            turn_id="turn_1",
            canonical_key="turn_1::input_evt_4",
            input_event_key="input_evt_4",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    tracker.maybe_schedule_empty_response_retry.assert_awaited_once_with(
        websocket=websocket,
        turn_id="turn_1",
        canonical_key="turn_1::input_evt_4",
        input_event_key="input_evt_4",
        origin="prompt",
        delivery_state_before_done="done",
    )
