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
    deliverable_observed = False
    deliverable_class = "unknown"


class _StubFunctionCallAccumulator:
    arguments_buffer = ""

    async def handle_output_item_added(self, event: dict) -> None:
        _ = event

    def handle_function_call_arguments_delta(self, event: dict) -> None:
        self.arguments_buffer += str(event.get("delta", ""))

    async def handle_function_call_arguments_done(self, event: dict, websocket: object) -> None:
        _ = (event, websocket)

    def reset_arguments_buffer(self) -> None:
        self.arguments_buffer = ""


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._assistant_reply_text_for_response = lambda _response_id: ""
    api._terminal_response_text = lambda _response_id: ""
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
    api._function_call_accumulator = _StubFunctionCallAccumulator()
    api._debug_dump_canonical_key_timeline = lambda **kwargs: None
    api._lifecycle_controller = lambda: type("_Lifecycle", (), {"on_response_done": lambda self, *_: None})()
    api._turn_has_pending_tool_followup = lambda **_kwargs: False
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


def test_server_auto_empty_retry_skipped_while_tool_followup_pending() -> None:
    api = _make_api()
    sent_events: list[dict] = []

    async def _capture_send_response_create(_websocket, event, **_kwargs):
        sent_events.append(event)
        return True

    api._send_response_create = _capture_send_response_create
    api._turn_has_pending_tool_followup = lambda **_kwargs: True

    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="synthetic_server_auto_3")

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="synthetic_server_auto_3",
            origin="server_auto",
            delivery_state_before_done="done",
        )
    )

    assert sent_events == []
    assert api._empty_response_retry_counts == {}
    assert api._empty_response_retry_canonical_keys == set()



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


def test_created_done_with_deliverable_marker_does_not_retry() -> None:
    api = _make_api()
    sent_events: list[dict] = []

    async def _capture_send_response_create(_websocket, event, **_kwargs):
        sent_events.append(event)
        return True

    api._send_response_create = _capture_send_response_create
    canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_tool")
    api._canonical_response_state = lambda _canonical_key: type(
        "_State",
        (),
        {"audio_started": False, "deliverable_observed": True},
    )()

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_tool",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    assert sent_events == []


def test_selected_response_with_partial_text_classification_counts_as_substantive() -> None:
    api = _make_api()
    tracker = ResponseLifecycleTracker(api)
    canonical_key = "turn_1::tool:call_1"
    response_id = "resp_tool_partial"
    api._response_trace_by_id = lambda: {response_id: {"canonical_key": canonical_key}}
    api._stale_response_context = lambda _response_id: {}
    api._canonical_response_state = lambda _canonical_key: type(
        "_State",
        (),
        {"audio_started": False, "deliverable_observed": True, "deliverable_class": "final"},
    )()

    assert api._selected_response_has_substantive_evidence(response_id=response_id) is True
    assert tracker.is_empty_response_done(canonical_key=canonical_key) is False


def test_is_empty_response_done_ignores_stale_global_text_when_response_scoped_evidence_missing() -> None:
    api = _make_api()
    tracker = ResponseLifecycleTracker(api)
    canonical_key = "turn_1::tool:call_empty"
    api.assistant_reply = "stale global reply"
    api._assistant_reply_accum = "stale global buffer"
    api._canonical_response_state = lambda _canonical_key: type(
        "_State",
        (),
        {
            "audio_started": False,
            "deliverable_observed": False,
            "deliverable_class": "unknown",
            "response_id": "resp_current_empty",
        },
    )()

    assert tracker.is_empty_response_done(canonical_key=canonical_key) is True


def test_tool_only_output_item_path_marks_response_non_empty() -> None:
    api = _make_api()
    sent_events: list[dict] = []

    async def _capture_send_response_create(_websocket, event, **_kwargs):
        sent_events.append(event)
        return True

    api._send_response_create = _capture_send_response_create
    states: dict[str, object] = {}

    def _mutate(*, canonical_key: str, turn_id: str, input_event_key: str | None, mutator):
        state = states.get(canonical_key)
        if state is None:
            state = type(
                "_State",
                (),
                {
                    "created": False,
                    "audio_started": False,
                    "deliverable_observed": False,
                    "done": False,
                    "cancel_sent": False,
                    "origin": "unknown",
                    "response_id": "",
                    "obligation_present": False,
                    "input_event_key": "",
                    "turn_id": "",
                    "obligation": None,
                },
            )()
        mutator(state)
        states[canonical_key] = state
        return state

    api._canonical_response_state_mutate = _mutate
    api._canonical_response_state = lambda canonical_key: states.get(canonical_key, _StubCanonicalState())
    api._active_response_canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_tool_only")
    api._active_response_input_event_key = "input_evt_tool_only"
    api._active_response_origin = "prompt"
    api._active_response_id = "resp_1"

    asyncio.run(
        api._handle_output_item_added_event(
            {"type": "response.output_item.added", "item": {"type": "function_call"}},
            websocket=object(),
        )
    )

    canonical_key = api._active_response_canonical_key
    assert bool(getattr(states[canonical_key], "deliverable_observed", False))

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_tool_only",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    assert sent_events == []


def test_tool_only_arguments_delta_path_marks_response_non_empty() -> None:
    api = _make_api()
    sent_events: list[dict] = []

    async def _capture_send_response_create(_websocket, event, **_kwargs):
        sent_events.append(event)
        return True

    api._send_response_create = _capture_send_response_create
    states: dict[str, object] = {}

    def _mutate(*, canonical_key: str, turn_id: str, input_event_key: str | None, mutator):
        state = states.get(canonical_key)
        if state is None:
            state = type(
                "_State",
                (),
                {
                    "created": False,
                    "audio_started": False,
                    "deliverable_observed": False,
                    "done": False,
                    "cancel_sent": False,
                    "origin": "unknown",
                    "response_id": "",
                    "obligation_present": False,
                    "input_event_key": "",
                    "turn_id": "",
                    "obligation": None,
                },
            )()
        mutator(state)
        states[canonical_key] = state
        return state

    api._canonical_response_state_mutate = _mutate
    api._canonical_response_state = lambda canonical_key: states.get(canonical_key, _StubCanonicalState())
    api._active_response_canonical_key = api._canonical_utterance_key(turn_id="turn_1", input_event_key="input_evt_args")
    api._active_response_input_event_key = "input_evt_args"
    api._active_response_origin = "prompt"
    api._active_response_id = "resp_args"

    asyncio.run(
        api._handle_function_call_arguments_delta_event(
            {"type": "response.function_call_arguments.delta", "delta": '{"query":"status"}'},
            websocket=object(),
        )
    )

    canonical_key = api._active_response_canonical_key
    assert bool(getattr(states[canonical_key], "deliverable_observed", False))

    asyncio.run(
        api._maybe_schedule_empty_response_retry(
            websocket=object(),
            turn_id="turn_1",
            canonical_key=canonical_key,
            input_event_key="input_evt_args",
            origin="prompt",
            delivery_state_before_done="done",
        )
    )

    assert sent_events == []
