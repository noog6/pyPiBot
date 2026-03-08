"""Seam tests for response terminal handler extraction."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from interaction import InteractionState
from ai.orchestration import OrchestrationPhase
from ai.realtime_api import RealtimeAPI


class _OrchestrationState:
    def __init__(self) -> None:
        self.phase = OrchestrationPhase.ACT
        self.transitions: list[tuple[OrchestrationPhase, str]] = []

    def transition(self, phase: OrchestrationPhase, reason: str) -> None:
        self.phase = phase
        self.transitions.append((phase, reason))


class _StateManager:
    def __init__(self) -> None:
        self.state = InteractionState.IDLE

    def update_state(self, state: InteractionState, _reason: str) -> None:
        self.state = state


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.websocket = object()
    api.rate_limits = {}
    api.state_manager = _StateManager()
    api.orchestration_state = _OrchestrationState()
    api._current_response_turn_id = "turn_1"
    api._active_response_input_event_key = "input_evt_1"
    api._active_response_origin = "assistant_message"
    api._active_response_id = "resp_1"
    api._active_response_canonical_key = "turn_1::input_evt_1"
    api._active_response_confirmation_guarded = False
    api._active_response_preference_guarded = False
    api._active_server_auto_input_event_key = None
    api._active_response_consumes_canonical_slot = True
    api._response_done_serial = 0
    api.response_in_progress = True
    api._apply_terminal_deliverable_selection = lambda **_kwargs: None
    api._response_in_flight = True
    api._response_obligations = {}
    api._response_create_queue = []
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False

    api._current_turn_id_or_unknown = lambda: "turn_1"
    api._canonical_utterance_key = lambda turn_id, input_event_key: f"{turn_id}::{input_event_key or 'none'}"
    api._response_delivery_state = lambda **_kwargs: "done"
    api._response_obligation_key = lambda **_kwargs: "obl"
    api._cancel_micro_ack = lambda **_kwargs: None
    api._lifecycle_controller = lambda: SimpleNamespace(on_response_done=lambda *_args, **_kwargs: None)
    api._log_lifecycle_event = lambda **_kwargs: None
    api._debug_dump_canonical_key_timeline = lambda **_kwargs: None
    api._set_response_delivery_state = lambda **_kwargs: None
    api._current_run_id = lambda: "run-test"
    api._is_empty_response_done = lambda **_kwargs: False
    api._is_provisional_response = lambda **_kwargs: False
    api._mark_provisional_response_completed_empty = lambda **_kwargs: None
    api._record_silent_turn_incident = lambda **_kwargs: None
    api._emit_preference_recall_skip_trace_if_needed = lambda **_kwargs: None
    api._log_turn_conversation_efficiency = lambda **_kwargs: None
    api._confirmation_hold_components = lambda: (False, False, None, False)
    api._enqueue_response_done_reflection = Mock()
    api._mark_confirmation_activity = lambda **_kwargs: None
    api._should_send_response_done_fallback_reminder = lambda: False
    api._is_guarded_server_auto_reminder_allowed = lambda **_kwargs: False
    api._recover_confirmation_guard_microphone = lambda *_args, **_kwargs: None
    api._drain_response_create_queue = AsyncMock()
    api._flush_pending_image_stimulus = AsyncMock()
    return api


def test_handle_response_done_still_schedules_empty_retry() -> None:
    api = _make_api()
    api._maybe_schedule_empty_response_retry = AsyncMock()
    api._build_confirmation_transition_decision = Mock(
        return_value=SimpleNamespace(
            allow_response_transition=True,
            close_reason="",
            emit_reminder=False,
            recover_mic=False,
        )
    )

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    api._maybe_schedule_empty_response_retry.assert_awaited_once()


def test_handle_response_done_still_runs_confirmation_transition_logic() -> None:
    api = _make_api()
    api._maybe_schedule_empty_response_retry = AsyncMock()
    api._build_confirmation_transition_decision = Mock(
        return_value=SimpleNamespace(
            allow_response_transition=True,
            close_reason="",
            emit_reminder=False,
            recover_mic=False,
        )
    )

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    api._build_confirmation_transition_decision.assert_called_once_with(
        reason="response_done",
        include_reminder_gate=True,
        was_confirmation_guarded=False,
    )
    assert api.orchestration_state.transitions == [
        (OrchestrationPhase.REFLECT, "response done"),
        (OrchestrationPhase.IDLE, "response done reflection"),
    ]


def test_handle_response_done_logs_normal_deliverable_selection() -> None:
    api = _make_api()
    api._maybe_schedule_empty_response_retry = AsyncMock()
    api._build_confirmation_transition_decision = Mock(
        return_value=SimpleNamespace(
            allow_response_transition=True,
            close_reason="",
            emit_reminder=False,
            recover_mic=False,
        )
    )

    with patch("ai.realtime.response_terminal_handlers.logger.info") as info_log:
        asyncio.run(api.handle_response_done({"type": "response.done"}))

    info_log.assert_any_call(
        "deliverable_selected response_id=%s selected=true reason=normal",
        "resp_1",
    )




def test_handle_response_done_applies_terminal_selection_to_canonical_state() -> None:
    api = _make_api()
    api._maybe_schedule_empty_response_retry = AsyncMock()
    api._build_confirmation_transition_decision = Mock(
        return_value=SimpleNamespace(
            allow_response_transition=True,
            close_reason="",
            emit_reminder=False,
            recover_mic=False,
        )
    )
    apply_selection = Mock()
    api._apply_terminal_deliverable_selection = apply_selection

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    apply_selection.assert_called_once_with(
        canonical_key="turn_1::input_evt_1",
        response_id="resp_1",
        turn_id="turn_1",
        input_event_key="input_evt_1",
        selected=True,
        selection_reason="normal",
    )

def test_handle_response_done_marks_micro_ack_as_non_deliverable() -> None:
    api = _make_api()
    api._active_response_origin = "micro_ack"
    api._maybe_schedule_empty_response_retry = AsyncMock()
    api._build_confirmation_transition_decision = Mock(
        return_value=SimpleNamespace(
            allow_response_transition=True,
            close_reason="",
            emit_reminder=False,
            recover_mic=False,
        )
    )

    with patch("ai.realtime.response_terminal_handlers.logger.info") as info_log:
        asyncio.run(api.handle_response_done({"type": "response.done"}))

    info_log.assert_any_call(
        "deliverable_selected response_id=%s selected=false reason=micro_ack_non_deliverable",
        "resp_1",
    )


def test_handle_response_done_uses_active_canonical_key_for_lifecycle() -> None:
    api = _make_api()
    lifecycle_calls: list[str] = []
    api._active_response_input_event_key = "item_real"
    api._active_response_canonical_key = "run-472:turn_1:synthetic_server_auto_2"
    api._response_delivery_state = lambda **_kwargs: None
    api._lifecycle_controller = lambda: SimpleNamespace(on_response_done=lambda key: lifecycle_calls.append(key))
    api._maybe_schedule_empty_response_retry = AsyncMock()
    api._build_confirmation_transition_decision = Mock(
        return_value=SimpleNamespace(
            allow_response_transition=True,
            close_reason="",
            emit_reminder=False,
            recover_mic=False,
        )
    )

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    assert lifecycle_calls == ["run-472:turn_1:synthetic_server_auto_2"]


def test_handle_response_done_marks_empty_provisional_as_non_deliverable_without_silent_incident() -> None:
    api = _make_api()
    api._active_response_origin = "server_auto"
    api._active_response_id = "resp_prov_1"
    api._is_empty_response_done = lambda **_kwargs: True
    api._is_provisional_response = lambda **_kwargs: True
    api._maybe_schedule_empty_response_retry = AsyncMock()
    api._build_confirmation_transition_decision = Mock(
        return_value=SimpleNamespace(
            allow_response_transition=True,
            close_reason="",
            emit_reminder=False,
            recover_mic=False,
        )
    )
    record_silent = Mock()
    mark_completed = Mock()
    api._record_silent_turn_incident = record_silent
    api._mark_provisional_response_completed_empty = mark_completed

    with patch("ai.realtime.response_terminal_handlers.logger.info") as info_log:
        asyncio.run(api.handle_response_done({"type": "response.done"}))

    info_log.assert_any_call(
        "deliverable_selected response_id=%s selected=false reason=provisional_empty_non_deliverable",
        "resp_prov_1",
    )
    mark_completed.assert_called_once_with(response_id="resp_prov_1")
    record_silent.assert_not_called()


def test_handle_response_done_records_silent_incident_for_non_provisional_empty() -> None:
    api = _make_api()
    api._active_response_origin = "server_auto"
    api._active_response_id = "resp_nonprov_1"
    api._is_empty_response_done = lambda **_kwargs: True
    api._is_provisional_response = lambda **_kwargs: False
    api._maybe_schedule_empty_response_retry = AsyncMock()
    api._build_confirmation_transition_decision = Mock(
        return_value=SimpleNamespace(
            allow_response_transition=True,
            close_reason="",
            emit_reminder=False,
            recover_mic=False,
        )
    )
    record_silent = Mock()
    mark_completed = Mock()
    api._record_silent_turn_incident = record_silent
    api._mark_provisional_response_completed_empty = mark_completed

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    record_silent.assert_called_once()
    mark_completed.assert_not_called()
