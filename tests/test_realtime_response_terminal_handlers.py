"""Seam tests for response terminal handler extraction."""

from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
