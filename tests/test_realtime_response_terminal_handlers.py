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
    api._reconcile_terminal_substantive_response = lambda **_kwargs: None
    api._response_in_flight = True
    api._response_obligations = {}
    api._response_create_queue = []
    api._preference_recall_suppressed_turns = set()
    api._preference_recall_suppressed_input_event_keys = set()
    api._pending_image_stimulus = None
    api._pending_image_flush_after_playback = False
    api._active_response_metadata = {}

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
    api._reflection_enqueued = False
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
        "deliverable_selected response_id=%s selected=%s reason=%s",
        "resp_1",
        "true",
        "normal",
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


def test_handle_response_done_reconciles_terminal_substantive_count() -> None:
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
    reconcile = Mock()
    api._reconcile_terminal_substantive_response = reconcile

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    reconcile.assert_called_once_with(
        turn_id="turn_1",
        canonical_key="turn_1::input_evt_1",
        response_id="resp_1",
        selected=True,
        selection_reason="normal",
    )



def test_handle_response_done_releases_blocked_followups_after_terminal_selection() -> None:
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
    call_order: list[str] = []

    def _decision(**_kwargs):
        call_order.append("decision")
        return True, "normal"

    def _apply(**_kwargs):
        call_order.append("apply")

    def _release(**_kwargs):
        call_order.append("release")

    api._response_done_deliverable_decision = Mock(side_effect=_decision)
    api._apply_terminal_deliverable_selection = Mock(side_effect=_apply)
    api._release_blocked_tool_followups_for_response_done = Mock(side_effect=_release)

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    assert call_order.index("decision") < call_order.index("apply") < call_order.index("release")
    api._release_blocked_tool_followups_for_response_done.assert_called_once_with(response_id="resp_1")

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
        "deliverable_selected response_id=%s selected=%s reason=%s",
        "resp_1",
        "false",
        "micro_ack_non_deliverable",
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
        "deliverable_selected response_id=%s selected=%s reason=%s",
        "resp_prov_1",
        "false",
        "provisional_empty_non_deliverable",
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


def test_handle_response_done_defers_terminal_close_when_exact_phrase_still_open() -> None:
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
    api._response_done_deliverable_decision = Mock(return_value=(False, "exact_phrase_obligation_open"))
    api._schedule_turn_contract_exact_phrase_repair_response = AsyncMock(return_value=True)
    api._log_turn_conversation_efficiency = Mock()

    with patch("ai.realtime.response_terminal_handlers.logger.info") as info_log:
        asyncio.run(api.handle_response_done({"type": "response.done"}))

    api._schedule_turn_contract_exact_phrase_repair_response.assert_awaited_once_with(
        turn_id="turn_1",
        input_event_key="input_evt_1",
        websocket=api.websocket,
    )
    api._log_turn_conversation_efficiency.assert_not_called()
    assert api.orchestration_state.transitions == []
    info_log.assert_any_call(
        "turn_terminal_close_eval run_id=%s turn_id=%s response_id=%s origin=%s transcript_final_seen=%s obligation_open=%s action=%s reason=%s",
        "run-test",
        "turn_1",
        "resp_1",
        "assistant_message",
        "false",
        "false",
        "defer",
        "exact_phrase_obligation_open",
    )




def test_handle_response_done_defers_terminal_close_for_provisional_server_auto_pending_transcript_final() -> None:
    api = _make_api()
    api._active_response_origin = "server_auto"
    api._active_response_input_event_key = "synthetic_server_auto_2"
    api._active_response_id = "resp_server_auto_2"
    api._is_provisional_response = lambda **_kwargs: True
    api._active_input_event_key_for_turn = lambda _turn_id: "synthetic_server_auto_2"
    api._response_done_deliverable_decision = Mock(return_value=(True, "normal"))
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

    api._drain_response_create_queue.assert_not_awaited()
    assert api.orchestration_state.transitions == []
    info_log.assert_any_call(
        "turn_terminal_close_eval run_id=%s turn_id=%s response_id=%s origin=%s transcript_final_seen=%s obligation_open=%s action=%s reason=%s",
        "run-test",
        "turn_1",
        "resp_server_auto_2",
        "server_auto",
        "false",
        "false",
        "defer",
        "provisional_server_auto_awaiting_transcript_final",
    )


def test_handle_response_done_allows_close_for_server_auto_with_transcript_final_linked() -> None:
    api = _make_api()
    api._active_response_origin = "server_auto"
    api._active_response_input_event_key = "item_linked_2"
    api._active_response_id = "resp_server_auto_2"
    api._is_provisional_response = lambda **_kwargs: True
    api._active_input_event_key_for_turn = lambda _turn_id: "item_linked_2"
    api._response_done_deliverable_decision = Mock(return_value=(True, "normal"))
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

    api._build_confirmation_transition_decision.assert_called_once()
    assert api.orchestration_state.transitions == [
        (OrchestrationPhase.REFLECT, "response done"),
        (OrchestrationPhase.IDLE, "response done reflection"),
    ]
def test_handle_response_done_allows_close_when_exact_phrase_repair_not_scheduled() -> None:
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
    api._response_done_deliverable_decision = Mock(return_value=(False, "exact_phrase_obligation_open"))
    api._schedule_turn_contract_exact_phrase_repair_response = AsyncMock(return_value=False)
    api._log_turn_conversation_efficiency = Mock()

    with patch("ai.realtime.response_terminal_handlers.logger.warning") as warn_log:
        asyncio.run(api.handle_response_done({"type": "response.done"}))

    api._schedule_turn_contract_exact_phrase_repair_response.assert_awaited_once_with(
        turn_id="turn_1",
        input_event_key="input_evt_1",
        websocket=api.websocket,
    )
    api._log_turn_conversation_efficiency.assert_called_once()
    assert api.orchestration_state.transitions == [
        (OrchestrationPhase.REFLECT, "response done"),
        (OrchestrationPhase.IDLE, "response done reflection"),
    ]
    warn_log.assert_any_call(
        "exact_phrase_close_guard_fallback run_id=%s turn_id=%s response_id=%s action=allow reason=repair_not_scheduled",
        "run-test",
        "turn_1",
        "resp_1",
    )


def test_handle_response_done_skips_repair_reschedule_when_already_latched() -> None:
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
    api._response_done_deliverable_decision = Mock(return_value=(False, "exact_phrase_obligation_open"))
    api._turn_contracts_by_turn_id = {
        "turn_1": {
            "exact_phrase": "Sentinel Theo online.",
            "exact_phrase_repair_scheduled": True,
        }
    }
    api._schedule_turn_contract_exact_phrase_repair_response = AsyncMock(return_value=False)
    api._log_turn_conversation_efficiency = Mock()

    asyncio.run(api.handle_response_done({"type": "response.done"}))

    api._schedule_turn_contract_exact_phrase_repair_response.assert_not_awaited()
    api._log_turn_conversation_efficiency.assert_not_called()
    assert api.orchestration_state.transitions == []


def test_terminal_substantive_reconcile_skips_when_create_time_already_counted() -> None:
    api = _make_api()
    api._reconcile_terminal_substantive_response = RealtimeAPI._reconcile_terminal_substantive_response.__get__(api, RealtimeAPI)
    api._record_substantive_response(turn_id="turn_1", canonical_key="turn_1::input_evt_1")

    api._reconcile_terminal_substantive_response(
        turn_id="turn_1",
        canonical_key="turn_1::input_evt_1",
        response_id="resp_1",
        selected=True,
        selection_reason="normal",
    )

    state = api._conversation_efficiency_state(turn_id="turn_1")
    assert state.substantive_count == 1


def test_terminal_substantive_reconcile_repairs_missed_create_time_once() -> None:
    api = _make_api()
    api._reconcile_terminal_substantive_response = RealtimeAPI._reconcile_terminal_substantive_response.__get__(api, RealtimeAPI)

    api._reconcile_terminal_substantive_response(
        turn_id="turn_1",
        canonical_key="turn_1::input_evt_1",
        response_id="resp_1",
        selected=True,
        selection_reason="normal",
    )
    api._reconcile_terminal_substantive_response(
        turn_id="turn_1",
        canonical_key="turn_1::input_evt_1",
        response_id="resp_2",
        selected=True,
        selection_reason="normal",
    )

    state = api._conversation_efficiency_state(turn_id="turn_1")
    assert state.substantive_count == 1


def test_terminal_substantive_reconcile_excludes_non_deliverable_reasons() -> None:
    api = _make_api()
    api._reconcile_terminal_substantive_response = RealtimeAPI._reconcile_terminal_substantive_response.__get__(api, RealtimeAPI)

    for reason in (
        "micro_ack_non_deliverable",
        "cancelled",
        "provisional_empty_non_deliverable",
        "exact_phrase_obligation_open",
        "tool_followup_precedence",
    ):
        api._reconcile_terminal_substantive_response(
            turn_id="turn_1",
            canonical_key="turn_1::input_evt_1",
            response_id=f"resp_{reason}",
            selected=False,
            selection_reason=reason,
        )

    state = api._conversation_efficiency_state(turn_id="turn_1")
    assert state.substantive_count == 0


def test_terminal_substantive_reconcile_exact_phrase_repair_path_totals_one() -> None:
    api = _make_api()
    api._reconcile_terminal_substantive_response = RealtimeAPI._reconcile_terminal_substantive_response.__get__(api, RealtimeAPI)

    api._reconcile_terminal_substantive_response(
        turn_id="turn_1",
        canonical_key="turn_1::input_evt_1",
        response_id="resp_parent",
        selected=False,
        selection_reason="exact_phrase_obligation_open",
    )
    api._reconcile_terminal_substantive_response(
        turn_id="turn_1",
        canonical_key="turn_1::input_evt_1",
        response_id="resp_repair",
        selected=True,
        selection_reason="normal",
    )

    state = api._conversation_efficiency_state(turn_id="turn_1")
    assert state.substantive_count == 1


def test_handle_transcribe_response_done_logs_from_per_response_buffer_prefix() -> None:
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
    api._assistant_reply_by_response_id = {
        "resp_1": "I can see rows of games stacked at different levels."
    }
    api._assistant_reply_response_id = "resp_1"
    api._maybe_enqueue_reflection = Mock()
    api.assistant_reply = "different levels."
    api._assistant_reply_accum = "different levels."

    with patch("ai.realtime.response_terminal_handlers.log_info") as info_log:
        asyncio.run(api.handle_transcribe_response_done())

    rendered = " ".join(str(arg) for arg in info_log.call_args.args)
    assert "I can see rows of games stacked at different levels." in rendered


def test_handle_transcribe_response_done_enforces_bounded_visual_clarify_exact_message() -> None:
    api = _make_api()
    api._assistant_reply_by_response_id = {
        "resp_1": "Once the camera is available, I'll help describe what's in front."
    }
    api._assistant_reply_response_id = "resp_1"
    api._maybe_enqueue_reflection = Mock()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._active_response_metadata = {
        "trigger": "asr_verify_on_risk",
        "reason": "visual_unavailable",
        "clarify_mode": "bounded",
        "turn_id": "turn_1",
    }
    api._normalize_memory_recall_answer = lambda text: text
    api._normalize_verify_clarify_message = lambda *, message, metadata: "I can’t take a fresh look right now because the camera isn’t available."

    with patch("ai.realtime.response_terminal_handlers.log_info") as info_log:
        asyncio.run(api.handle_transcribe_response_done())

    rendered = " ".join(str(arg) for arg in info_log.call_args.args)
    assert "I can’t take a fresh look right now because the camera isn’t available." in rendered


def test_clear_assistant_reply_buffers_for_other_response_preserves_active_prefix() -> None:
    api = _make_api()
    api._assistant_reply_by_response_id = {
        "resp_stale": "stale prefix",
        "resp_1": "surviving prefix",
    }
    api._assistant_reply_response_id = "resp_1"
    api.assistant_reply = "surviving prefix"
    api._assistant_reply_accum = "surviving prefix"

    api._clear_assistant_reply_buffers(response_id="resp_stale")

    assert api._assistant_reply_text_for_response("resp_stale") == ""
    assert api._assistant_reply_text_for_response("resp_1") == "surviving prefix"
    assert api.assistant_reply == "surviving prefix"
