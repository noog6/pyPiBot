from __future__ import annotations

from dataclasses import replace
import os
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.decision_arbitration_adapter import build_response_create_observation
from ai.interaction_lifecycle_policy import ResponseCreateDecision, ResponseCreateDecisionAction
from ai.realtime.response_create_runtime import ResponseCreateOutcomeAction
from tests.test_response_create_runtime_split import _make_api_stub


def test_direct_send_snapshot_maps_to_allow_now() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_send", "input_event_key": "item_send"}}}
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision, lifecycle_decision = runtime._decide_response_create_action_with_lifecycle(snapshot)

    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=lifecycle_decision,
        same_turn_owner_reason=snapshot.same_turn_owner_reason,
        canonical_audio_started=False,
    )

    assert decision.action is ResponseCreateOutcomeAction.SEND
    assert observation.decision.decision_disposition == "allow_now"
    assert observation.decision.selected_candidate_id == "direct_send"


def test_same_turn_owner_maps_to_drop_not_block() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._assistant_message_same_turn_owner_reason = lambda **_kwargs: "tool_followup_owned"
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_owner", "input_event_key": "item_owner"}}}
    snapshot, decision = runtime.evaluate_response_create_attempt(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=None,
        same_turn_owner_reason=snapshot.same_turn_owner_reason,
        canonical_audio_started=None,
    )

    assert decision.action is ResponseCreateOutcomeAction.DROP
    assert observation.decision.decision_disposition == "drop"
    assert observation.decision.owner_scope == "pending_tool_followup"


def test_terminal_state_blocked_maps_to_block() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._drop_response_create_for_terminal_state = lambda **_kwargs: True
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_terminal", "input_event_key": "item_terminal"}}}
    snapshot, decision = runtime.evaluate_response_create_attempt(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=None,
        same_turn_owner_reason=snapshot.same_turn_owner_reason,
        canonical_audio_started=None,
    )

    assert decision.action is ResponseCreateOutcomeAction.BLOCK
    assert observation.decision.decision_disposition == "block"
    assert observation.decision.deliverable_status == "blocked_terminal"


def test_awaiting_transcript_final_maps_to_defer_with_transcript_state() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_wait", "input_event_key": "synthetic_server_auto_5"}}}
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="server_auto",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision, lifecycle_decision = runtime._decide_response_create_action_with_lifecycle(snapshot)

    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=lifecycle_decision,
        same_turn_owner_reason=snapshot.same_turn_owner_reason,
        canonical_audio_started=False,
    )

    assert decision.action is ResponseCreateOutcomeAction.SCHEDULE
    assert observation.decision.decision_disposition == "defer"
    assert observation.decision.transcript_final_state == "awaiting_transcript_final"


def test_normalization_preserves_native_reason_action() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._response_in_flight = True
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_busy", "input_event_key": "item_busy"}}}
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision, lifecycle_decision = runtime._decide_response_create_action_with_lifecycle(snapshot)
    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=lifecycle_decision,
        same_turn_owner_reason=snapshot.same_turn_owner_reason,
        canonical_audio_started=False,
    )

    assert observation.decision.native_outcome_action == decision.action.value
    assert observation.decision.native_reason_code == decision.reason_code
    assert observation.decision.selected_candidate_id == decision.selected_candidate_id


def test_normalization_warnings_when_optional_inputs_absent() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_warn", "input_event_key": "item_warn"}}}
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(snapshot)
    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=None,
        same_turn_owner_reason=None,
        canonical_audio_started=None,
    )

    assert "lifecycle_decision_unavailable" in observation.normalization_warnings
    assert "canonical_audio_started_unavailable" in observation.normalization_warnings


def test_adapter_is_observational_only_and_runtime_outputs_unchanged() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_obs", "input_event_key": "item_obs"}}}

    direct_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    direct_decision = runtime.decide_response_create_action(direct_snapshot)

    eval_snapshot, eval_decision = runtime.evaluate_response_create_attempt(
        response_create_event={"type": "response.create", "response": {"metadata": {"turn_id": "turn_obs", "input_event_key": "item_obs"}}},
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )

    assert eval_snapshot.canonical_key == direct_snapshot.canonical_key
    assert eval_decision == direct_decision


def test_manual_lifecycle_shape_can_be_normalized() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_manual", "input_event_key": "item_manual"}}}
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(snapshot)
    lifecycle_decision = ResponseCreateDecision(
        action=ResponseCreateDecisionAction.SEND,
        reason_code="direct_send",
        selected_candidate_id="direct_send",
    )

    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=lifecycle_decision,
        same_turn_owner_reason=None,
        canonical_audio_started=False,
    )

    assert observation.context.authority_retained_by == "ai.realtime.response_create_runtime"
    assert observation.decision.observational_only is True


def test_same_turn_owner_missing_reason_uses_conservative_fallback_warning() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_owner_warn", "input_event_key": "item_owner_warn"}}}
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    snapshot = replace(
        snapshot,
        same_turn_owner_present=True,
        same_turn_owner_reason=None,
    )
    decision = runtime._build_execution_decision(
        action=ResponseCreateOutcomeAction.DROP,
        reason_code="same_turn_already_owned",
        explanation="Assistant message suppressed by same-turn owner.",
        selected_candidate_id="same_turn_owner",
    )

    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=None,
        same_turn_owner_reason=None,
        canonical_audio_started=None,
    )

    assert observation.decision.owner_scope == "subsystem_local"
    assert "owner_scope_ambiguous_same_turn_owner" in observation.decision.normalization_warnings


def test_lineage_guard_uses_conservative_owner_scope_warning() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_lineage", "input_event_key": "item_lineage"}}}
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    snapshot = replace(snapshot, lineage_allowed=False, lineage_reason="lineage_blocked")
    decision = runtime._build_execution_decision(
        action=ResponseCreateOutcomeAction.BLOCK,
        reason_code="lineage_blocked",
        explanation="Tool lineage guard blocked response.create.",
        selected_candidate_id="tool_lineage_guard",
    )

    observation = build_response_create_observation(
        snapshot=snapshot,
        execution_decision=decision,
        lifecycle_decision=None,
        same_turn_owner_reason=None,
        canonical_audio_started=None,
    )
    candidate = next(candidate for candidate in observation.candidates if candidate.candidate_id == "tool_lineage_guard")

    assert candidate.owner_scope == "subsystem_local"
    assert "owner_scope_conservative_for_lineage_guard" in candidate.normalization_warnings
