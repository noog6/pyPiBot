from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import replace
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.realtime.response_create_runtime import ResponseCreateOutcomeAction, ResponseCreateRuntime
from ai.realtime.types import CanonicalResponseState
from ai.realtime.transport import RealtimeTransport
from ai.realtime_api import RealtimeAPI
from ai.interaction_lifecycle_policy import ResponseCreateDecision, ResponseCreateDecisionAction
from core.logging import logger


class _Ws:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.events.append(json.loads(payload))


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._transport = RealtimeTransport(
        connect_fn=lambda *args, **kwargs: None,
        validate_outbound_endpoint=lambda _url: None,
    )
    api._last_response_create_ts = None
    api._response_create_debug_trace = False
    api._active_response_id = None
    api._response_in_flight = False
    api._audio_playback_busy = False
    api._response_create_queue = deque()
    api._pending_response_create = None
    api._response_create_turn_counter = 0
    api._current_response_turn_id = None
    api._queued_confirmation_reminder_keys = set()
    api._response_done_serial = 0
    api._record_ai_call = lambda: None
    api._track_outgoing_event = lambda *args, **kwargs: None
    api._response_created_canonical_keys = set()
    api._preference_recall_suppressed_turns = set()
    api._response_create_runtime = ResponseCreateRuntime(api)
    return api


def _capture_info_messages(monkeypatch) -> list[str]:
    info_messages: list[str] = []
    monkeypatch.setattr(logger, "info", lambda msg, *args: info_messages.append(msg % args if args else msg))
    return info_messages


def test_prepare_response_create_snapshot_captures_expected_facts_and_mutations() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    expected_canonical_key = api._canonical_utterance_key(turn_id="turn_ctx", input_event_key="item_ctx")
    api._response_created_canonical_keys = {expected_canonical_key}
    api._preference_recall_suppressed_turns = {"turn_ctx"}
    api._peek_pending_preference_memory_context_payload = lambda *, turn_id, input_event_key: {
        "prompt_note": "Preference recall context for this SAME response: Top recalled value: Vim",
        "hit": True,
        "source": "memory",
    }

    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_ctx",
                "input_event_key": "item_ctx",
                "memory_intent_subtype": "preference_recall",
            }
        },
    }

    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=123.0,
    )

    assert prepared_snapshot.turn_id == "turn_ctx"
    assert prepared_snapshot.input_event_key == "item_ctx"
    assert prepared_snapshot.canonical_key == expected_canonical_key
    assert prepared_snapshot.effective_memory_note and "Vim" in prepared_snapshot.effective_memory_note
    assert prepared_snapshot.had_pending_preference_context is True
    assert prepared_snapshot.created_keys == {expected_canonical_key}
    assert prepared_snapshot.suppression_turns == {"turn_ctx"}
    assert "Memory-intent response mode" in event["response"]["instructions"]


def test_decide_response_create_action_uses_prepared_snapshot_for_server_auto_defer() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime

    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_srv", "input_event_key": "synthetic_server_auto_7"}},
    }

    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="server_auto",
        utterance_context=None,
        memory_brief_note=None,
        now=321.0,
    )
    decision = runtime.decide_response_create_action(prepared_snapshot)

    assert decision.action is ResponseCreateOutcomeAction.SCHEDULE
    assert decision.reason_code == "awaiting_transcript_final"


def test_prepare_response_create_snapshot_does_not_let_nonrelease_tool_followup_steal_turn_ownership() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._active_input_event_key_by_turn_id = {}
    api._active_input_event_key_by_turn_id["turn_tool_parent"] = "item_parent_owner"

    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_tool_parent",
                "input_event_key": "tool:call_owner_prepare",
                "tool_followup": "true",
                "tool_call_id": "call_owner_prepare",
            }
        },
    }

    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=456.0,
    )

    assert prepared_snapshot.tool_followup is True
    assert prepared_snapshot.tool_followup_release is False
    assert api._active_input_event_key_for_turn("turn_tool_parent") == "item_parent_owner"


def test_prepare_response_create_snapshot_prefers_provisional_response_facade_lookup() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._pending_server_auto_response_for_turn = lambda *, turn_id: None
    api._provisional_response_for_turn = lambda *, turn_id: object()

    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_facade", "input_event_key": "item_facade"}},
    }

    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=789.0,
    )

    assert prepared_snapshot.pending_server_auto_present is True


def test_send_response_create_executes_side_effects_after_direct_send_decision() -> None:
    api = _make_api_stub()
    ws = _Ws()

    sent = asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create", "response": {"metadata": {"turn_id": "turn_send", "input_event_key": "item_send"}}},
            origin="assistant_message",
            memory_brief_note="Turn memory brief: ...",
        )
    )

    assert sent is True
    assert [event["type"] for event in ws.events] == [
        "conversation.item.create",
        "response.create",
    ]
    assert api._last_response_create_ts is not None
    assert api._response_in_flight is True
    assert api._is_response_already_delivered(turn_id="turn_send", input_event_key="item_send") is False


def test_response_create_arbitration_snapshot_parity_for_active_response_schedule() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._response_in_flight = True
    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_busy", "input_event_key": "item_busy"}},
    }

    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(prepared_snapshot)

    assert decision.action is ResponseCreateOutcomeAction.SCHEDULE
    assert decision.reason_code == "active_response"

    scheduled = runtime.schedule_pending_response_create(
        websocket=object(),
        response_create_event=event,
        origin="assistant_message",
        reason="legacy_active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert scheduled is False
    assert api._pending_response_create is not None
    pending_metadata = api._extract_response_create_metadata(api._pending_response_create.event)
    assert pending_metadata["input_event_key"] == "item_busy"


def test_evaluate_response_create_attempt_records_partial_turn_trace_without_changing_decision() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_trace", "input_event_key": "item_trace"}},
    }

    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    expected_decision = runtime.decide_response_create_action(snapshot)

    prepared_snapshot, decision = runtime.evaluate_response_create_attempt(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )

    assert prepared_snapshot == snapshot
    assert decision == expected_decision
    trace = api._turn_arbitration_trace_by_key[(prepared_snapshot.run_id, prepared_snapshot.turn_id)]
    assert trace.trace_partial is True
    assert trace.trace_complete is False
    assert trace.response_create_observation is not None


def test_response_create_arbitration_snapshot_parity_for_awaiting_transcript_schedule() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_wait", "input_event_key": "synthetic_server_auto_9"}},
    }

    prepared_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="server_auto",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(prepared_snapshot)

    assert decision.action is ResponseCreateOutcomeAction.SCHEDULE
    assert decision.reason_code == "awaiting_transcript_final"

    scheduled = runtime.schedule_pending_response_create(
        websocket=object(),
        response_create_event=event,
        origin="server_auto",
        reason="legacy_awaiting_transcript_final",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert scheduled is False
    assert api._pending_response_create is not None


def test_response_create_arbitration_blocks_already_delivered_and_already_created() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    turn_id = "turn_dup"
    input_event_key = "item_dup"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._canonical_first_audio_started = lambda _canonical_key: False
    api._set_response_delivery_state(turn_id=turn_id, input_event_key=input_event_key, state="delivered")
    api._response_created_canonical_keys = {canonical_key}
    event = {"type": "response.create", "response": {"metadata": {"turn_id": turn_id, "input_event_key": input_event_key}}}

    delivered_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    delivered_decision = runtime.decide_response_create_action(delivered_snapshot)

    assert delivered_decision.action is ResponseCreateOutcomeAction.BLOCK
    assert delivered_decision.reason_code == "already_delivered"

    created_api = _make_api_stub()
    created_runtime = created_api._response_create_runtime
    created_api._canonical_first_audio_started = lambda _canonical_key: False
    created_api._response_created_canonical_keys = {canonical_key}
    created_snapshot = created_runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=2.0,
    )
    created_decision = created_runtime.decide_response_create_action(created_snapshot)

    assert created_decision.action is ResponseCreateOutcomeAction.BLOCK
    assert created_decision.reason_code == "canonical_response_already_created"


def test_required_deliverable_motion_gate_followthrough_bypasses_already_done_and_created_guards() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    turn_id = "turn_motion_followthrough"
    input_event_key = "item_motion_followthrough"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._response_created_canonical_keys = {canonical_key}
    api._set_response_delivery_state(turn_id=turn_id, input_event_key=input_event_key, state="done")
    api._continuity_ledger_instance = lambda: SimpleNamespace(compound_owner_turn_id=lambda: turn_id)
    api._should_suppress_nonessential_runtime_emission_during_followthrough = lambda **_kwargs: (False, "")
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": turn_id,
                "input_event_key": input_event_key,
                "local_runtime_followthrough": "true",
                "followthrough_step_output_policy": "required_deliverable",
                "followthrough_post_completion_reason": "required_deliverable_owed",
                "followthrough_dispatch_source": "deterministic_followthrough_motion_gate",
            }
        },
    }

    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(snapshot)

    assert snapshot.single_flight_block_reason == ""
    assert snapshot.already_created_for_canonical_key is False
    assert decision.action is ResponseCreateOutcomeAction.SEND
    assert decision.reason_code == "direct_send"


def test_non_motion_gate_required_deliverable_followthrough_keeps_duplicate_guards() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    turn_id = "turn_no_override"
    input_event_key = "item_no_override"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._response_created_canonical_keys = {canonical_key}
    api._set_response_delivery_state(turn_id=turn_id, input_event_key=input_event_key, state="done")
    api._continuity_ledger_instance = lambda: SimpleNamespace(compound_owner_turn_id=lambda: turn_id)
    api._should_suppress_nonessential_runtime_emission_during_followthrough = lambda **_kwargs: (False, "")
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": turn_id,
                "input_event_key": input_event_key,
                "local_runtime_followthrough": "true",
                "followthrough_step_output_policy": "required_deliverable",
                "followthrough_post_completion_reason": "required_deliverable_owed",
                "followthrough_dispatch_source": "other_source",
            }
        },
    }

    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(snapshot)

    assert snapshot.single_flight_block_reason == "already_done"
    assert snapshot.already_created_for_canonical_key is True
    assert decision.action is ResponseCreateOutcomeAction.BLOCK
    assert decision.reason_code == "already_delivered"


def test_already_done_reason_normalization_stays_in_runtime_layer() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    turn_id = "turn_already_done"
    input_event_key = "item_already_done"
    api._set_response_delivery_state(turn_id=turn_id, input_event_key=input_event_key, state="delivered")
    event = {"type": "response.create", "response": {"metadata": {"turn_id": turn_id, "input_event_key": input_event_key}}}
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    api._lifecycle_policy = lambda: types.SimpleNamespace(
        decide_response_create=lambda **_kwargs: ResponseCreateDecision(
            action=ResponseCreateDecisionAction.BLOCK,
            reason_code="already_done",
            selected_candidate_id="already_delivered",
        )
    )

    decision = runtime.decide_response_create_action(snapshot)

    assert decision.action is ResponseCreateOutcomeAction.BLOCK
    assert decision.reason_code == "already_delivered"
    assert decision.selected_candidate_id == "already_delivered"


def test_response_create_arbitration_drops_same_turn_owned_assistant_message() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._assistant_message_same_turn_owner_reason = lambda **_kwargs: "tool_followup_owned"
    event = {"type": "response.create", "response": {"metadata": {"turn_id": "turn_owner", "input_event_key": "item_owner"}}}

    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(snapshot)

    assert snapshot.same_turn_owner_present is True
    assert decision.action is ResponseCreateOutcomeAction.DROP
    assert decision.reason_code == "same_turn_already_owned"


def test_response_create_arbitration_bounded_clarify_allowlist_bypasses_same_turn_owner_drop() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._assistant_message_same_turn_owner_reason = lambda **_kwargs: "terminal_deliverable_owned"
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_owner",
                "input_event_key": "item_owner:clarify",
                "trigger": "asr_verify_on_risk",
                "reason": "visual_unavailable",
                "clarify_mode": "bounded",
            }
        },
    }

    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(snapshot)

    assert snapshot.same_turn_owner_present is False
    assert decision.action is not ResponseCreateOutcomeAction.DROP
    assert decision.reason_code != "same_turn_already_owned"


def test_response_create_arbitration_non_clarify_assistant_message_still_drops_on_same_turn_owner() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._assistant_message_same_turn_owner_reason = lambda **_kwargs: "terminal_deliverable_owned"
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_owner",
                "input_event_key": "item_owner",
                "trigger": "assistant_message",
            }
        },
    }

    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="assistant_message",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime.decide_response_create_action(snapshot)

    assert snapshot.same_turn_owner_present is True
    assert decision.action is ResponseCreateOutcomeAction.DROP
    assert decision.reason_code == "same_turn_already_owned"


def test_response_create_terminal_state_is_block_not_drop() -> None:
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

    assert snapshot.terminal_state_blocked is True
    assert decision.action is ResponseCreateOutcomeAction.BLOCK
    assert decision.reason_code == "canonical_terminal_state"


def test_response_create_outcome_is_canonical_info_log_and_arbitration_log_is_secondary(monkeypatch) -> None:
    api = _make_api_stub()
    ws = _Ws()
    info_messages = _capture_info_messages(monkeypatch)
    debug_messages: list[str] = []
    monkeypatch.setattr(logger, "debug", lambda msg, *args: debug_messages.append(msg % args if args else msg))

    sent = asyncio.run(
        api._send_response_create(
            ws,
            {"type": "response.create", "response": {"metadata": {"turn_id": "turn_log", "input_event_key": "item_log"}}},
            origin="assistant_message",
        )
    )

    assert sent is True
    assert any("response_create_outcome" in message and "action=SEND" in message for message in info_messages)
    assert not any("arbitration_decision surface=response_create" in message for message in info_messages)
    assert any("arbitration_decision surface=response_create" in message for message in debug_messages)


def test_response_create_parity_across_direct_schedule_and_drain_evaluation() -> None:
    cases = [
        (
            "active_response",
            "assistant_message",
            {"turn_id": "turn_busy", "input_event_key": "item_busy"},
            lambda api: setattr(api, "_response_in_flight", True),
            ResponseCreateOutcomeAction.SCHEDULE,
            "active_response",
        ),
        (
            "awaiting_transcript_final",
            "server_auto",
            {"turn_id": "turn_wait", "input_event_key": "synthetic_server_auto_11"},
            lambda api: None,
            ResponseCreateOutcomeAction.SCHEDULE,
            "awaiting_transcript_final",
        ),
        (
            "same_turn_already_owned",
            "assistant_message",
            {"turn_id": "turn_owned", "input_event_key": "item_owned"},
            lambda api: setattr(api, "_assistant_message_same_turn_owner_reason", lambda **_kwargs: "tool_followup_owned"),
            ResponseCreateOutcomeAction.DROP,
            "same_turn_already_owned",
        ),
        (
            "canonical_response_already_created",
            "assistant_message",
            {"turn_id": "turn_created", "input_event_key": "item_created"},
            lambda api: api._response_created_canonical_keys.add(
                api._canonical_utterance_key(turn_id="turn_created", input_event_key="item_created")
            ),
            ResponseCreateOutcomeAction.BLOCK,
            "canonical_response_already_created",
        ),
        (
            "already_delivered",
            "assistant_message",
            {"turn_id": "turn_delivered", "input_event_key": "item_delivered"},
            lambda api: api._set_response_delivery_state(
                turn_id="turn_delivered",
                input_event_key="item_delivered",
                state="delivered",
            ),
            ResponseCreateOutcomeAction.BLOCK,
            "already_delivered",
        ),
    ]

    for _name, origin, metadata, configure, expected_action, expected_reason in cases:
        api = _make_api_stub()
        runtime = api._response_create_runtime
        api._canonical_first_audio_started = lambda _canonical_key: False
        configure(api)
        event = {"type": "response.create", "response": {"metadata": dict(metadata)}}

        direct_snapshot, direct_decision = runtime.evaluate_response_create_attempt(
            response_create_event=event,
            origin=origin,
            utterance_context=None,
            memory_brief_note=None,
            now=1.0,
        )
        assert direct_decision.action is expected_action
        assert direct_decision.reason_code == expected_reason

        runtime.schedule_pending_response_create(
            websocket=object(),
            response_create_event={"type": "response.create", "response": {"metadata": dict(metadata)}},
            origin=origin,
            reason="parity_probe",
            record_ai_call=False,
            debug_context=None,
            memory_brief_note=None,
        )

        if expected_action is ResponseCreateOutcomeAction.SCHEDULE:
            assert api._pending_response_create is not None
            drain_snapshot, drain_decision = runtime.evaluate_response_create_attempt(
                response_create_event=api._pending_response_create.event,
                origin=api._pending_response_create.origin,
                utterance_context=None,
                memory_brief_note=api._pending_response_create.memory_brief_note,
                now=2.0,
            )
            assert drain_snapshot.canonical_key == direct_snapshot.canonical_key
            assert drain_decision.action is expected_action
            assert drain_decision.reason_code == expected_reason
        else:
            assert api._pending_response_create is None


def test_post_decision_finalizer_returns_same_tool_followup_deliverable_drop_for_send_and_schedule() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._tool_followup_state_by_canonical_key = {}
    api._should_suppress_tool_followup_after_turn_deliverable = lambda **_kwargs: True
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_finalizer_parent",
                "input_event_key": "item_finalizer_tool",
                "parent_turn_id": "turn_finalizer_parent",
                "parent_input_event_key": "item_parent",
                "tool_followup": "true",
                "tool_call_id": "call-finalizer",
            }
        },
    }
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    selected = runtime._build_execution_decision(
        action=ResponseCreateOutcomeAction.SCHEDULE,
        reason_code="active_response",
        explanation="queued while response active",
        selected_candidate_id="active_response",
    )

    finalized_send = runtime._finalize_response_create_execution_decision(
        prepared_snapshot=snapshot,
        decision=selected,
        execution_path="send",
    )
    finalized_schedule = runtime._finalize_response_create_execution_decision(
        prepared_snapshot=snapshot,
        decision=selected,
        execution_path="schedule",
    )

    assert finalized_send.action is ResponseCreateOutcomeAction.DROP
    assert finalized_send.reason_code == "tool_followup_final_deliverable_already_sent"
    assert finalized_schedule.action is ResponseCreateOutcomeAction.DROP
    assert finalized_schedule.reason_code == finalized_send.reason_code


def test_post_decision_finalizer_allows_tool_followup_when_parent_turn_followthrough_still_open() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._tool_followup_state_by_canonical_key = {}
    parent_canonical_key = api._canonical_utterance_key(
        turn_id="turn_finalizer_parent_open_followthrough",
        input_event_key="item_parent",
    )
    api._canonical_response_state_by_key = {
        parent_canonical_key: CanonicalResponseState(
            created=True,
            done=True,
            deliverable_observed=True,
            deliverable_class="final",
            origin="assistant_message",
            response_id="resp-parent",
            turn_id="turn_finalizer_parent_open_followthrough",
            input_event_key="item_parent",
        )
    }
    api._turn_followthrough_chain_remaining = lambda *, turn_id, include_report_followup=True: (
        turn_id == "turn_finalizer_parent_open_followthrough" and include_report_followup
    )
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_finalizer_parent_open_followthrough",
                "input_event_key": "item_tool_followup",
                "parent_turn_id": "turn_finalizer_parent_open_followthrough",
                "parent_input_event_key": "item_parent",
                "tool_followup": "true",
                "tool_call_id": "call-finalizer-open",
            }
        },
    }
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    selected = runtime._build_execution_decision(
        action=ResponseCreateOutcomeAction.SCHEDULE,
        reason_code="active_response",
        explanation="queued while response active",
        selected_candidate_id="active_response",
    )

    finalized_schedule = runtime._finalize_response_create_execution_decision(
        prepared_snapshot=snapshot,
        decision=selected,
        execution_path="schedule",
    )

    assert finalized_schedule.action is ResponseCreateOutcomeAction.SCHEDULE
    assert finalized_schedule.reason_code == "active_response"


def test_post_decision_finalizer_returns_same_existing_state_drop_for_send_and_schedule() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    canonical_key = api._canonical_utterance_key(turn_id="turn_finalizer_existing", input_event_key="item_finalizer_existing")
    api._tool_followup_state_by_canonical_key = {canonical_key: "created"}
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_finalizer_existing",
                "input_event_key": "item_finalizer_existing",
                "tool_followup": "true",
                "tool_call_id": "call-existing",
            }
        },
    }
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    selected = runtime._build_execution_decision(
        action=ResponseCreateOutcomeAction.SCHEDULE,
        reason_code="active_response",
        explanation="queued while response active",
        selected_candidate_id="active_response",
    )

    finalized_send = runtime._finalize_response_create_execution_decision(
        prepared_snapshot=snapshot,
        decision=selected,
        execution_path="send",
    )
    finalized_schedule = runtime._finalize_response_create_execution_decision(
        prepared_snapshot=snapshot,
        decision=selected,
        execution_path="schedule",
    )

    assert finalized_send.action is ResponseCreateOutcomeAction.DROP
    assert finalized_send.reason_code == "already_created"
    assert finalized_schedule.action is ResponseCreateOutcomeAction.DROP
    assert finalized_schedule.reason_code == finalized_send.reason_code


def test_tool_followup_final_deliverable_drop_logs_outcome_and_leaves_no_zombie_retry(monkeypatch) -> None:
    api = _make_api_stub()
    api._tool_followup_state_by_canonical_key = {}
    api._should_suppress_tool_followup_after_turn_deliverable = lambda **_kwargs: True
    ws = _Ws()
    info_messages = _capture_info_messages(monkeypatch)
    tool_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_parent",
                "input_event_key": "item_tool_followup",
                "parent_turn_id": "turn_parent",
                "parent_input_event_key": "item_parent",
                "tool_followup": "true",
                "tool_call_id": "call-final",
            }
        },
    }
    canonical_key = api._canonical_utterance_key(turn_id="turn_parent", input_event_key="item_tool_followup")

    sent = asyncio.run(api._send_response_create(ws, tool_event, origin="tool_output"))

    assert sent is False
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert api._pending_response_create is None
    assert list(api._response_create_queue) == []
    assert any(
        "response_create_outcome" in message
        and "action=DROP" in message
        and "reason_code=tool_followup_final_deliverable_already_sent" in message
        for message in info_messages
    )


def test_schedule_pending_tool_followup_final_deliverable_overrides_schedule_and_leaves_no_retry(monkeypatch) -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    api._tool_followup_state_by_canonical_key = {}
    api._should_suppress_tool_followup_after_turn_deliverable = lambda **_kwargs: True
    api._response_in_flight = True
    info_messages = _capture_info_messages(monkeypatch)
    tool_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_parent_busy",
                "input_event_key": "item_tool_followup_busy",
                "parent_turn_id": "turn_parent_busy",
                "parent_input_event_key": "item_parent_busy",
                "tool_followup": "true",
                "tool_call_id": "call-final-busy",
            }
        },
    }
    canonical_key = api._canonical_utterance_key(turn_id="turn_parent_busy", input_event_key="item_tool_followup_busy")

    scheduled = runtime.schedule_pending_response_create(
        websocket=object(),
        response_create_event=tool_event,
        origin="tool_output",
        reason="active_response",
        record_ai_call=False,
        debug_context=None,
        memory_brief_note=None,
    )

    assert scheduled is False
    assert api._tool_followup_state(canonical_key=canonical_key) == "dropped"
    assert api._pending_response_create is None
    assert list(api._response_create_queue) == []
    assert any(
        "response_create_outcome" in message
        and "action=DROP" in message
        and "reason_code=tool_followup_final_deliverable_already_sent" in message
        for message in info_messages
    )


def test_response_create_arbitration_blocks_preference_suppression_and_drops_tool_followup(monkeypatch) -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    info_messages: list[str] = []
    monkeypatch.setattr(logger, "info", lambda msg, *args: info_messages.append(msg % args if args else msg))

    api._preference_recall_suppressed_turns = {"turn_pref"}
    suppression_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_pref", "input_event_key": ""}},
    }
    suppression_snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=suppression_event,
        origin="server_auto",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    suppression_snapshot = replace(
        suppression_snapshot,
        awaiting_transcript_final=False,
        suppression_active=True,
        preference_recall_suppression_active=True,
    )
    suppression_decision = runtime.decide_response_create_action(suppression_snapshot)

    assert suppression_decision.action is ResponseCreateOutcomeAction.BLOCK
    assert suppression_decision.reason_code == "preference_recall_suppressed"

    turn_id = "turn_tool"
    input_event_key = "item_tool"
    canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
    api._tool_followup_state_by_canonical_key = {}
    api._tool_followup_state_by_canonical_key[canonical_key] = "created"
    tool_event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": turn_id, "input_event_key": input_event_key, "tool_followup": "true", "tool_call_id": "call-1"}},
    }

    sent = asyncio.run(api._send_response_create(object(), tool_event, origin="tool_output"))

    assert sent is False
    assert any("response_create_outcome" in message for message in info_messages)
    assert any("action=DROP" in message and "reason_code=already_created" in message for message in info_messages)


def test_evaluate_response_create_attempt_does_not_log_info_turn_summary_for_partial_expected_trace() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_review_info_skip", "input_event_key": "item_review_info_skip"}},
    }

    with patch("ai.realtime.response_create_runtime.logger.info") as info_log:
        runtime.evaluate_response_create_attempt(
            response_create_event=event,
            origin="assistant_message",
            utterance_context=None,
            memory_brief_note=None,
            now=1.0,
        )

    assert all(
        not (call.args and call.args[0] == "decision_arbitration_turn_summary run_id=%s turn_id=%s review_bucket=%s review_priority=%s overall_verdict=%s overall_summary=%s")
        for call in info_log.call_args_list
    )


def test_evaluate_response_create_attempt_logs_turn_review_summary() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_review_log", "input_event_key": "item_review_log"}},
    }

    with patch("ai.realtime.response_create_runtime.logger.debug") as debug_log:
        _prepared_snapshot, _decision = runtime.evaluate_response_create_attempt(
            response_create_event=event,
            origin="assistant_message",
            utterance_context=None,
            memory_brief_note=None,
            now=1.0,
        )

    payload = None
    for call in debug_log.call_args_list:
        if call.args and call.args[0] == "decision_adapter_turn_review_summary payload=%s":
            payload = call.args[1]
            break
    assert payload is not None
    assert payload["trace_partial"] is True
    assert payload["response_create_summary"] == "response.create allowed (direct_send)"
    assert payload["observational_only"] is True


def test_tool_followup_metadata_cap_preserves_required_deliverable_markers() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn-cap",
                "input_event_key": "tool:call-cap",
                "tool_followup": "true",
                "tool_call_id": "call-cap",
                "tool_followup_release": "true",
                "blocked_by_response_id": "resp-1",
                "parent_turn_id": "turn-cap",
                "parent_input_event_key": "item-parent",
                "tool_followup_suppress_if_parent_covered": "true",
                "tool_followup_status_only": "true",
                "tool_followup_silent_audio": "true",
                "tool_followup_silent_user_facing_output": "true",
                "tool_followup_step_output_policy": "required_deliverable",
                "followthrough_step_output_policy": "required_deliverable",
                "tool_followup_post_completion_reason": "required_deliverable_owed",
                "followthrough_post_completion_reason": "required_deliverable_owed",
                "followthrough_catchup_payload": "{\"turn_id\":\"turn-cap\",\"completed_steps\":[{\"tool_name\":\"gesture_look_center\"}]}",
                "local_runtime_followthrough": "true",
                "followthrough_dispatch_source": "deterministic_followthrough_motion_gate",
                "tool_name": "gesture_look_center",
                "gesture_motion_status": "completed",
            }
        },
    }

    runtime._enforce_tool_followup_metadata_limit(
        response_create_event=event,
        canonical_key="run-1:turn-cap:tool:call-cap",
    )

    metadata = event["response"]["metadata"]
    assert len(metadata) <= runtime._PROVIDER_METADATA_MAX_PROPERTIES
    assert metadata.get("followthrough_step_output_policy") == "required_deliverable"
    assert metadata.get("followthrough_post_completion_reason") == "required_deliverable_owed"
    assert "followthrough_catchup_payload" in metadata
    assert metadata.get("tool_followup_step_output_policy") is None
    assert metadata.get("tool_followup_post_completion_reason") is None
    assert metadata.get("tool_name") is None


def test_record_tool_followup_observation_logs_info_turn_summary_for_suspicious_partial_trace() -> None:
    api = _make_api_stub()
    api._current_run_id = lambda: "run-tool-info"
    api._turn_arbitration_trace_by_key = {}

    with patch("ai.realtime_api.logger.info") as info_log, patch("ai.realtime_api.logger.debug"):
        api._record_tool_followup_observation(
            turn_id="turn-tool-info",
            input_event_key="tool:call_tool_info",
            canonical_key="turn-tool-info::tool:call_tool_info",
            origin="tool_output",
            parent_coverage_state="unknown",
            followup_outcome_posture="suppressed",
            native_reason_code="parent_unresolved",
            native_outcome_action="DROP",
            authority_seam="ai.realtime_api",
        )
    trace = api._turn_arbitration_trace_by_key[(api._current_run_id() or "", "turn-tool-info")]
    observation = trace.tool_followup_observations[-1]
    assert observation.context.authority_retained_by == "ai.tool_followup_arbitration.decide_tool_followup_arbitration"
    assert observation.decision.authority_retained_by == "ai.tool_followup_arbitration.decide_tool_followup_arbitration"
    assert "provenance_source:tool_followup:ai.realtime_api" in observation.normalization_warnings

    summary_call = None
    for call in info_log.call_args_list:
        if call.args and call.args[0] == "decision_arbitration_turn_summary run_id=%s turn_id=%s review_bucket=%s review_priority=%s overall_verdict=%s overall_summary=%s":
            summary_call = call
            break
    assert summary_call is not None
    assert summary_call.args[1] == "run-tool-info"
    assert summary_call.args[2] == "turn-tool-info"
    assert summary_call.args[3] == "suspicious"


def test_record_tool_followup_observation_dedupes_identical_info_turn_summary() -> None:
    api = _make_api_stub()
    api._current_run_id = lambda: "run-tool-info"
    api._turn_arbitration_trace_by_key = {}

    with patch("ai.realtime_api.logger.info") as info_log, patch("ai.realtime_api.logger.debug"):
        api._record_tool_followup_observation(
            turn_id="turn-tool-info",
            input_event_key="tool:call_tool_info",
            canonical_key="turn-tool-info::tool:call_tool_info",
            origin="tool_output",
            parent_coverage_state="unknown",
            followup_outcome_posture="suppressed",
            native_reason_code="parent_unresolved",
            native_outcome_action="DROP",
            authority_seam="ai.realtime_api",
        )
        api._record_tool_followup_observation(
            turn_id="turn-tool-info",
            input_event_key="tool:call_tool_info",
            canonical_key="turn-tool-info::tool:call_tool_info",
            origin="tool_output",
            parent_coverage_state="unknown",
            followup_outcome_posture="suppressed",
            native_reason_code="parent_unresolved",
            native_outcome_action="DROP",
            authority_seam="ai.realtime_api",
        )

    summary_calls = [
        call for call in info_log.call_args_list
        if call.args
        and call.args[0] == "decision_arbitration_turn_summary run_id=%s turn_id=%s review_bucket=%s review_priority=%s overall_verdict=%s overall_summary=%s"
    ]
    assert len(summary_calls) == 1


def test_record_tool_followup_observation_keeps_suspicious_updates_visible_in_info() -> None:
    api = _make_api_stub()
    api._current_run_id = lambda: "run-tool-info"
    api._turn_arbitration_trace_by_key = {}

    with patch("ai.realtime_api.logger.info") as info_log, patch("ai.realtime_api.logger.debug"):
        api._record_tool_followup_observation(
            turn_id="turn-tool-info",
            input_event_key="tool:call_tool_info",
            canonical_key="turn-tool-info::tool:call_tool_info",
            origin="tool_output",
            parent_coverage_state="unknown",
            followup_outcome_posture="suppressed",
            native_reason_code="parent_unresolved",
            native_outcome_action="DROP",
            authority_seam="ai.realtime_api",
        )
        api._record_tool_followup_observation(
            turn_id="turn-tool-info",
            input_event_key="tool:call_tool_info",
            canonical_key="turn-tool-info::tool:call_tool_info",
            origin="tool_output",
            parent_coverage_state="uncovered",
            followup_outcome_posture="released",
            native_reason_code="release_after_response_done",
            native_outcome_action="SCHEDULE",
            authority_seam="ai.realtime_api",
        )

    summary_calls = [
        call for call in info_log.call_args_list
        if call.args
        and call.args[0] == "decision_arbitration_turn_summary run_id=%s turn_id=%s review_bucket=%s review_priority=%s overall_verdict=%s overall_summary=%s"
    ]
    assert len(summary_calls) == 2
    assert "tool followup suppressed" in summary_calls[0].args[6]
    assert "tool followup released" in summary_calls[1].args[6]


def test_emit_turn_review_summary_info_skips_coherent_complete_deferred_summary() -> None:
    api = _make_api_stub()
    trace = SimpleNamespace(
        run_id="run-deferred",
        turn_id="turn-deferred",
        trace_complete=True,
        trace_partial=False,
        review_summary=SimpleNamespace(
            run_id="run-deferred",
            turn_id="turn-deferred",
            review_bucket="coherent",
            review_priority="low",
            overall_verdict="coherent turn trace",
            overall_summary="response.create deferred (active_response); terminal deliverable selected; semantic owner stayed on execution canonical; no tool followup observations; no suspicious signals",
            response_create_summary="response.create deferred (active_response)",
            terminal_summary="terminal deliverable selected",
            semantic_owner_summary="semantic owner stayed on execution canonical",
            tool_followup_summary="no tool followup observations",
            trace_complete=True,
        ),
    )

    with patch("ai.realtime_api.logger.info") as info_log:
        emitted = api._emit_turn_review_summary_info_if_material(trace)

    assert emitted is False
    info_log.assert_not_called()


def test_evaluate_response_create_attempt_logs_turn_diagnostics() -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    event = {
        "type": "response.create",
        "response": {"metadata": {"turn_id": "turn_trace_log", "input_event_key": "item_trace_log"}},
    }

    with patch("ai.realtime.response_create_runtime.logger.debug") as debug_log:
        _prepared_snapshot, _decision = runtime.evaluate_response_create_attempt(
            response_create_event=event,
            origin="assistant_message",
            utterance_context=None,
            memory_brief_note=None,
            now=1.0,
        )

    payload = None
    for call in debug_log.call_args_list:
        if call.args and call.args[0] == "decision_adapter_turn_diagnostics payload=%s":
            payload = call.args[1]
            break
    assert payload is not None
    assert payload["trace_partial"] is True
    assert "expected_terminal_selection_missing" in payload["diagnostic_codes"]
    assert payload["observational_only"] is True


def test_response_create_runtime_logs_contract_breach_for_tool_followup_owner_mismatch(monkeypatch) -> None:
    api = _make_api_stub()
    runtime = api._response_create_runtime
    info_messages = _capture_info_messages(monkeypatch)
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_tool",
                "input_event_key": "tool:call_1",
                "tool_followup": "true",
                "tool_call_id": "call_1",
            }
        },
    }
    snapshot = runtime.prepare_response_create_snapshot(
        response_create_event=event,
        origin="tool_output",
        utterance_context=None,
        memory_brief_note=None,
        now=1.0,
    )
    decision = runtime._build_execution_decision(
        action=ResponseCreateOutcomeAction.DROP,
        reason_code="same_turn_already_owned",
        explanation="drop",
        selected_candidate_id="same_turn_owner",
    )

    runtime._log_response_create_outcome(snapshot=snapshot, decision=decision)

    assert any("contract_breach_detected" in msg for msg in info_messages)
