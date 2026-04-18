from __future__ import annotations

import os
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.realtime.response_create_runtime import (
    ResponseCreateOutcomeAction,
    ResponseCreatePreparedSnapshot,
    ResponseCreateRuntime,
)
from ai.realtime_api import RealtimeAPI


def _make_attention_api(*, assistant_name: str = "Theo Prime") -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._assistant_name = assistant_name
    api._attention_gate_hold_window_s = 2.0
    api._attention_gate_hold_until_monotonic = None
    api._current_run_id = lambda: "run-attn"
    return api


def _snapshot(*, attention_gate_closed: bool) -> ResponseCreatePreparedSnapshot:
    return ResponseCreatePreparedSnapshot(
        now=10.0,
        run_id="run-attn",
        origin="assistant_message",
        normalized_origin="assistant_message",
        response_create_event={"type": "response.create"},
        response_metadata={},
        turn_id="turn-1",
        input_event_key="item-1",
        canonical_key="turn-1:item-1",
        effective_memory_note=None,
        had_pending_preference_context=False,
        preference_note="",
        tool_followup=False,
        tool_call_id="",
        tool_followup_release=False,
        tool_followup_state="none",
        consumes_canonical_slot=True,
        explicit_multipart=False,
        transcript_upgrade_replacement=False,
        allow_audio_started_upgrade=False,
        suppression_turns=set(),
        created_keys=set(),
        single_flight_block_reason="",
        suppression_active=False,
        awaiting_transcript_final=False,
        preference_recall_lock_blocked=False,
        preference_recall_suppression_active=False,
        preference_recall_lock_active=False,
        response_in_flight=False,
        active_response_present=False,
        already_delivered=False,
        already_created_for_canonical_key=False,
        same_turn_owner_reason=None,
        same_turn_owner_present=False,
        pending_server_auto_present=False,
        has_safety_override=False,
        audio_playback_busy=False,
        terminal_state_blocked=False,
        lineage_allowed=True,
        lineage_reason="",
        attention_gate_closed=attention_gate_closed,
    )


def test_direct_address_opens_attention_hold_window() -> None:
    api = _make_attention_api()

    admitted, reason = api._evaluate_attention_gate_admission(
        transcript="Theo Prime, what's the status?",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )

    assert admitted is True
    assert reason == "direct_address"
    assert api._attention_gate_hold_until_monotonic == 102.0


def test_configured_hold_window_value_is_applied() -> None:
    api = _make_attention_api()
    api._attention_gate_hold_window_s = 0.4

    admitted, reason = api._evaluate_attention_gate_admission(
        transcript="Theo Prime, status check.",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )

    assert admitted is True
    assert reason == "direct_address"
    assert api._attention_gate_hold_until_monotonic == 100.4


def test_non_addressed_background_speech_is_ignored_by_default() -> None:
    api = _make_attention_api()

    admitted, reason = api._evaluate_attention_gate_admission(
        transcript="Looks like rain this weekend.",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )

    assert admitted is False
    assert reason == "attention_gate_closed"


def test_followup_within_hold_window_is_admitted() -> None:
    api = _make_attention_api()
    api._attention_gate_hold_window_s = 0.5
    api._evaluate_attention_gate_admission(
        transcript="Theo Prime, status.",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )

    admitted, reason = api._evaluate_attention_gate_admission(
        transcript="How about tomorrow?",
        turn_id="turn-1",
        input_event_key="item-2",
        now=100.49,
    )

    assert admitted is True
    assert reason == "active_hold_window"


def test_followup_outside_hold_window_is_ignored() -> None:
    api = _make_attention_api()
    api._attention_gate_hold_window_s = 0.5
    api._evaluate_attention_gate_admission(
        transcript="Theo Prime, status.",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )

    admitted, reason = api._evaluate_attention_gate_admission(
        transcript="And what else?",
        turn_id="turn-1",
        input_event_key="item-3",
        now=100.51,
    )

    assert admitted is False
    assert reason == "attention_gate_closed"


def test_configured_name_change_updates_direct_address_detection() -> None:
    api = _make_attention_api(assistant_name="Nova")

    admitted_old, reason_old = api._evaluate_attention_gate_admission(
        transcript="Theo, can you help?",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )
    admitted_new, reason_new = api._evaluate_attention_gate_admission(
        transcript="Nova, can you help?",
        turn_id="turn-1",
        input_event_key="item-2",
        now=100.5,
    )

    assert admitted_old is False
    assert reason_old == "attention_gate_closed"
    assert admitted_new is True
    assert reason_new == "direct_address"


def test_response_create_backstop_blocks_when_attention_gate_closed() -> None:
    api = types.SimpleNamespace(
        _should_suppress_nonessential_runtime_emission_during_followthrough=lambda **_kwargs: (False, "none"),
    )
    runtime = ResponseCreateRuntime(api)

    decision, lifecycle_decision = runtime._decide_response_create_action_with_lifecycle(
        prepared_snapshot=_snapshot(attention_gate_closed=True),
    )

    assert lifecycle_decision is None
    assert decision.action is ResponseCreateOutcomeAction.BLOCK
    assert decision.reason_code == "attention_gate_closed"
