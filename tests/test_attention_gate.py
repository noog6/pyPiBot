from __future__ import annotations

import os
import sys
import types
from unittest.mock import patch

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.realtime.response_create_runtime import (
    ResponseCreateOutcomeAction,
    ResponseCreatePreparedSnapshot,
    ResponseCreateRuntime,
)
from ai.realtime_api import AttentionState, RealtimeAPI


def _make_attention_api(*, assistant_name: str = "Theo Prime") -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._assistant_name = assistant_name
    api._stop_words = ["stop", "abort"]
    api._attention_gate_hold_window_s = 2.0
    api._attention_gate_hold_until_monotonic = None
    api._attention_gate_blocked_listening_cooldown_s = 1.0
    api._attention_gate_blocked_listening_suppress_until_ts = 0.0
    api._log_user_transcript_redact_enabled = True
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


def _render_logged_line(info_mock) -> str:
    msg, *values = info_mock.call_args.args
    return msg % tuple(values)


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
    assert api._attention_state_snapshot().state is AttentionState.DIRECT_ADDRESS_ACTIVE


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
    assert api._attention_state_snapshot().state is AttentionState.HOLD_ACTIVE


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
    assert api._attention_state_snapshot().state is AttentionState.CLOSED


def test_repeated_hold_admissions_do_not_extend_hold_window_forever() -> None:
    api = _make_attention_api()
    api._attention_gate_hold_window_s = 0.5
    api._evaluate_attention_gate_admission(
        transcript="Theo Prime, status.",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )

    admitted_1, reason_1 = api._evaluate_attention_gate_admission(
        transcript="And tomorrow?",
        turn_id="turn-1",
        input_event_key="item-2",
        now=100.2,
    )
    admitted_2, reason_2 = api._evaluate_attention_gate_admission(
        transcript="And later?",
        turn_id="turn-1",
        input_event_key="item-3",
        now=100.4,
    )
    admitted_3, reason_3 = api._evaluate_attention_gate_admission(
        transcript="Still there?",
        turn_id="turn-1",
        input_event_key="item-4",
        now=100.51,
    )

    assert admitted_1 is True
    assert reason_1 == "active_hold_window"
    assert admitted_2 is True
    assert reason_2 == "active_hold_window"
    assert admitted_3 is False
    assert reason_3 == "attention_gate_closed"
    assert api._attention_gate_hold_until_monotonic is None


def test_stop_abort_exception_bypasses_attention_gate_when_closed() -> None:
    api = _make_attention_api()
    admitted, reason = api._evaluate_attention_gate_exception(
        transcript="please stop right now",
        confirmation_active=False,
    )

    assert admitted is True
    assert reason == "stop_abort_safety_interrupt"
    assert api._attention_state_snapshot().state is AttentionState.BYPASS_ACTIVE


def test_confirmation_reply_bypasses_attention_gate_when_awaiting_confirmation() -> None:
    api = _make_attention_api()
    admitted, reason = api._evaluate_attention_gate_exception(
        transcript="yes",
        confirmation_active=True,
    )

    assert admitted is True
    assert reason == "awaiting_confirmation"
    assert api._attention_state_snapshot().state is AttentionState.BYPASS_ACTIVE


def test_blocked_recovery_cooldown_state_expires_to_closed() -> None:
    api = _make_attention_api()
    api._attention_gate_blocked_listening_cooldown_s = 0.5

    with patch("ai.realtime_api.time.monotonic", return_value=10.0):
        api._mark_attention_gate_closed_recovery(reason="attention_gate_closed")

    assert api._attention_state_snapshot().state is AttentionState.BLOCKED_RECOVERY_COOLDOWN
    assert api._attention_gate_blocked_listening_cue_suppressed(now=10.25) is True
    assert api._attention_gate_blocked_listening_cue_suppressed(now=10.51) is False
    assert api._attention_state_snapshot().state is AttentionState.CLOSED


def test_empty_transcript_recovery_marks_blocked_cooldown_state() -> None:
    api = _make_attention_api()
    api._attention_gate_blocked_listening_cooldown_s = 0.25

    with patch("ai.realtime_api.time.monotonic", return_value=22.0):
        api._mark_blocked_listening_recovery(reason="empty_transcript")

    snapshot = api._attention_state_snapshot()
    assert snapshot.state is AttentionState.BLOCKED_RECOVERY_COOLDOWN
    assert snapshot.reason == "empty_transcript"
    assert snapshot.suppress_until_monotonic == 22.25


def test_hold_expiry_uses_formal_state_not_legacy_timestamp() -> None:
    api = _make_attention_api()
    api._attention_gate_hold_until_monotonic = 999.0
    api._set_attention_state_closed(reason="test_closed", now=50.0)

    assert api._attention_gate_hold_active(now=100.0) is False


def test_blocked_cooldown_expiry_uses_formal_state_not_legacy_timestamp() -> None:
    api = _make_attention_api()
    api._attention_gate_blocked_listening_suppress_until_ts = 999.0
    api._set_attention_state_closed(reason="test_closed", now=10.0)

    assert api._attention_gate_blocked_listening_cue_suppressed(now=100.0) is False


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


def test_secondary_direct_address_terms_admit_when_request_shaped() -> None:
    api = _make_attention_api()

    admitted_robot, reason_robot = api._evaluate_attention_gate_admission(
        transcript="Robot, look over there.",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )
    admitted_droid, reason_droid = api._evaluate_attention_gate_admission(
        transcript="Droid, what do you see?",
        turn_id="turn-1",
        input_event_key="item-2",
        now=110.0,
    )
    admitted_android, reason_android = api._evaluate_attention_gate_admission(
        transcript="Android, can you check that?",
        turn_id="turn-1",
        input_event_key="item-3",
        now=120.0,
    )
    admitted_modal, reason_modal = api._evaluate_attention_gate_admission(
        transcript="Can the droid look left?",
        turn_id="turn-1",
        input_event_key="item-4",
        now=130.0,
    )

    assert admitted_robot is True
    assert reason_robot == "direct_address"
    assert admitted_droid is True
    assert reason_droid == "direct_address"
    assert admitted_android is True
    assert reason_android == "direct_address"
    assert admitted_modal is True
    assert reason_modal == "direct_address"


def test_secondary_terms_without_direct_address_evidence_remain_blocked() -> None:
    api = _make_attention_api()

    admitted_robot, reason_robot = api._evaluate_attention_gate_admission(
        transcript="That robot on the desk is cute.",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )
    admitted_droid, reason_droid = api._evaluate_attention_gate_admission(
        transcript="The droid moved earlier.",
        turn_id="turn-1",
        input_event_key="item-2",
        now=110.0,
    )
    admitted_android, reason_android = api._evaluate_attention_gate_admission(
        transcript="This android looks expensive.",
        turn_id="turn-1",
        input_event_key="item-3",
        now=120.0,
    )

    assert admitted_robot is False
    assert reason_robot == "attention_gate_closed"
    assert admitted_droid is False
    assert reason_droid == "attention_gate_closed"
    assert admitted_android is False
    assert reason_android == "attention_gate_closed"


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


def test_clear_direct_address_question_classifier_is_name_and_question_bound() -> None:
    api = _make_attention_api()

    assert api._is_clear_direct_address_question("Theo Prime, what time is it") is True
    assert api._is_clear_direct_address_question("what time is it") is False
    assert api._is_clear_direct_address_question("Theo Prime, I like this") is False


def test_attention_decision_log_direct_address_admitted() -> None:
    api = _make_attention_api()
    admitted, reason = api._evaluate_attention_gate_admission(
        transcript="Theo Prime, what's the status?",
        turn_id="turn-1",
        input_event_key="item-1",
        now=100.0,
    )
    assert admitted is True

    with patch("ai.realtime_api.logger.info") as info_mock:
        api._log_transcript_attention_decision(
            transcript="Theo Prime, what's the status?",
            attention="admitted",
            reason=reason,
            turn_id="turn-1",
            input_event_key="item-1",
        )

    assert info_mock.call_count == 1
    line = _render_logged_line(info_mock)
    assert "transcript_attention_decision" in line
    assert "attention=admitted" in line
    assert "reason=direct_address" in line
    assert "current_state=direct_address_active" in line
    assert 'heard_transcript="Theo Prime, what\'s the status?"' in line


def test_attention_decision_log_non_addressed_blocked() -> None:
    api = _make_attention_api()
    admitted, reason = api._evaluate_attention_gate_admission(
        transcript="PJs, are you comfy?",
        turn_id="turn-1",
        input_event_key="item-2",
        now=100.0,
    )
    assert admitted is False

    with patch("ai.realtime_api.logger.info") as info_mock:
        api._log_transcript_attention_decision(
            transcript="PJs, are you comfy?",
            attention="blocked",
            reason=reason,
            turn_id="turn-1",
            input_event_key="item-2",
        )

    assert info_mock.call_count == 1
    line = _render_logged_line(info_mock)
    assert "attention=blocked" in line
    assert "reason=attention_gate_closed" in line
    assert "current_state=closed" in line
    assert 'heard_transcript="PJs, are you comfy?"' in line


def test_attention_decision_log_confirmation_bypass() -> None:
    api = _make_attention_api()
    admitted, reason = api._evaluate_attention_gate_exception(
        transcript="yes",
        confirmation_active=True,
    )
    assert admitted is True

    with patch("ai.realtime_api.logger.info") as info_mock:
        api._log_transcript_attention_decision(
            transcript="yes",
            attention="bypass",
            reason=reason,
            turn_id="turn-2",
            input_event_key="item-3",
        )

    assert info_mock.call_count == 1
    line = _render_logged_line(info_mock)
    assert "attention=bypass" in line
    assert "reason=awaiting_confirmation" in line
    assert "current_state=bypass_active" in line
    assert 'heard_transcript="yes"' in line
