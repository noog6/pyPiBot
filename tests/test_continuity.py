"""Focused tests for the continuity bookkeeping seam."""

from __future__ import annotations

import logging
import sys
import types

import pytest

sys.modules.setdefault("audioop", types.ModuleType("audioop"))

from ai.continuity import (
    ContinuityBrief,
    ContinuityItem,
    ContinuityLedger,
    ContinuityTurnSettlement,
)
from ai.realtime_api import RealtimeAPI




def test_simple_single_step_request_does_not_create_compound_state() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Look at the window.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-single", "turn-1", "single_step")
    assert brief.commitments[0].summary == "Look at the window."
    assert brief.compound_request is None




def test_compound_parser_keeps_action_plus_report_as_small_chain() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Look right and tell me what you see.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-chain", "turn-1", "small_chain")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "report"]
    assert [step.summary for step in brief.compound_request.steps] == [
        "Look right.",
        "tell me what you see.",
    ]
    assert [step.implicit_observation_required for step in brief.compound_request.steps] == [False, True]
    assert brief.compound_request.steps[1].requires_perception is True
    assert brief.compound_request.steps[1].perception_mode == "visual"
    assert brief.compound_request.steps[1].report_intent == "describe"


def test_compound_parser_captures_go_back_to_center_followed_by_visual_followup() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Hey Theo, can you go back to center and then tell me what you see me holding in my hand?",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-center-chain", "turn-1", "center_then_report")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "report"]
    assert [step.summary for step in brief.compound_request.steps] == [
        "can you go back to center.",
        "tell me what you see me holding in my hand?",
    ]
    assert brief.compound_request.steps[1].requires_perception is True
    assert brief.compound_request.steps[1].perception_mode == "visual"
    assert brief.compound_request.steps[1].report_intent == "identify"


def test_compound_parser_ignores_plain_declarative_description() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="It's a blue cordless drill, and it's got a little light on the front.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-declarative", "turn-1", "description_only")
    assert brief.compound_request is None


def test_compound_visual_report_marks_implicit_observation_in_diagnostics() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._continuity_ledger = ContinuityLedger()

    api._apply_continuity_event(
        "transcript_final",
        text="Look right and let me know what you see.",
        source="input_audio_transcription",
    )

    brief = api.get_continuity_brief("run-visual", "turn-1", reason="visual_report")
    diagnostics = api.get_continuity_diagnostics("run-visual", "turn-1", reason="visual_report")
    summary = api.get_continuity_debug_summary("run-visual", "turn-1", reason="visual_report")

    assert brief.compound_request is not None
    assert len(brief.compound_request.steps) == 2
    assert brief.compound_request.steps[1].kind == "report"
    assert brief.compound_request.steps[1].implicit_observation_required is True
    assert brief.compound_request.steps[1].requires_perception is True
    assert brief.compound_request.steps[1].perception_mode == "visual"
    assert brief.compound_request.steps[1].report_intent == "describe"
    assert diagnostics["compound"]["implicit_observation_substeps"] == ("let me know what you see.",)
    assert diagnostics["compound"]["perception_required_substeps"] == ("let me know what you see.",)
    assert diagnostics["compound"]["report_traits"] == (("step_2", "describe", "visual", True),)
    assert "implicit_obs=1" in summary
    assert "perception_required=1" in summary


def test_compound_parser_preserves_order_for_comma_then_chain() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics, then report done.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-order", "turn-1", "comma_then_chain")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "diagnostics", "report"]
    assert [step.summary for step in brief.compound_request.steps] == [
        "Look right.",
        "check diagnostics.",
        "report done.",
    ]
    assert [step.implicit_observation_required for step in brief.compound_request.steps] == [False, False, False]
    assert brief.compound_request.steps[2].requires_perception is False
    assert brief.compound_request.steps[2].report_intent == "status"


def test_followup_only_status_phrase_stays_non_perception_report() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Tell me when you're centered.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-status", "turn-1", "followup_status")
    assert brief.compound_request is None
    assert brief.stance == "idle"
    assert brief.ongoing[0].summary == "Tell me when you're centered."


def test_compound_parser_does_not_oversplit_ordinary_phrasing() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Look at the window and the door.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-nosplit", "turn-1", "ordinary_phrase")
    assert brief.compound_request is None
    assert brief.commitments[0].summary == "Look at the window and the door."
def test_compound_request_creates_bounded_ordered_substeps() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text=(
            "Look right, check diagnostics silently, look left, check diagnostics silently, "
            "center, check diagnostics, then report done."
        ),
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-compound", "turn-1", "compound_steps")

    assert brief.compound_request is not None
    compound = brief.compound_request
    assert compound.summary.startswith("Look right")
    assert len(compound.steps) == 7
    assert [step.kind for step in compound.steps] == [
        "gesture",
        "diagnostics",
        "gesture",
        "diagnostics",
        "gesture",
        "diagnostics",
        "report",
    ]
    assert compound.active_step_index == 0
    assert compound.completed_step_ids == ()
    assert compound.steps[0].status == "active"
    assert compound.steps[1].status == "pending"
    assert compound.next_pending_step_id == "step_2"
    assert compound.final_followup_pending is True


def test_compound_tool_progress_updates_matching_step_only() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics silently, then report done.",
        source="input_audio_transcription",
    )

    ledger.update_from_event(
        "tool_call_started",
        tool_name="gesture_look_right",
        call_id="call-1",
        commitment_summary="Look right, check diagnostics silently, then report done.",
    )
    during_call = ledger.build_brief("run-progress", "turn-1", "tool_start")
    assert during_call.compound_request is not None
    assert [step.status for step in during_call.compound_request.steps] == ["active", "pending", "pending"]

    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-1",
    )

    after_result = ledger.build_brief("run-progress", "turn-1", "tool_result")
    assert after_result.compound_request is not None
    assert [step.status for step in after_result.compound_request.steps] == ["completed", "active", "pending"]
    assert after_result.compound_request.recent_completed_step_id == "step_1"
    assert after_result.compound_request.next_pending_step_id == "step_3"


def test_compound_final_followup_stays_pending_until_explicitly_closed() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics, then report done.",
        source="input_audio_transcription",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-2",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="read_runtime_diagnostics",
        call_id="call-3",
    )

    before_followup = ledger.build_brief("run-followup", "turn-1", "before_followup_close")
    assert before_followup.compound_request is not None
    assert before_followup.compound_request.final_followup_pending is True
    assert before_followup.compound_request.steps[2].status == "pending"

    ledger.update_from_event("response_done", close_unresolved="true")

    after_followup = ledger.build_brief("run-followup", "turn-1", "after_followup_close")
    assert after_followup.compound_request is None


def test_compound_observability_does_not_mutate_runtime_authority_state() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._continuity_ledger = ContinuityLedger()
    api._response_in_flight = True
    api.function_call = {"name": "gesture_look_right"}
    api._active_response_origin = "server_auto"

    api._apply_continuity_event(
        "transcript_final",
        text="Look right, check diagnostics, then report done.",
        source="input_audio_transcription",
    )

    before_items = dict(api._continuity_ledger._items)
    before_compound = api._continuity_ledger._compound_state
    brief = api.get_continuity_brief("run-observe", "turn-1", reason="observational_only")
    diagnostics = api.get_continuity_diagnostics("run-observe", "turn-1", reason="observational_only")
    after_items = dict(api._continuity_ledger._items)
    after_compound = api._continuity_ledger._compound_state

    assert brief.compound_request is not None
    assert diagnostics["compound"]["substeps_total"] == 3
    assert diagnostics["compound"]["final_followup_pending"] is True
    assert before_items == after_items
    assert before_compound == after_compound
    assert api._response_in_flight is True
    assert api.function_call == {"name": "gesture_look_right"}
    assert api._active_response_origin == "server_auto"
def test_transcript_final_creates_simple_ongoing_item() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="What do you see on the desk?",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-1", "turn-1", "unit_test")
    assert brief.stance == "assisting_observation"
    assert brief.ongoing[0].summary == "What do you see on the desk?"
    assert brief.ongoing[0].kind == "ongoing"


def test_action_request_creates_commitment() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Look at the window and tell me what you see.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-1", "turn-1", "action")
    assert brief.stance == "assisting_execution"
    assert brief.commitments[0].summary == "Look at the window and tell me what you see."
    assert brief.commitments[0].detail == "origin=user_request"
    assert brief.unresolved[0].summary == "tell me what you see."
    assert brief.unresolved[0].detail == "opened_by=transcript_final"
    assert brief.current[0].kind == "commitment"
    assert brief.stance_detail == "origin=user_request"


def test_read_query_request_uses_non_idle_stance() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Do you know what your battery voltage is at?",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-1", "turn-1", "query")
    assert brief.stance == "assisting_query"
    assert brief.stance_detail == "read_query_detected"
    assert brief.commitments == ()
    assert brief.ongoing[0].summary == "Do you know what your battery voltage is at?"


def test_environment_read_queries_stay_in_assisting_query() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Can you check the air pressure and temperature?",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-1", "turn-2", "environment_query")
    assert brief.stance == "assisting_query"
    assert brief.stance_detail == "read_query_detected"


def test_tool_named_read_environment_stays_in_assisting_query() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Please run read_environment and report the air pressure.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-1", "turn-2b", "tool_named_query")
    assert brief.stance == "assisting_query"
    assert brief.stance_detail == "read_query_detected"


def test_status_check_requests_do_not_collapse_into_assisting_query() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Can you still hear me?",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-1", "turn-3", "status_check")
    assert brief.stance == "assisting_observation"
    assert brief.stance_detail == ""


def test_transcript_final_uses_tool_waiting_stance_only_when_needed() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Please check the camera feed.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-1", "turn-1", "tool_wait")
    assert brief.stance == "awaiting_tool"
    assert brief.commitments == ()


def test_tool_start_adds_blocker() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "tool_call_started",
        tool_name="gesture_look_center",
        call_id="call-1",
        commitment_summary="Look at the center of the room.",
    )

    brief = ledger.build_brief("run-1", "turn-1", "tool_started")
    assert brief.stance == "awaiting_tool"
    assert brief.blockers[0].summary == "Waiting for tool result: gesture_look_center"
    assert brief.blockers[0].detail == "tool=gesture_look_center call_id=call-1"
    assert brief.commitments[0].summary == "Look at the center of the room."
    assert brief.current[0].kind == "blocker"
    assert brief.stance_detail == "tool=gesture_look_center call_id=call-1"


def test_tool_result_resolves_blocker() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "tool_call_started",
        tool_name="gesture_look_center",
        call_id="call-1",
        commitment_summary="Look at the center of the room.",
    )

    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_center",
        call_id="call-1",
    )

    brief = ledger.build_brief("run-1", "turn-1", "tool_result")
    assert brief.blockers == ()
    assert brief.recently_closed[0].summary == "Waiting for tool result: gesture_look_center"
    assert brief.recently_closed[0].detail == "tool=gesture_look_center call_id=call-1"
    assert brief.commitments[0].status == "active"


def test_response_done_moves_completed_item_to_recently_closed_when_explicitly_closed() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Center on the whiteboard.",
        source="input_audio_transcription",
    )

    ledger.update_from_event(
        "response_done",
        close_ongoing="true",
        close_commitment="true",
    )

    brief = ledger.build_brief("run-1", "turn-1", "response_done")
    assert brief.commitments == ()
    assert brief.recently_closed[0].summary == "Center on the whiteboard."


def test_response_done_keeps_followup_open_without_explicit_close_signal() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look at the door and tell me whether it is open.",
        source="input_audio_transcription",
    )

    ledger.update_from_event("response_done")

    brief = ledger.build_brief("run-1", "turn-1", "response_done")
    assert brief.stance == "awaiting_user"
    assert brief.commitments[0].summary == "Look at the door and tell me whether it is open."
    assert brief.unresolved[0].summary == "tell me whether it is open."


def test_build_turn_settlement_reports_settled_without_open_items() -> None:
    ledger = ContinuityLedger()

    settlement = ledger.build_turn_settlement(ledger.build_brief("run-0", "turn-0", "settled_case"))

    assert isinstance(settlement, ContinuityTurnSettlement)
    assert settlement.settlement_state == "settled"
    assert settlement.settlement_detail == "no_open_continuity_items"
    assert settlement.has_current_items is False
    assert settlement.has_commitments is False
    assert settlement.has_unresolved is False
    assert settlement.has_blockers is False
    assert settlement.has_recently_closed is False


def test_build_turn_settlement_reports_awaiting_tool_when_blocker_present() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "tool_call_started",
        tool_name="gesture_look_center",
        call_id="call-1",
        commitment_summary="Look at the center of the room.",
    )

    settlement = ledger.build_turn_settlement(ledger.build_brief("run-1", "turn-1", "awaiting_tool_case"))

    assert settlement.settlement_state == "awaiting_tool"
    assert settlement.settlement_detail == "tool=gesture_look_center call_id=call-1"
    assert settlement.has_current_items is True
    assert settlement.has_commitments is True
    assert settlement.has_unresolved is False
    assert settlement.has_blockers is True
    assert settlement.has_recently_closed is False


def test_build_turn_settlement_reports_unresolved_followup_when_commitment_and_question_open() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look at the door and tell me whether it is open.",
        source="input_audio_transcription",
    )

    settlement = ledger.build_turn_settlement(ledger.build_brief("run-2", "turn-2", "followthrough_case"))

    assert settlement.settlement_state == "unresolved_followup"
    assert settlement.settlement_detail == "opened_by=transcript_final"
    assert settlement.has_current_items is True
    assert settlement.has_commitments is True
    assert settlement.has_unresolved is True
    assert settlement.has_blockers is False
    assert settlement.has_recently_closed is False


def test_build_turn_settlement_reports_recently_closed_only_when_open_items_are_closed() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Center on the whiteboard.",
        source="input_audio_transcription",
    )
    ledger.update_from_event(
        "response_done",
        close_ongoing="true",
        close_commitment="true",
    )

    settlement = ledger.build_turn_settlement(
        ledger.build_brief("run-3", "turn-3", "recently_closed_case")
    )

    assert settlement.settlement_state == "recently_closed_only"
    assert settlement.settlement_detail == "origin=user_request"
    assert settlement.has_current_items is False
    assert settlement.has_commitments is False
    assert settlement.has_unresolved is False
    assert settlement.has_blockers is False
    assert settlement.has_recently_closed is True


def test_build_continuity_brief_returns_compact_structured_output() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look at the door and tell me whether it is open?",
        source="input_audio_transcription",
    )
    ledger.update_from_event(
        "tool_call_started",
        tool_name="gesture_look_center",
        call_id="call-1",
        commitment_summary="Look at the door and tell me whether it is open?",
    )

    brief = ledger.build_brief("run-7", "turn-9", "projection")
    assert isinstance(brief, ContinuityBrief)
    assert brief.run_id == "run-7"
    assert brief.turn_id == "turn-9"
    assert brief.generated_reason == "projection"
    assert brief.current[0].kind == "blocker"
    assert len(brief.ongoing) <= 3
    assert len(brief.blockers) <= 3
    assert len(brief.current) <= 3


def test_inspect_state_reuses_brief_projection_shape() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "tool_call_started",
        tool_name="gesture_look_center",
        call_id="call-9",
        commitment_summary="Look at the center of the room.",
    )

    brief = ledger.build_brief("run-5", "turn-6", "shared_projection")
    inspection = ledger.inspect_state(run_id="run-5", turn_id="turn-6", event_type="inspection")
    settlement = ledger.build_turn_settlement(brief)

    assert inspection["stance"] == brief.stance
    assert inspection["stance_detail"] == brief.stance_detail
    assert inspection["current"] == len(brief.current)
    assert inspection["ongoing"] == len(brief.ongoing)
    assert inspection["commitments"] == len(brief.commitments)
    assert inspection["blockers"] == len(brief.blockers)
    assert inspection["settlement_state"] == settlement.settlement_state
    assert inspection["settlement_detail"] == settlement.settlement_detail


def test_continuity_inspection_logging_is_bounded_and_deterministic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ledger = ContinuityLedger()
    long_summary = (
        "Tell me your current battery voltage and air pressure while preserving "
        "a compact inspection payload for operators."
    )

    with caplog.at_level(logging.INFO, logger="ai.continuity"):
        ledger.update_from_event(
            "transcript_final",
            run_id="run-42",
            turn_id="turn-7",
            text=long_summary,
            source="input_audio_transcription",
        )

    summaries = [
        record for record in caplog.records if record.message.startswith("continuity_inspection_summary")
    ]
    assert len(summaries) == 1
    message = summaries[0].message
    assert "run_id=run-42" in message
    assert "turn_id=turn-7" in message
    assert "event_type=transcript_final" in message
    assert "stance=assisting_query" in message
    assert "stance_detail=read_query_detected" in message
    assert "settlement=active_items_only" in message
    assert "settlement_detail=origin=user_transcript" in message
    assert "current_items=(" in message
    assert "recently_closed_items=()" in message
    assert "while preserving…" in message
    assert long_summary not in message

    inspection = ledger.inspect_state(run_id="run-42", turn_id="turn-7", event_type="transcript_final")
    assert inspection["stance"] == "assisting_query"
    assert inspection["stance_detail"] == "read_query_detected"
    assert inspection["settlement_state"] == "active_items_only"
    assert inspection["settlement_detail"] == "origin=user_transcript"
    assert inspection["current"] == 1
    assert inspection["current_items"] == (
        {
            "kind": "ongoing",
            "status": "active",
            "id": "request:current",
            "summary": "Tell me your current battery voltage and air pressure while preserving…",
            "detail": "origin=user_transcript",
        },
    )


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_identical_repeated_build_brief_calls_do_not_log_before_cooldown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    ledger = ContinuityLedger(time_source=clock, brief_log_cooldown_s=10.0)

    with caplog.at_level(logging.INFO, logger="ai.continuity"):
        ledger.build_brief("run-1", "turn-1", "session_health")
        clock.advance(3.0)
        ledger.build_brief("run-1", "turn-1", "session_health")
        clock.advance(3.0)
        ledger.build_brief("run-1", "turn-1", "session_health")

    brief_logs = [
        record for record in caplog.records if record.message.startswith("continuity_brief_built")
    ]
    assert len(brief_logs) == 1
    assert "reason=session_health" in brief_logs[0].message
    assert "fingerprint_changed=True" in brief_logs[0].message
    assert "reminder=False" in brief_logs[0].message
    assert "cooldown_elapsed_s=0.0" in brief_logs[0].message


def test_materially_changed_brief_logs_immediately_even_before_cooldown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    ledger = ContinuityLedger(time_source=clock, brief_log_cooldown_s=10.0)

    with caplog.at_level(logging.INFO, logger="ai.continuity"):
        ledger.build_brief("run-1", "turn-1", "session_health")
        clock.advance(1.0)
        ledger.update_from_event(
            "transcript_final",
            text="Hey Theo, can you run a diagnostic?",
            source="input_audio_transcription",
        )
        ledger.build_brief("run-1", "turn-1", "session_health")

    brief_logs = [
        record for record in caplog.records if record.message.startswith("continuity_brief_built")
    ]
    assert len(brief_logs) == 2
    assert "fingerprint_changed=True" in brief_logs[0].message
    assert "fingerprint_changed=True" in brief_logs[1].message
    assert "ongoing=1" in brief_logs[1].message
    assert "reason=session_health" in brief_logs[1].message
    assert "cooldown_elapsed_s=1.0" in brief_logs[1].message


def test_unchanged_brief_logs_again_after_cooldown_expiry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    ledger = ContinuityLedger(time_source=clock, brief_log_cooldown_s=10.0)

    with caplog.at_level(logging.INFO, logger="ai.continuity"):
        ledger.build_brief("run-1", "turn-1", "session_health")
        clock.advance(10.0)
        ledger.build_brief("run-1", "turn-1", "session_health")

    brief_logs = [
        record for record in caplog.records if record.message.startswith("continuity_brief_built")
    ]
    assert len(brief_logs) == 2
    assert "fingerprint_changed=True" in brief_logs[0].message
    assert "fingerprint_changed=False" in brief_logs[1].message
    assert "reminder=True" in brief_logs[1].message
    assert "cooldown_elapsed_s=10.0" in brief_logs[1].message


def test_session_health_style_repeated_build_brief_calls_are_suppressed_when_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    ledger = ContinuityLedger(time_source=clock, brief_log_cooldown_s=10.0)

    ledger.update_from_event(
        "transcript_final",
        text="Hey Theo, can you run a diagnostic?",
        source="input_audio_transcription",
    )

    with caplog.at_level(logging.INFO, logger="ai.continuity"):
        for _ in range(5):
            ledger.build_brief("run-7", "turn-2", "session_health")
            clock.advance(1.0)

    brief_logs = [
        record for record in caplog.records if record.message.startswith("continuity_brief_built")
    ]
    assert len(brief_logs) == 1
    assert "ongoing=1" in brief_logs[0].message
    assert "reason=session_health" in brief_logs[0].message


def test_build_brief_logging_cache_is_observational_only() -> None:
    clock = _FakeClock()
    ledger = ContinuityLedger(time_source=clock, brief_log_cooldown_s=10.0)
    ledger.update_from_event(
        "tool_call_started",
        tool_name="read_runtime_diagnostics",
        call_id="call-1",
        commitment_summary="Run a diagnostic and report back.",
    )

    before_items = dict(ledger._items)

    ledger.build_brief("run-1", "turn-1", "session_health")
    clock.advance(1.0)
    ledger.build_brief("run-1", "turn-1", "session_health")

    after_items = dict(ledger._items)
    assert before_items == after_items
    assert ledger._items["blocker:tool:call-1"].detail == "tool=read_runtime_diagnostics call_id=call-1"


def test_realtime_api_get_continuity_brief_is_observational() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._continuity_ledger = ContinuityLedger()
    api._response_in_flight = False

    api._apply_continuity_event(
        "transcript_final",
        text="Look at the whiteboard and tell me whether it changed.",
        source="input_audio_transcription",
    )

    brief = api.get_continuity_brief("run-3", "turn-8", reason="inspection")
    assert isinstance(brief, ContinuityBrief)
    assert brief.run_id == "run-3"
    assert brief.turn_id == "turn-8"
    assert brief.generated_reason == "inspection"
    assert brief.current[0].kind == "commitment"
    assert brief.unresolved[0].detail == "opened_by=transcript_final"
    assert api._response_in_flight is False


def test_realtime_api_continuity_debug_summary_is_read_only_and_deterministic() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._continuity_ledger = ContinuityLedger()
    api._response_in_flight = False
    api._active_response_origin = "tool_output"

    api._apply_continuity_event(
        "transcript_final",
        text=(
            "Look at the whiteboard and tell me whether the planning notes "
            "mention a delivery date after the next sprint review."
        ),
        source="input_audio_transcription",
    )
    api._apply_continuity_event(
        "tool_call_started",
        tool_name="gesture_look_center",
        call_id="call-22",
        commitment_summary=(
            "Look at the whiteboard and tell me whether the planning notes "
            "mention a delivery date after the next sprint review."
        ),
    )
    api._apply_continuity_event(
        "tool_result_received",
        tool_name="gesture_look_center",
        call_id="call-22",
    )

    before_items = dict(api._continuity_ledger._items)
    settlement = api.get_continuity_turn_settlement("run-7", "turn-11")
    summary = api.get_continuity_debug_summary("run-7", "turn-11")
    after_items = dict(api._continuity_ledger._items)

    assert settlement.settlement_state == "unresolved_followup"
    assert settlement.settlement_detail == "opened_by=transcript_final"
    assert settlement.has_current_items is True
    assert settlement.has_commitments is True
    assert settlement.has_unresolved is True
    assert settlement.has_blockers is False
    assert settlement.has_recently_closed is True
    assert summary.startswith("stance=idle | detail=current=commitment:active | settlement=unresolved_followup | settlement_detail=opened_by=transcript_final | compound=[")
    assert "followup_pending=True" in summary
    assert "current=[commitment/active:" in summary
    assert "[origin=tool_followthrough tool=…]" in summary
    assert "unresolved/pending:" in summary
    assert "[opened_by=transcript_final]" in summary
    assert "ongoing/active:" in summary
    assert "[origin=user_transcript]" in summary
    assert " | recently_closed=[recently_closed/resolved:Waiting for tool result: gesture_look_center " in summary
    assert "[tool=gesture_look_center call_i…]" in summary
    assert "next sprint review." not in summary
    assert before_items == after_items
    assert api._response_in_flight is False
    assert api._active_response_origin == "tool_output"


def test_realtime_api_continuity_debug_summary_bounds_current_and_closed_items() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._continuity_ledger = ContinuityLedger()

    for idx in range(5):
        api._continuity_ledger._set_item(
            ContinuityItem(
                id=f"constraint:{idx}",
                kind="constraint",
                summary=f"Constraint item number {idx} with extra detail for trimming checks.",
                status="active",
                priority="medium",
                source="unit_test",
                detail=f"detail={idx}",
                expires_after_turns=3,
            )
        )
        api._continuity_ledger._set_item(
            ContinuityItem(
                id=f"closed:{idx}",
                kind="recently_closed",
                summary=f"Closed item number {idx} with extra detail for trimming checks.",
                status="resolved",
                priority="medium",
                source="unit_test",
                detail=f"closed_detail={idx}",
                expires_after_turns=1,
            )
        )

    summary = api.get_continuity_debug_summary("run-8", "turn-12")

    assert "constraint:3" not in summary
    assert "constraint:4" not in summary
    assert "closed:3" not in summary
    assert "closed:4" not in summary
    assert summary.count(" | ") >= 5
    assert "stance=idle |" in summary
    assert "detail=current=constraint:active |" in summary
    assert "settlement=active_items_only |" in summary
    assert "settlement_detail=detail=0 |" in summary
    assert "recently_closed=[" in summary


def test_realtime_api_continuity_debug_summary_exposes_assisting_query_stance() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._continuity_ledger = ContinuityLedger()

    api._apply_continuity_event(
        "transcript_final",
        text="Do you know what your battery voltage is at?",
        source="input_audio_transcription",
    )

    settlement = api.get_continuity_turn_settlement("run-9", "turn-13")
    summary = api.get_continuity_debug_summary("run-9", "turn-13")

    assert settlement.settlement_state == "active_items_only"
    assert settlement.settlement_detail == "origin=user_transcript"
    assert "stance=assisting_query |" in summary
    assert "detail=read_query_detected |" in summary
    assert "settlement=active_items_only |" in summary
    assert "settlement_detail=origin=user_transcript |" in summary
    assert "current=[ongoing/active:Do you know what your battery voltage is at?" in summary
    assert "compound=[-]" in summary
    assert "recently_closed=[-]" in summary


def test_continuity_does_not_invent_authority_or_mutate_unrelated_runtime_state() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._continuity_ledger = ContinuityLedger()
    api._response_in_flight = True
    api.function_call = {"name": "gesture_look_center"}
    api._active_response_origin = "server_auto"

    api._apply_continuity_event(
        "tool_call_started",
        tool_name="gesture_look_center",
        call_id="call-1",
        commitment_summary="Look at the center of the room.",
    )

    before_items = dict(api._continuity_ledger._items)
    brief = api.build_continuity_brief("run-2", "turn-4", "authority_boundary")
    settlement = api.get_continuity_turn_settlement("run-2", "turn-4", reason="authority_boundary")
    after_items = dict(api._continuity_ledger._items)

    assert brief.stance == "awaiting_tool"
    assert settlement.settlement_state == "awaiting_tool"
    assert settlement.settlement_detail == "tool=gesture_look_center call_id=call-1"
    assert before_items == after_items
    assert api._response_in_flight is True
    assert api.function_call == {"name": "gesture_look_center"}
    assert api._active_response_origin == "server_auto"
