"""Focused tests for the continuity bookkeeping seam."""

from __future__ import annotations

import logging
import sys
import types

import pytest

sys.modules.setdefault("audioop", types.ModuleType("audioop"))

from ai.continuity import ContinuityBrief, ContinuityItem, ContinuityLedger
from ai.realtime_api import RealtimeAPI


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
    assert "current_items=(" in message
    assert "recently_closed_items=()" in message
    assert "while preserving…" in message
    assert long_summary not in message

    inspection = ledger.inspect_state(run_id="run-42", turn_id="turn-7", event_type="transcript_final")
    assert inspection["stance"] == "assisting_query"
    assert inspection["stance_detail"] == "read_query_detected"
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
    summary = api.get_continuity_debug_summary("run-7", "turn-11")
    after_items = dict(api._continuity_ledger._items)

    assert summary.startswith("stance=idle | detail=current=commitment:active | current=[")
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
    assert "recently_closed=[" in summary


def test_realtime_api_continuity_debug_summary_exposes_assisting_query_stance() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._continuity_ledger = ContinuityLedger()

    api._apply_continuity_event(
        "transcript_final",
        text="Do you know what your battery voltage is at?",
        source="input_audio_transcription",
    )

    summary = api.get_continuity_debug_summary("run-9", "turn-13")

    assert "stance=assisting_query |" in summary
    assert "detail=read_query_detected |" in summary
    assert "current=[ongoing/active:Do you know what your battery voltage is at?" in summary
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

    brief = api.build_continuity_brief("run-2", "turn-4", "authority_boundary")
    assert brief.stance == "awaiting_tool"
    assert api._response_in_flight is True
    assert api.function_call == {"name": "gesture_look_center"}
    assert api._active_response_origin == "server_auto"
