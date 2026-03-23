"""Focused tests for the continuity bookkeeping seam."""

from __future__ import annotations

import sys
import types

sys.modules.setdefault("audioop", types.ModuleType("audioop"))

from ai.continuity import ContinuityBrief, ContinuityLedger
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
