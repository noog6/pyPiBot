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
        text="Hey Theo, can you go back to center and then tell me what I'm holding in my hand?",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-center-chain", "turn-1", "center_then_report")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "report"]
    assert [step.summary for step in brief.compound_request.steps] == [
        "can you go back to center.",
        "tell me what I'm holding in my hand?",
    ]
    assert brief.compound_request.steps[1].requires_perception is True
    assert brief.compound_request.steps[1].perception_mode == "visual"
    assert brief.compound_request.steps[1].report_intent == "identify"


def test_compound_parser_classifies_verb_led_looking_sequence_as_gestures_then_report() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text=(
            "Hey Theo, can you start by looking left, and then look right, and then come back to center "
            "and tell me what I'm holding in my hand?"
        ),
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-verb-led", "turn-1", "verb_led_gesture_sequence")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "gesture", "gesture", "report"]
    assert [step.summary for step in brief.compound_request.steps] == [
        "can you start by looking left.",
        "look right.",
        "come back to center.",
        "tell me what I'm holding in my hand?",
    ]


def test_compound_parser_exact_phrase_family_preserves_expected_summaries() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="start by looking left, then look right, then come back to center",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-exact-phrase", "turn-1", "exact_phrase_family")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "gesture", "gesture"]
    assert [step.summary for step in brief.compound_request.steps] == [
        "start by looking left.",
        "look right.",
        "come back to center.",
    ]


def test_deterministic_followthrough_step_descriptor_tracks_active_directional_gesture() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="look left then look right and tell me what you see",
        source="input_audio_transcription",
        turn_id="turn_1",
    )

    descriptor = ledger.deterministic_followthrough_step()
    assert descriptor is not None
    assert descriptor.step_id == "step_1"
    assert descriptor.tool_name == "gesture_look_left"

    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_left",
        turn_id="turn_1",
    )
    descriptor = ledger.deterministic_followthrough_step()
    assert descriptor is not None
    assert descriptor.step_id == "step_2"
    assert descriptor.tool_name == "gesture_look_right"


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
    assert "report_traits=step_2:describe/visual/1" in summary


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


def test_compound_parser_classifies_diagnostic_reading_chain_between_gestures_and_report() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text=(
            "look right and then take a diagnostic reading and then come back to center "
            "and tell me what that diagnostic says"
        ),
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-diagnostic-reading-chain", "turn-1", "diagnostic_reading_chain")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "diagnostics", "gesture", "report"]
    assert [step.summary for step in brief.compound_request.steps] == [
        "look right.",
        "take a diagnostic reading.",
        "come back to center.",
        "tell me what that diagnostic says.",
    ]


@pytest.mark.parametrize(
    "clause",
    [
        "take a diagnostic reading",
        "get a diagnostic reading",
        "do a diagnostic check",
        "run diagnostics",
    ],
)
def test_compound_step_classifier_marks_diagnostic_phrase_family(clause: str) -> None:
    ledger = ContinuityLedger()
    assert ledger._classify_compound_step_kind(clause, unresolved_summary=None) == "diagnostics"


def test_compound_parser_status_report_remains_non_perception() -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text="Look right and report done.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-status-report", "turn-1", "report_done")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "report"]
    assert brief.compound_request.steps[1].requires_perception is False
    assert brief.compound_request.steps[1].report_intent == "status"
    assert brief.compound_request.steps[1].implicit_observation_required is False


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    [
        ("Look right and say hi.", "gesture"),
        ("Turn left and say hello.", "gesture"),
        ("Go back to center and say something.", "gesture"),
    ],
)
def test_compound_parser_bare_say_followups_do_not_become_report_steps(text: str, expected_kind: str) -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text=text,
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-bare-say", "turn-1", "bare_say_followup")
    assert brief.compound_request is None
    assert brief.commitments[0].summary == text
    assert ledger._classify_compound_step_kind(text.split(" and ", 1)[1], unresolved_summary=None) != "report"
    assert ledger._classify_compound_step_kind(text.split(" and ", 1)[0], unresolved_summary=None) == expected_kind


@pytest.mark.parametrize(
    "text",
    [
        "Look right and say what you see.",
        "Look right and say whether you can see it.",
        "Look right and tell me what you see.",
        "Look right, check diagnostics, then report done.",
    ],
)
def test_compound_parser_structured_report_followups_remain_supported(text: str) -> None:
    ledger = ContinuityLedger()

    ledger.update_from_event(
        "transcript_final",
        text=text,
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-structured-report", "turn-1", "structured_report_followup")
    assert brief.compound_request is not None
    assert any(step.kind == "report" for step in brief.compound_request.steps)


def test_report_semantics_classifier_marks_auditory_verify() -> None:
    ledger = ContinuityLedger()
    traits = ledger._classify_report_semantics(
        clause="report whether you can hear me now",
        kind="report",
        prior_steps=(),
    )

    assert traits["requires_perception"] is True
    assert traits["perception_mode"] == "auditory"
    assert traits["report_intent"] == "verify"


def test_report_semantics_status_helper_keeps_centered_perception_dependent() -> None:
    ledger = ContinuityLedger()
    traits = ledger._classify_report_semantics(
        clause="tell me what you see when you're centered on the object",
        kind="report",
        prior_steps=(),
    )

    assert traits["requires_perception"] is True
    assert traits["perception_mode"] == "visual"
    assert traits["report_intent"] == "describe"


def test_report_semantics_status_helper_keeps_holding_perception_dependent() -> None:
    ledger = ContinuityLedger()
    traits = ledger._classify_report_semantics(
        clause="tell me what i'm holding in my hand",
        kind="report",
        prior_steps=(),
    )

    assert traits["requires_perception"] is True
    assert traits["perception_mode"] == "visual"
    assert traits["report_intent"] == "identify"


def test_status_only_helper_negative_for_perception_clause() -> None:
    ledger = ContinuityLedger()
    assert ledger._is_status_only_report_clause("tell me what you see when you're centered on the object") is False


def test_status_only_helper_positive_for_explicit_centered_clause() -> None:
    ledger = ContinuityLedger()
    assert ledger._is_status_only_report_clause("let me know once you're centered") is True


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


def test_compound_tool_result_does_not_miscredit_left_to_right_step() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, then look left, then report done.",
        source="input_audio_transcription",
    )

    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_left",
        call_id="call-1",
    )

    after_wrong_direction = ledger.build_brief("run-direction", "turn-1", "wrong_direction_first")
    assert after_wrong_direction.compound_request is not None
    assert [step.status for step in after_wrong_direction.compound_request.steps] == ["active", "pending", "pending"]
    assert after_wrong_direction.compound_request.recent_completed_step_id is None
    assert after_wrong_direction.compound_request.next_pending_step_id == "step_2"

    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-2",
    )
    after_correct_direction = ledger.build_brief("run-direction", "turn-1", "correct_direction_second")
    assert after_correct_direction.compound_request is not None
    assert [step.status for step in after_correct_direction.compound_request.steps] == ["completed", "active", "pending"]
    assert after_correct_direction.compound_request.recent_completed_step_id == "step_1"
    assert after_correct_direction.compound_request.next_pending_step_id == "step_3"


def test_compound_tool_result_center_matches_only_center_step() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Come back to center, then tell me what you see.",
        source="input_audio_transcription",
    )

    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_left",
        call_id="call-1",
    )
    wrong_direction = ledger.build_brief("run-center", "turn-1", "left_not_center")
    assert wrong_direction.compound_request is not None
    assert [step.status for step in wrong_direction.compound_request.steps] == ["active", "pending"]

    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_center",
        call_id="call-2",
    )
    correct_direction = ledger.build_brief("run-center", "turn-1", "center_match")
    assert correct_direction.compound_request is not None
    assert [step.status for step in correct_direction.compound_request.steps] == ["completed", "pending"]
    assert correct_direction.compound_request.recent_completed_step_id == "step_1"


def test_compound_left_right_center_report_sequence_advances_in_order() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text=(
            "Hey Theo, can you start by looking left, and then look right, and then come back to center "
            "and tell me what I'm holding in my hand?"
        ),
        source="input_audio_transcription",
    )

    initial = ledger.build_brief("run-sequence", "turn-1", "initial")
    assert initial.compound_request is not None
    assert [step.status for step in initial.compound_request.steps] == ["active", "pending", "pending", "pending"]

    ledger.update_from_event("tool_result_received", tool_name="gesture_look_left", call_id="call-1")
    after_left = ledger.build_brief("run-sequence", "turn-1", "after_left")
    assert after_left.compound_request is not None
    assert [step.status for step in after_left.compound_request.steps] == ["completed", "active", "pending", "pending"]
    assert after_left.compound_request.recent_completed_step_id == "step_1"
    assert after_left.compound_request.next_pending_step_id == "step_3"

    ledger.update_from_event("tool_result_received", tool_name="gesture_look_right", call_id="call-2")
    after_right = ledger.build_brief("run-sequence", "turn-1", "after_right")
    assert after_right.compound_request is not None
    assert [step.status for step in after_right.compound_request.steps] == ["completed", "completed", "active", "pending"]
    assert after_right.compound_request.recent_completed_step_id == "step_2"
    assert after_right.compound_request.next_pending_step_id == "step_4"

    ledger.update_from_event("tool_result_received", tool_name="gesture_look_center", call_id="call-3")
    after_center = ledger.build_brief("run-sequence", "turn-1", "after_center")
    assert after_center.compound_request is not None
    assert [step.status for step in after_center.compound_request.steps] == ["completed", "completed", "completed", "pending"]
    assert after_center.compound_request.recent_completed_step_id == "step_3"
    assert after_center.compound_request.next_pending_step_id == "step_4"
    assert after_center.compound_request.final_followup_pending is True


def test_compound_parser_classifies_directional_fragments_as_gesture_steps() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look up and then down and then back to center.",
        source="input_audio_transcription",
    )

    brief = ledger.build_brief("run-directional", "turn-1", "directional_fragments")
    assert brief.compound_request is not None
    assert [step.kind for step in brief.compound_request.steps] == ["gesture", "gesture", "gesture"]
    assert [step.summary for step in brief.compound_request.steps] == [
        "Look up.",
        "down.",
        "back to center.",
    ]


def test_compound_look_up_down_center_report_progression_stays_ordered() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Can you look up and then down and then back to center, and then tell me what your voltage is?",
        source="input_audio_transcription",
    )

    initial = ledger.build_brief("run-up-down", "turn-1", "initial")
    assert initial.compound_request is not None
    assert [step.kind for step in initial.compound_request.steps] == ["gesture", "gesture", "gesture", "report"]
    assert [step.status for step in initial.compound_request.steps] == ["active", "pending", "pending", "pending"]
    assert initial.compound_request.active_step_index == 0
    assert initial.compound_request.next_pending_step_id == "step_2"

    ledger.update_from_event("tool_result_received", tool_name="gesture_look_up", call_id="call-up")
    after_up = ledger.build_brief("run-up-down", "turn-1", "after_up")
    assert after_up.compound_request is not None
    assert [step.status for step in after_up.compound_request.steps] == ["completed", "active", "pending", "pending"]
    assert after_up.compound_request.active_step_index == 1
    assert after_up.compound_request.next_pending_step_id == "step_3"

    ledger.update_from_event("tool_call_started", tool_name="gesture_look_down", call_id="call-down")
    during_down = ledger.build_brief("run-up-down", "turn-1", "during_down")
    assert during_down.compound_request is not None
    assert [step.status for step in during_down.compound_request.steps] == ["completed", "active", "pending", "pending"]
    assert during_down.compound_request.active_step_index == 1
    assert during_down.compound_request.next_pending_step_id == "step_3"

    ledger.update_from_event("tool_result_received", tool_name="gesture_look_down", call_id="call-down")
    after_down = ledger.build_brief("run-up-down", "turn-1", "after_down")
    assert after_down.compound_request is not None
    assert [step.status for step in after_down.compound_request.steps] == ["completed", "completed", "active", "pending"]
    assert after_down.compound_request.active_step_index == 2
    assert after_down.compound_request.next_pending_step_id == "step_4"


def test_compound_look_left_right_progression_stays_ordered() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look left and then right.",
        source="input_audio_transcription",
    )

    initial = ledger.build_brief("run-left-right", "turn-1", "initial")
    assert initial.compound_request is not None
    assert [step.kind for step in initial.compound_request.steps] == ["gesture", "gesture"]
    assert [step.status for step in initial.compound_request.steps] == ["active", "pending"]

    ledger.update_from_event("tool_result_received", tool_name="gesture_look_left", call_id="call-left")
    after_left = ledger.build_brief("run-left-right", "turn-1", "after_left")
    assert after_left.compound_request is not None
    assert [step.status for step in after_left.compound_request.steps] == ["completed", "active"]
    assert after_left.compound_request.active_step_index == 1
    assert after_left.compound_request.next_pending_step_id is None

    ledger.update_from_event("tool_result_received", tool_name="gesture_look_right", call_id="call-right")
    after_right = ledger.build_brief("run-left-right", "turn-1", "after_right")
    assert after_right.compound_request is not None
    assert [step.status for step in after_right.compound_request.steps] == ["completed", "completed"]
    assert after_right.compound_request.active_step_index is None
    assert after_right.compound_request.next_pending_step_id is None


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
    assert after_followup.compound_request is not None
    assert after_followup.compound_request.final_followup_pending is True
    assert after_followup.compound_request.steps[2].status == "pending"


def test_compound_response_done_does_not_complete_report_before_prior_non_report_step() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics, look left, go back to center, then tell me what you saw.",
        source="input_audio_transcription",
    )
    ledger.update_from_event("tool_result_received", tool_name="gesture_look_right", call_id="call-1")
    ledger.update_from_event("tool_result_received", tool_name="read_runtime_diagnostics", call_id="call-2")
    ledger.update_from_event("tool_result_received", tool_name="gesture_look_left", call_id="call-3")

    before_done = ledger.build_brief("run-order", "turn-1", "before_intermediate_done")
    assert before_done.compound_request is not None
    assert [step.status for step in before_done.compound_request.steps] == [
        "completed",
        "completed",
        "completed",
        "active",
        "pending",
    ]
    assert before_done.compound_request.final_followup_pending is True

    ledger.update_from_event("response_done", close_unresolved="true")

    after_intermediate_done = ledger.build_brief("run-order", "turn-1", "after_intermediate_done")
    assert after_intermediate_done.compound_request is not None
    assert [step.status for step in after_intermediate_done.compound_request.steps] == [
        "completed",
        "completed",
        "completed",
        "active",
        "pending",
    ]
    assert after_intermediate_done.compound_request.final_followup_pending is True


def test_compound_chain_closes_only_after_last_non_report_step_then_report() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics, look left, go back to center, then tell me what you saw.",
        source="input_audio_transcription",
    )
    ledger.update_from_event("tool_result_received", tool_name="gesture_look_right", call_id="call-1")
    ledger.update_from_event("tool_result_received", tool_name="read_runtime_diagnostics", call_id="call-2")
    ledger.update_from_event("tool_result_received", tool_name="gesture_look_left", call_id="call-3")
    ledger.update_from_event("response_done", close_unresolved="true")
    ledger.update_from_event("tool_result_received", tool_name="gesture_look_center", call_id="call-4")

    before_final_report_done = ledger.build_brief("run-order", "turn-1", "before_final_report_done")
    assert before_final_report_done.compound_request is not None
    assert [step.status for step in before_final_report_done.compound_request.steps] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "pending",
    ]
    assert before_final_report_done.compound_request.final_followup_pending is True

    ledger.update_from_event("response_done", close_unresolved="true")
    after_non_final_response_done = ledger.build_brief("run-order", "turn-1", "after_non_final_response_done")
    assert after_non_final_response_done.compound_request is not None
    assert [step.status for step in after_non_final_response_done.compound_request.steps] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "pending",
    ]
    assert after_non_final_response_done.compound_request.final_followup_pending is True

    ledger.update_from_event("response_done", complete_final_report="true")
    after_final_report_done = ledger.build_brief("run-order", "turn-1", "after_final_report_done")
    assert after_final_report_done.compound_request is None


def test_compound_response_done_with_close_commitment_does_not_skip_pending_followthrough_steps() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics, look center, then tell me what diagnostics reported.",
        source="input_audio_transcription",
    )
    ledger.update_from_event("tool_result_received", tool_name="gesture_look_right", call_id="call-1")
    ledger.update_from_event("tool_result_received", tool_name="read_runtime_diagnostics", call_id="call-2")

    ledger.update_from_event("response_done", close_commitment="true", close_unresolved="true")

    after_done = ledger.build_brief("run-order", "turn-1", "after_mid_chain_done")
    assert after_done.compound_request is not None
    assert [step.status for step in after_done.compound_request.steps] == [
        "completed",
        "completed",
        "active",
        "pending",
    ]
    assert after_done.compound_request.final_followup_pending is True


def test_compound_followthrough_does_not_cross_turn_without_explicit_rebind() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics, look left, then return to center and report done.",
        source="input_audio_transcription",
        turn_id="turn_2",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-1",
        turn_id="turn_2",
    )

    before_cross_turn = ledger.build_brief("run-owner", "turn_2", "before_cross_turn")
    assert before_cross_turn.compound_request is not None
    assert [step.status for step in before_cross_turn.compound_request.steps][:4] == [
        "completed",
        "active",
        "pending",
        "pending",
    ]

    ledger.update_from_event(
        "tool_call_started",
        tool_name="gesture_look_left",
        call_id="call-2",
        turn_id="turn_4",
        commitment_summary="Did you get stuck?",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_left",
        call_id="call-2",
        turn_id="turn_4",
    )

    after_cross_turn = ledger.build_brief("run-owner", "turn_4", "after_cross_turn")
    assert after_cross_turn.compound_request is not None
    assert [step.status for step in after_cross_turn.compound_request.steps][:4] == [
        "completed",
        "active",
        "pending",
        "pending",
    ]
    assert after_cross_turn.commitments[0].summary != "Did you get stuck?"


def test_compound_followthrough_can_rebind_cross_turn_with_explicit_reason() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics, look left, then return to center and report done.",
        source="input_audio_transcription",
        turn_id="turn_2",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-1",
        turn_id="turn_2",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="read_runtime_diagnostics",
        call_id="call-2",
        turn_id="turn_4",
        allow_cross_turn_rebind=True,
        cross_turn_rebind_reason="followup_adopted_after_user_ping",
    )

    after_rebind = ledger.build_brief("run-owner", "turn_4", "after_rebind")
    assert after_rebind.compound_request is not None
    assert [step.status for step in after_rebind.compound_request.steps][:4] == [
        "completed",
        "completed",
        "active",
        "pending",
    ]


def test_compound_response_done_cross_turn_rebind_completes_report_and_logs_reason(caplog: pytest.LogCaptureFixture) -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, check diagnostics, then report done.",
        source="input_audio_transcription",
        turn_id="turn_parent",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-1",
        turn_id="turn_parent",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="read_runtime_diagnostics",
        call_id="call-2",
        turn_id="turn_parent",
    )

    with caplog.at_level(logging.INFO):
        ledger.update_from_event(
            "response_done",
            turn_id="turn_parent",
            close_commitment="true",
            complete_final_report="true",
            allow_cross_turn_rebind="true",
            cross_turn_rebind_reason="semantic_owner_parent_promoted",
        )

    brief = ledger.build_brief("run-rebind", "turn_parent", "after_response_done_rebind")
    assert brief.compound_request is None
    assert "continuity_compound_owner_rebind_accepted" in caplog.text
    assert "reason=semantic_owner_parent_promoted" in caplog.text


def test_compound_final_followthrough_after_intermediate_turn_rebind_clears_pending_state() -> None:
    ledger = ContinuityLedger()
    ledger.update_from_event(
        "transcript_final",
        text="Look right, look left, then look center and tell me when done.",
        source="input_audio_transcription",
        turn_id="turn_2",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_right",
        call_id="call-1",
        turn_id="turn_2",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_left",
        call_id="call-2",
        turn_id="turn_2",
    )

    ledger.update_from_event(
        "response_done",
        turn_id="turn_3",
        keep_ongoing="true",
    )

    before_followthrough = ledger.build_brief("run-rebind", "turn_3", "before_followthrough")
    assert before_followthrough.compound_request is not None
    assert before_followthrough.compound_request.active_step_index == 2
    assert before_followthrough.compound_request.final_followup_pending is True

    ledger.update_from_event(
        "tool_call_started",
        tool_name="gesture_look_center",
        call_id="call-3",
        turn_id="turn_3",
        allow_cross_turn_rebind="true",
        cross_turn_rebind_reason="tool_followup_semantic_owner_handoff",
    )
    ledger.update_from_event(
        "tool_result_received",
        tool_name="gesture_look_center",
        call_id="call-3",
        turn_id="turn_3",
        allow_cross_turn_rebind="true",
        cross_turn_rebind_reason="tool_followup_semantic_owner_handoff",
    )
    ledger.update_from_event(
        "response_done",
        turn_id="turn_3",
        close_commitment="true",
        close_unresolved="true",
        complete_final_report="true",
        allow_cross_turn_rebind="true",
        cross_turn_rebind_reason="semantic_owner_parent_promoted",
    )

    after_done = ledger.build_brief("run-rebind", "turn_3", "after_followthrough")
    assert after_done.compound_request is None


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

    with caplog.at_level(logging.DEBUG, logger="ai.continuity"):
        ledger.build_brief("run-1", "turn-1", "session_health")
        clock.advance(10.0)
        ledger.build_brief("run-1", "turn-1", "session_health")

    info_brief_logs = [
        record
        for record in caplog.records
        if record.message.startswith("continuity_brief_built") and record.levelno == logging.INFO
    ]
    debug_brief_logs = [
        record for record in caplog.records if record.message.startswith("continuity_brief_built")
    ]
    assert len(info_brief_logs) == 1
    assert len(debug_brief_logs) == 2
    assert "fingerprint_changed=True" in debug_brief_logs[0].message
    assert "fingerprint_changed=False" in debug_brief_logs[1].message
    assert "reminder=True" in debug_brief_logs[1].message
    assert "cooldown_elapsed_s=10.0" in debug_brief_logs[1].message


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
