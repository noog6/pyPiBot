from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.terminal_deliverable_arbitration import (
    TerminalDeliverableDecision,
    arbitrate_terminal_deliverable_selection,
)


def _decide(**overrides: object) -> TerminalDeliverableDecision:
    facts: dict[str, object] = {
        "delivery_state_before_done": "done",
        "origin": "assistant_message",
        "active_response_was_provisional": False,
        "response_done_is_empty": False,
        "transcript_final_seen": True,
        "turn_has_pending_tool_followup": False,
        "exact_phrase_obligation_open": False,
        "descriptive_turn": False,
        "tool_output_gesture_only": False,
    }
    facts.update(overrides)
    return arbitrate_terminal_deliverable_selection(**facts)


def test_terminal_deliverable_arbitration_selects_normal_terminal_response() -> None:
    decision = _decide()

    assert decision == TerminalDeliverableDecision(True, "normal", "terminal_selected")


def test_terminal_deliverable_arbitration_enforces_tool_followup_precedence() -> None:
    decision = _decide(turn_has_pending_tool_followup=True, transcript_final_seen=True)

    assert decision == TerminalDeliverableDecision(
        False,
        "tool_followup_precedence",
        "tool_followup_precedence",
    )


def test_terminal_deliverable_arbitration_blocks_open_exact_phrase_obligation() -> None:
    decision = _decide(exact_phrase_obligation_open=True)

    assert decision == TerminalDeliverableDecision(
        False,
        "exact_phrase_obligation_open",
        "exact_phrase_obligation_open",
    )


def test_terminal_deliverable_arbitration_defers_provisional_server_auto_before_transcript_final() -> None:
    decision = _decide(
        origin="server_auto",
        active_response_was_provisional=True,
        transcript_final_seen=False,
    )

    assert decision == TerminalDeliverableDecision(
        False,
        "provisional_server_auto_awaiting_transcript_final",
        "provisional_server_auto_awaiting_transcript_final",
    )


def test_terminal_deliverable_arbitration_rejects_descriptive_gesture_only_tool_output() -> None:
    decision = _decide(
        origin="tool_output",
        descriptive_turn=True,
        tool_output_gesture_only=True,
    )

    assert decision == TerminalDeliverableDecision(
        False,
        "tool_output_descriptive_gesture_only",
        "tool_output_descriptive_gesture_only",
    )


def test_terminal_deliverable_arbitration_covers_cancelled_and_other_non_deliverables() -> None:
    cancelled = _decide(delivery_state_before_done="cancelled")
    micro_ack = _decide(origin="micro_ack")
    empty_tool_followup = _decide(
        origin="tool_output",
        response_done_is_empty=True,
    )
    provisional_empty = _decide(
        active_response_was_provisional=True,
        response_done_is_empty=True,
        transcript_final_seen=False,
    )

    assert cancelled == TerminalDeliverableDecision(False, "cancelled", "cancelled")
    assert micro_ack == TerminalDeliverableDecision(
        False,
        "micro_ack_non_deliverable",
        "micro_ack_non_deliverable",
    )
    assert empty_tool_followup == TerminalDeliverableDecision(
        False,
        "empty_tool_followup_non_deliverable",
        "empty_tool_followup_non_deliverable",
    )
    assert provisional_empty == TerminalDeliverableDecision(
        False,
        "provisional_empty_non_deliverable",
        "provisional_empty_non_deliverable",
    )
