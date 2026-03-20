"""Pure decision seam for terminal deliverable arbitration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalDeliverableDecision:
    """Structured terminal deliverable arbitration result."""

    selected: bool
    reason_code: str
    selected_candidate_id: str


def arbitrate_terminal_deliverable_selection(
    *,
    delivery_state_before_done: str | None,
    origin: str,
    active_response_was_provisional: bool,
    response_done_is_empty: bool,
    transcript_final_seen: bool,
    turn_has_pending_tool_followup: bool,
    exact_phrase_obligation_open: bool,
    descriptive_turn: bool,
    tool_output_gesture_only: bool,
) -> TerminalDeliverableDecision:
    """Choose whether a terminal response should become the turn deliverable."""

    normalized_origin = str(origin or "").strip().lower()

    if delivery_state_before_done == "cancelled":
        return TerminalDeliverableDecision(
            selected=False,
            reason_code="cancelled",
            selected_candidate_id="cancelled",
        )
    if normalized_origin == "micro_ack":
        return TerminalDeliverableDecision(
            selected=False,
            reason_code="micro_ack_non_deliverable",
            selected_candidate_id="micro_ack_non_deliverable",
        )
    if active_response_was_provisional and response_done_is_empty:
        return TerminalDeliverableDecision(
            selected=False,
            reason_code="provisional_empty_non_deliverable",
            selected_candidate_id="provisional_empty_non_deliverable",
        )
    if active_response_was_provisional and normalized_origin == "server_auto" and not bool(transcript_final_seen):
        return TerminalDeliverableDecision(
            selected=False,
            reason_code="provisional_server_auto_awaiting_transcript_final",
            selected_candidate_id="provisional_server_auto_awaiting_transcript_final",
        )
    if turn_has_pending_tool_followup and normalized_origin != "tool_output":
        return TerminalDeliverableDecision(
            selected=False,
            reason_code="tool_followup_precedence",
            selected_candidate_id="tool_followup_precedence",
        )
    if exact_phrase_obligation_open:
        return TerminalDeliverableDecision(
            selected=False,
            reason_code="exact_phrase_obligation_open",
            selected_candidate_id="exact_phrase_obligation_open",
        )
    if normalized_origin == "tool_output" and descriptive_turn and tool_output_gesture_only:
        return TerminalDeliverableDecision(
            selected=False,
            reason_code="tool_output_descriptive_gesture_only",
            selected_candidate_id="tool_output_descriptive_gesture_only",
        )
    return TerminalDeliverableDecision(
        selected=True,
        reason_code="normal",
        selected_candidate_id="terminal_selected",
    )
