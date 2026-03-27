from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.interaction_lifecycle_policy import ResponseCreateDecision, ResponseCreateDecisionAction
from ai.response_create_arbitration import decide_response_create_arbitration


@dataclass(frozen=True)
class _Snapshot:
    lineage_allowed: bool = True
    lineage_reason: str = ""
    terminal_state_blocked: bool = False
    same_turn_owner_present: bool = False
    same_turn_owner_reason: str | None = None


def test_arbitration_prefers_same_turn_owner_over_lifecycle() -> None:
    decision = decide_response_create_arbitration(
        prepared_snapshot=_Snapshot(same_turn_owner_present=True, same_turn_owner_reason="tool_followup_owned"),
        lifecycle_decision=ResponseCreateDecision(
            action=ResponseCreateDecisionAction.SEND,
            reason_code="direct_send",
            selected_candidate_id="direct_send",
        ),
    )

    assert decision.action == "DROP"
    assert decision.reason_code == "same_turn_already_owned"
    assert decision.selected_candidate_id == "same_turn_owner"


def test_arbitration_uses_lifecycle_send_when_no_runtime_overlays() -> None:
    decision = decide_response_create_arbitration(
        prepared_snapshot=_Snapshot(),
        lifecycle_decision=ResponseCreateDecision(
            action=ResponseCreateDecisionAction.SEND,
            reason_code="direct_send",
            selected_candidate_id="direct_send",
        ),
    )

    assert decision.action == "SEND"
    assert decision.reason_code == "direct_send"
    assert decision.selected_candidate_id == "direct_send"


def test_arbitration_uses_lifecycle_when_no_runtime_overlays() -> None:
    decision = decide_response_create_arbitration(
        prepared_snapshot=_Snapshot(),
        lifecycle_decision=ResponseCreateDecision(
            action=ResponseCreateDecisionAction.SCHEDULE,
            reason_code="active_response",
            selected_candidate_id="active_response",
            queue_reason="active_response",
        ),
    )

    assert decision.action == "SCHEDULE"
    assert decision.reason_code == "active_response"
    assert decision.queue_reason == "active_response"


def test_arbitration_terminal_state_overrides_lifecycle_send() -> None:
    decision = decide_response_create_arbitration(
        prepared_snapshot=_Snapshot(terminal_state_blocked=True),
        lifecycle_decision=ResponseCreateDecision(
            action=ResponseCreateDecisionAction.SEND,
            reason_code="direct_send",
            selected_candidate_id="direct_send",
        ),
    )

    assert decision.action == "BLOCK"
    assert decision.reason_code == "canonical_terminal_state"
    assert decision.selected_candidate_id == "canonical_terminal_state"


def test_arbitration_lineage_guard_overrides_lifecycle_send() -> None:
    decision = decide_response_create_arbitration(
        prepared_snapshot=_Snapshot(lineage_allowed=False, lineage_reason="parent_lineage_unavailable"),
        lifecycle_decision=ResponseCreateDecision(
            action=ResponseCreateDecisionAction.SEND,
            reason_code="direct_send",
            selected_candidate_id="direct_send",
        ),
    )

    assert decision.action == "BLOCK"
    assert decision.reason_code == "parent_lineage_unavailable"
    assert decision.selected_candidate_id == "tool_lineage_guard"
