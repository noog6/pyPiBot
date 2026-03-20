import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.tool_followup_arbitration import (
    ToolFollowupDecisionAction,
    decide_tool_followup_arbitration,
)


def test_decide_tool_followup_arbitration_suppresses_parent_covered_gesture_followup() -> None:
    decision = decide_tool_followup_arbitration(
        suppressible=True,
        has_distinct_info=False,
        is_low_risk_reversible_gesture_tool=True,
        parent_resolved=True,
        parent_origin="assistant_message",
        parent_done=True,
        parent_covered=True,
        coverage_source="canonical",
        deliverable_class="progress",
        terminal_selected=False,
        terminal_reason="",
        classification_pending=False,
    )

    assert decision.action is ToolFollowupDecisionAction.SUPPRESS
    assert decision.reason_code == "parent_covered_tool_result"
    assert decision.parent_coverage_state == "covered_canonical"
    assert decision.blocked_by_parent_final_coverage is False


def test_decide_tool_followup_arbitration_holds_when_parent_classification_pending() -> None:
    decision = decide_tool_followup_arbitration(
        suppressible=True,
        has_distinct_info=False,
        is_low_risk_reversible_gesture_tool=True,
        parent_resolved=True,
        parent_origin="server_auto",
        parent_done=True,
        parent_covered=False,
        coverage_source="canonical",
        deliverable_class="unknown",
        terminal_selected=False,
        terminal_reason="",
        classification_pending=True,
    )

    assert decision.action is ToolFollowupDecisionAction.HOLD
    assert decision.reason_code == "parent_deliverable_pending"
    assert decision.parent_coverage_state == "coverage_pending"
    assert decision.blocked_by_parent_final_coverage is False


def test_decide_tool_followup_arbitration_releases_distinct_info_without_parent_coverage() -> None:
    decision = decide_tool_followup_arbitration(
        suppressible=True,
        has_distinct_info=True,
        is_low_risk_reversible_gesture_tool=False,
        parent_resolved=False,
        parent_origin="",
        parent_done=False,
        parent_covered=False,
        coverage_source="none",
        deliverable_class="unknown",
        terminal_selected=False,
        terminal_reason="",
        classification_pending=False,
    )

    assert decision.action is ToolFollowupDecisionAction.RELEASE
    assert decision.reason_code == "distinct_info"
    assert decision.parent_coverage_state == "not_applicable"
    assert decision.blocked_by_parent_final_coverage is None
