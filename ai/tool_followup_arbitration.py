"""Pure seam-local arbitration helper for tool-followup release decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolFollowupDecisionAction(str, Enum):
    RELEASE = "release"
    SUPPRESS = "suppress"
    HOLD = "hold"


@dataclass(frozen=True)
class ToolFollowupArbitrationDecision:
    action: ToolFollowupDecisionAction
    reason_code: str
    parent_coverage_state: str
    blocked_by_parent_final_coverage: bool | None

    @property
    def should_release(self) -> bool:
        return self.action is ToolFollowupDecisionAction.RELEASE

    @property
    def should_suppress(self) -> bool:
        return self.action is ToolFollowupDecisionAction.SUPPRESS

    @property
    def should_hold(self) -> bool:
        return self.action is ToolFollowupDecisionAction.HOLD


def _release_reason_for_uncovered_parent(*, terminal_selected: bool, terminal_reason: str, coverage_source: str) -> str:
    normalized_terminal_reason = str(terminal_reason or "").strip().lower()
    normalized_coverage_source = str(coverage_source or "").strip().lower()
    if terminal_selected and normalized_terminal_reason == "normal":
        return "parent_terminal_selection_not_coverage_qualified"
    if normalized_coverage_source not in {"", "none"}:
        return "parent_coverage_unknown"
    return "parent_not_coverage_qualified"


def decide_tool_followup_arbitration(
    *,
    suppressible: bool,
    has_distinct_info: bool,
    is_low_risk_reversible_gesture_tool: bool,
    parent_resolved: bool,
    parent_origin: str,
    parent_done: bool,
    parent_covered: bool,
    coverage_source: str,
    deliverable_class: str,
    terminal_selected: bool,
    terminal_reason: str,
    classification_pending: bool,
) -> ToolFollowupArbitrationDecision:
    normalized_parent_origin = str(parent_origin or "").strip().lower()
    normalized_coverage_source = str(coverage_source or "").strip().lower()
    normalized_deliverable_class = str(deliverable_class or "").strip().lower()
    normalized_terminal_reason = str(terminal_reason or "").strip().lower()

    if not suppressible:
        return ToolFollowupArbitrationDecision(
            action=ToolFollowupDecisionAction.RELEASE,
            reason_code="not_suppressible",
            parent_coverage_state="not_applicable",
            blocked_by_parent_final_coverage=None,
        )
    if has_distinct_info:
        return ToolFollowupArbitrationDecision(
            action=ToolFollowupDecisionAction.RELEASE,
            reason_code="distinct_info",
            parent_coverage_state="not_applicable",
            blocked_by_parent_final_coverage=None,
        )
    if not is_low_risk_reversible_gesture_tool:
        return ToolFollowupArbitrationDecision(
            action=ToolFollowupDecisionAction.RELEASE,
            reason_code="non_gesture_tool",
            parent_coverage_state="not_applicable",
            blocked_by_parent_final_coverage=None,
        )
    if not parent_resolved:
        return ToolFollowupArbitrationDecision(
            action=ToolFollowupDecisionAction.RELEASE,
            reason_code="parent_unresolved",
            parent_coverage_state="unknown",
            blocked_by_parent_final_coverage=None,
        )
    if normalized_parent_origin in {"micro_ack", "tool_output"}:
        return ToolFollowupArbitrationDecision(
            action=ToolFollowupDecisionAction.RELEASE,
            reason_code="parent_origin_excluded",
            parent_coverage_state="unknown",
            blocked_by_parent_final_coverage=False,
        )
    if not parent_done:
        return ToolFollowupArbitrationDecision(
            action=ToolFollowupDecisionAction.RELEASE,
            reason_code="parent_not_done",
            parent_coverage_state="uncovered",
            blocked_by_parent_final_coverage=False,
        )
    if not parent_covered:
        if classification_pending:
            return ToolFollowupArbitrationDecision(
                action=ToolFollowupDecisionAction.HOLD,
                reason_code="parent_deliverable_pending",
                parent_coverage_state="coverage_pending",
                blocked_by_parent_final_coverage=False,
            )
        return ToolFollowupArbitrationDecision(
            action=ToolFollowupDecisionAction.RELEASE,
            reason_code=_release_reason_for_uncovered_parent(
                terminal_selected=terminal_selected,
                terminal_reason=normalized_terminal_reason,
                coverage_source=normalized_coverage_source,
            ),
            parent_coverage_state="unknown" if normalized_coverage_source not in {"", "none"} else "uncovered",
            blocked_by_parent_final_coverage=False,
        )

    parent_coverage_state = (
        "covered_terminal_selection"
        if normalized_coverage_source == "terminal_selection"
        else "covered_canonical"
    )
    blocked_by_parent_final_coverage = bool(
        normalized_deliverable_class == "final"
        or (bool(terminal_selected) and normalized_terminal_reason == "normal")
    )
    return ToolFollowupArbitrationDecision(
        action=ToolFollowupDecisionAction.SUPPRESS,
        reason_code="parent_covered_tool_result",
        parent_coverage_state=parent_coverage_state,
        blocked_by_parent_final_coverage=blocked_by_parent_final_coverage,
    )
