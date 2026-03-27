"""Authoritative response.create arbitration contract.

This seam owns final contender selection for response.create decisions while
accepting lifecycle-policy output as one contender input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from ai.interaction_lifecycle_policy import ResponseCreateDecision, ResponseCreateDecisionAction

ResponseCreateArbitrationAction = Literal["SEND", "SCHEDULE", "BLOCK", "DROP"]

_RESPONSE_PATH_PRIORITY_LINEAGE_BLOCK = 300
_RESPONSE_PATH_PRIORITY_TERMINAL_BLOCK = 290
_RESPONSE_PATH_PRIORITY_SAME_TURN_OWNER_DROP = 280
_RESPONSE_PATH_PRIORITY_LIFECYCLE = 100


class _ResponseCreateSnapshot(Protocol):
    lineage_allowed: bool
    lineage_reason: str
    terminal_state_blocked: bool
    same_turn_owner_present: bool
    same_turn_owner_reason: str | None


@dataclass(frozen=True)
class ResponseCreateArbitrationCandidate:
    candidate_id: str
    action: ResponseCreateArbitrationAction
    reason_code: str
    explanation: str
    priority: int
    queue_reason: str | None = None
    blocked_by_terminal_state: bool = False
    should_log_arbitration: bool = True


@dataclass(frozen=True)
class ResponseCreateArbitrationDecision:
    action: ResponseCreateArbitrationAction
    reason_code: str
    explanation: str
    selected_candidate_id: str
    queue_reason: str | None = None
    blocked_by_terminal_state: bool = False
    should_log_arbitration: bool = True


def _select_response_path_candidate(
    candidates: list[ResponseCreateArbitrationCandidate],
) -> ResponseCreateArbitrationCandidate:
    return sorted(
        candidates,
        key=lambda candidate: (-int(candidate.priority), str(candidate.candidate_id or "")),
    )[0]


def _build_response_path_candidates(
    *,
    prepared_snapshot: _ResponseCreateSnapshot,
    lifecycle_decision: ResponseCreateDecision | None,
) -> list[ResponseCreateArbitrationCandidate]:
    candidates: list[ResponseCreateArbitrationCandidate] = []
    if not prepared_snapshot.lineage_allowed:
        candidates.append(
            ResponseCreateArbitrationCandidate(
                candidate_id="tool_lineage_guard",
                action="BLOCK",
                reason_code=prepared_snapshot.lineage_reason or "lineage_blocked",
                explanation="Tool lineage guard blocked response.create.",
                priority=_RESPONSE_PATH_PRIORITY_LINEAGE_BLOCK,
                should_log_arbitration=False,
            )
        )
    if prepared_snapshot.terminal_state_blocked:
        candidates.append(
            ResponseCreateArbitrationCandidate(
                candidate_id="canonical_terminal_state",
                action="BLOCK",
                reason_code="canonical_terminal_state",
                explanation="Canonical turn is already terminal.",
                priority=_RESPONSE_PATH_PRIORITY_TERMINAL_BLOCK,
                blocked_by_terminal_state=True,
                should_log_arbitration=False,
            )
        )
    if prepared_snapshot.same_turn_owner_present:
        candidates.append(
            ResponseCreateArbitrationCandidate(
                candidate_id="same_turn_owner",
                action="DROP",
                reason_code="same_turn_already_owned",
                explanation=f"Assistant message suppressed by same-turn owner: {prepared_snapshot.same_turn_owner_reason}.",
                priority=_RESPONSE_PATH_PRIORITY_SAME_TURN_OWNER_DROP,
            )
        )
    if lifecycle_decision is not None:
        lifecycle_action: ResponseCreateArbitrationAction = "SEND"
        lifecycle_explanation = "Response.create allowed for immediate send."
        lifecycle_queue_reason: str | None = None
        if lifecycle_decision.action is ResponseCreateDecisionAction.SCHEDULE:
            lifecycle_action = "SCHEDULE"
            lifecycle_explanation = f"Response.create deferred: {lifecycle_decision.reason_code}."
            lifecycle_queue_reason = lifecycle_decision.queue_reason
        elif lifecycle_decision.action is ResponseCreateDecisionAction.BLOCK:
            lifecycle_action = "BLOCK"
            lifecycle_explanation = f"Response.create blocked: {lifecycle_decision.reason_code}."
        candidates.append(
            ResponseCreateArbitrationCandidate(
                candidate_id=lifecycle_decision.selected_candidate_id,
                action=lifecycle_action,
                reason_code=lifecycle_decision.reason_code,
                explanation=lifecycle_explanation,
                priority=_RESPONSE_PATH_PRIORITY_LIFECYCLE,
                queue_reason=lifecycle_queue_reason,
            )
        )
    return candidates


def decide_response_create_arbitration(
    *,
    prepared_snapshot: _ResponseCreateSnapshot,
    lifecycle_decision: ResponseCreateDecision | None,
) -> ResponseCreateArbitrationDecision:
    selected_candidate = _select_response_path_candidate(
        _build_response_path_candidates(
            prepared_snapshot=prepared_snapshot,
            lifecycle_decision=lifecycle_decision,
        )
    )
    return ResponseCreateArbitrationDecision(
        action=selected_candidate.action,
        reason_code=selected_candidate.reason_code,
        explanation=selected_candidate.explanation,
        selected_candidate_id=selected_candidate.candidate_id,
        queue_reason=selected_candidate.queue_reason,
        blocked_by_terminal_state=selected_candidate.blocked_by_terminal_state,
        should_log_arbitration=selected_candidate.should_log_arbitration,
    )
