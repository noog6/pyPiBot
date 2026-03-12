"""Deterministic seam-local arbitration for opportunistic actions."""

from __future__ import annotations

from dataclasses import dataclass

from ai.decision_arbitration import ArbitrationAction, ArbitrationCandidate, decide_arbitration
from ai.governance_spine import GovernanceDecision


@dataclass(frozen=True)
class OpportunisticActionCandidate:
    action_kind: str
    source: str
    priority: int
    reason_code: str
    opportunistic: bool
    ttl_s: float | None = None


@dataclass(frozen=True)
class OpportunisticSuppression:
    action_kind: str
    source: str
    result: str
    reason_code: str


@dataclass(frozen=True)
class OpportunisticArbitrationResult:
    selected_action_kind: str
    selected_source: str
    reason_code: str
    selected_native_reason_code: str | None
    is_opportunistic: bool
    suppressed_or_deferred: tuple[OpportunisticSuppression, ...]


def arbitrate_opportunistic_actions(
    *,
    user_turn_priority_active: bool,
    response_obligation_priority_active: bool,
    confirmation_pending: bool,
    busy_turn: bool,
    safety_interlock: bool = False,
    candidate_curiosity: OpportunisticActionCandidate | None = None,
    candidate_curiosity_governance: GovernanceDecision | None = None,
    candidate_low_priority_injection: OpportunisticActionCandidate | None = None,
    candidate_embodiment_flourish: OpportunisticActionCandidate | None = None,
    candidate_embodiment_governance: GovernanceDecision | None = None,
) -> OpportunisticArbitrationResult:
    # Seam contract (explicit):
    # - governance.decision gates candidate eligibility for this seam.
    # - candidate.priority remains the seam-local comparator for winner selection.
    # - governance.priority stays observational/local metadata and is not used for
    #   cross-subsystem ordering in this arbiter.
    governance_by_key: dict[tuple[str, str], GovernanceDecision] = {}
    if isinstance(candidate_curiosity, OpportunisticActionCandidate) and isinstance(
        candidate_curiosity_governance, GovernanceDecision
    ):
        governance_by_key[(candidate_curiosity.action_kind, candidate_curiosity.source)] = (
            candidate_curiosity_governance
        )
    if isinstance(candidate_embodiment_flourish, OpportunisticActionCandidate) and isinstance(
        candidate_embodiment_governance, GovernanceDecision
    ):
        governance_by_key[(candidate_embodiment_flourish.action_kind, candidate_embodiment_flourish.source)] = (
            candidate_embodiment_governance
        )

    candidates: list[OpportunisticActionCandidate] = []
    for candidate in (
        candidate_curiosity,
        candidate_low_priority_injection,
        candidate_embodiment_flourish,
    ):
        if not isinstance(candidate, OpportunisticActionCandidate):
            continue
        governance = governance_by_key.get((candidate.action_kind, candidate.source))
        if governance is not None and governance.decision != "allow":
            continue
        candidates.append(candidate)

    selected_action_kind = "wait"
    selected_source = "governance_spine"
    selected_reason = "no_op_selected"
    selected_native_reason_code: str | None = None
    selected_is_opportunistic = False

    arbitration_candidates: list[ArbitrationCandidate] = []
    if safety_interlock:
        selected_reason = "opportunistic_suppressed"
    elif user_turn_priority_active:
        selected_action_kind = "explicit_user_turn"
        selected_source = "user_turn"
        selected_reason = "explicit_intent_priority"
    elif confirmation_pending:
        selected_reason = "confirmation_pending"
    elif response_obligation_priority_active:
        selected_reason = "obligation_open"
    elif busy_turn:
        selected_reason = "busy_turn"
    elif candidates:
        for candidate in candidates:
            arbitration_candidates.append(
                ArbitrationCandidate(
                    candidate_id=f"{candidate.action_kind}:{candidate.source}",
                    action=ArbitrationAction.DO_NOW,
                    reason_code=candidate.reason_code,
                    priority=int(candidate.priority),
                    defer_until=None,
                )
            )
        decision = decide_arbitration(
            policy_name="opportunistic_surface",
            candidates=arbitration_candidates,
            default_candidate=ArbitrationCandidate(
                candidate_id="wait:governance_spine",
                action=ArbitrationAction.DEFER,
                reason_code="no_op_selected",
                priority=0,
            ),
        )
        selected_id = str(decision.selected_candidate_id or "")
        selected_action_kind, _, selected_source = selected_id.partition(":")
        selected_reason = "arbitration_selected"
        selected_native_reason_code = decision.reason_code
        selected_is_opportunistic = True

    suppressed: list[OpportunisticSuppression] = []
    all_candidates = [
        candidate
        for candidate in (
            candidate_curiosity,
            candidate_low_priority_injection,
            candidate_embodiment_flourish,
        )
        if isinstance(candidate, OpportunisticActionCandidate)
    ]
    for candidate in all_candidates:
        governance = governance_by_key.get((candidate.action_kind, candidate.source))
        if candidate.action_kind == selected_action_kind and candidate.source == selected_source:
            continue
        if governance is not None and governance.decision != "allow":
            suppressed.append(
                OpportunisticSuppression(
                    action_kind=candidate.action_kind,
                    source=candidate.source,
                    result="defer" if governance.decision == "defer" else "suppress",
                    reason_code=governance.reason_code,
                )
            )
            continue
        non_selected_reason = (
            "opportunistic_deferred"
            if selected_reason in {"confirmation_pending", "obligation_open", "busy_turn"}
            else "opportunistic_suppressed"
        )
        non_selected_result = "defer" if non_selected_reason == "opportunistic_deferred" else "suppress"
        suppressed.append(
            OpportunisticSuppression(
                action_kind=candidate.action_kind,
                source=candidate.source,
                result=non_selected_result,
                reason_code=non_selected_reason,
            )
        )

    return OpportunisticArbitrationResult(
        selected_action_kind=selected_action_kind,
        selected_source=selected_source,
        reason_code=selected_reason,
        selected_native_reason_code=selected_native_reason_code,
        is_opportunistic=selected_is_opportunistic,
        suppressed_or_deferred=tuple(suppressed),
    )
