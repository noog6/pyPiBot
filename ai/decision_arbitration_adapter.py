"""Decision-arbitration normalization for the response.create observation seam.

This adapter is observational only and must not be used to drive runtime authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeAlias

from ai.interaction_lifecycle_policy import ResponseCreateDecision

if TYPE_CHECKING:
    from ai.realtime.response_create_runtime import (
        ResponseCreateExecutionDecision,
        ResponseCreatePreparedSnapshot,
    )


NormalizedOwnerScope: TypeAlias = Literal[
    "none",
    "same_turn_other_response",
    "active_response_same_canonical",
    "pending_tool_followup",
    "terminal_deliverable",
    "semantic_parent",
    "subsystem_local",
    "user_unresolved",
    "governance_hold",
]
NormalizedDeliverableStatus: TypeAlias = Literal[
    "none_observed",
    "pending_creation",
    "provisional_only",
    "progress_observed",
    "final_observed",
    "already_delivered",
    "non_deliverable",
    "blocked_terminal",
]
NormalizedDecisionDisposition: TypeAlias = Literal[
    "allow_now",
    "defer",
    "block",
    "drop",
    "clarify_hold",
    "observe_only",
]
NormalizedConfirmationHoldState: TypeAlias = Literal[
    "none",
    "confirmation_pending",
    "clarify_required",
    "governance_defer",
    "governance_block",
    "not_evaluated",
]
NormalizedTranscriptFinalState: TypeAlias = Literal[
    "not_applicable",
    "awaiting_transcript_final",
    "transcript_final_linked",
    "replacement_upgrade",
    "partial_only",
    "independent",
]
NormalizedParentCoverageState: TypeAlias = Literal[
    "not_applicable",
    "unknown",
    "uncovered",
    "covered_canonical",
    "covered_terminal_selection",
    "coverage_pending",
]
NormalizedCandidateId: TypeAlias = Literal[
    "same_turn_owner",
    "tool_lineage_guard",
    "canonical_terminal_state",
    "active_response",
    "audio_playback_busy",
    "canonical_audio_already_started",
    "single_flight_block",
    "already_delivered",
    "preference_recall_lock_blocked",
    "canonical_response_already_created",
    "preference_recall_suppressed",
    "awaiting_transcript_final",
    "direct_send",
]


class NormalizedObservationLogPayload(TypedDict):
    run_id: str
    turn_id: str
    input_event_key: str
    canonical_key: str
    policy_domain: str
    origin: str
    selected_candidate_id: NormalizedCandidateId
    decision_disposition: NormalizedDecisionDisposition
    native_outcome_action: str
    reason_code: str
    owner_scope: NormalizedOwnerScope
    deliverable_status: NormalizedDeliverableStatus
    confirmation_hold_state: NormalizedConfirmationHoldState
    transcript_final_state: NormalizedTranscriptFinalState
    parent_coverage_state: NormalizedParentCoverageState
    authority_retained_by: str
    candidate_count: int
    observational_only: bool
    normalization_warnings: tuple[str, ...]


@dataclass(frozen=True)
class NormalizedDecisionCandidate:
    candidate_id: NormalizedCandidateId
    observed: bool
    native_action: str | None
    native_reason_code: str | None
    owner_scope: NormalizedOwnerScope
    deliverable_status: NormalizedDeliverableStatus
    decision_disposition: NormalizedDecisionDisposition
    confirmation_hold_state: NormalizedConfirmationHoldState
    transcript_final_state: NormalizedTranscriptFinalState
    parent_coverage_state: NormalizedParentCoverageState
    authority_seam: str
    normalization_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedDecisionArtifact:
    native_outcome_action: str
    native_reason_code: str
    selected_candidate_id: NormalizedCandidateId
    decision_disposition: NormalizedDecisionDisposition
    owner_scope: NormalizedOwnerScope
    deliverable_status: NormalizedDeliverableStatus
    confirmation_hold_state: NormalizedConfirmationHoldState
    transcript_final_state: NormalizedTranscriptFinalState
    parent_coverage_state: NormalizedParentCoverageState
    authority_retained_by: str
    observational_only: bool
    normalization_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedArbitrationContext:
    run_id: str
    turn_id: str
    input_event_key: str
    canonical_key: str
    policy_domain: str
    origin: str
    authority_retained_by: str
    observational_only: bool


@dataclass(frozen=True)
class NormalizedArbitrationObservation:
    context: NormalizedArbitrationContext
    candidates: tuple[NormalizedDecisionCandidate, ...]
    decision: NormalizedDecisionArtifact
    normalization_warnings: tuple[str, ...] = ()

    def to_log_payload(self) -> NormalizedObservationLogPayload:
        return {
            "run_id": self.context.run_id,
            "turn_id": self.context.turn_id,
            "input_event_key": self.context.input_event_key,
            "canonical_key": self.context.canonical_key,
            "policy_domain": self.context.policy_domain,
            "origin": self.context.origin,
            "selected_candidate_id": self.decision.selected_candidate_id,
            "decision_disposition": self.decision.decision_disposition,
            "native_outcome_action": self.decision.native_outcome_action,
            "reason_code": self.decision.native_reason_code,
            "owner_scope": self.decision.owner_scope,
            "deliverable_status": self.decision.deliverable_status,
            "confirmation_hold_state": self.decision.confirmation_hold_state,
            "transcript_final_state": self.decision.transcript_final_state,
            "parent_coverage_state": self.decision.parent_coverage_state,
            "authority_retained_by": self.decision.authority_retained_by,
            "candidate_count": len(self.candidates),
            "observational_only": self.context.observational_only,
            "normalization_warnings": self.normalization_warnings,
        }


_AUTHORITY_SEAM = "ai.realtime.response_create_runtime"
_POLICY_DOMAIN = "response_create"


def _owner_scope_for_candidate(
    candidate_id: NormalizedCandidateId,
    *,
    same_turn_owner_reason: str | None,
) -> tuple[NormalizedOwnerScope, tuple[str, ...]]:
    warnings: list[str] = []
    if candidate_id == "same_turn_owner":
        if not same_turn_owner_reason:
            warnings.extend(("same_turn_owner_reason_unavailable", "owner_scope_ambiguous_same_turn_owner"))
            return "subsystem_local", tuple(warnings)
        reason = same_turn_owner_reason.strip().lower()
        if "tool_followup" in reason:
            return "pending_tool_followup", tuple(warnings)
        if "response" in reason or "owner" in reason:
            warnings.append("owner_scope_inferred_from_same_turn_reason")
            return "same_turn_other_response", tuple(warnings)
        if "terminal" in reason or "deliverable" in reason:
            return "terminal_deliverable", tuple(warnings)
        warnings.extend(("owner_scope_ambiguous_same_turn_owner", "owner_scope_inferred_from_same_turn_reason"))
        return "subsystem_local", tuple(warnings)
    if candidate_id == "canonical_terminal_state":
        return "terminal_deliverable", tuple(warnings)
    if candidate_id in {"already_delivered", "canonical_response_already_created"}:
        return "terminal_deliverable", tuple(warnings)
    if candidate_id == "awaiting_transcript_final":
        warnings.append("owner_scope_conservative_for_transcript_wait")
        return "subsystem_local", tuple(warnings)
    if candidate_id == "tool_lineage_guard":
        warnings.append("owner_scope_conservative_for_lineage_guard")
        return "subsystem_local", tuple(warnings)
    if candidate_id == "preference_recall_lock_blocked":
        return "subsystem_local", tuple(warnings)
    if candidate_id in {"active_response", "audio_playback_busy", "canonical_audio_already_started"}:
        return "active_response_same_canonical", tuple(warnings)
    if candidate_id == "single_flight_block":
        return "subsystem_local", tuple(warnings)
    if candidate_id == "preference_recall_suppressed":
        return "subsystem_local", tuple(warnings)
    if candidate_id == "direct_send":
        return "none", tuple(warnings)
    return "subsystem_local", ("owner_scope_defaulted",)


def _deliverable_status(
    candidate_id: NormalizedCandidateId,
    snapshot: ResponseCreatePreparedSnapshot,
) -> NormalizedDeliverableStatus:
    if candidate_id == "canonical_terminal_state":
        return "blocked_terminal"
    if candidate_id == "already_delivered":
        return "already_delivered"
    if candidate_id == "canonical_response_already_created":
        return "pending_creation"
    if snapshot.terminal_state_blocked:
        return "blocked_terminal"
    if snapshot.already_delivered:
        return "already_delivered"
    if snapshot.already_created_for_canonical_key:
        return "pending_creation"
    if snapshot.awaiting_transcript_final:
        return "provisional_only"
    return "none_observed"


def _confirmation_hold_state(snapshot: ResponseCreatePreparedSnapshot) -> NormalizedConfirmationHoldState:
    if snapshot.has_safety_override:
        return "none"
    return "not_evaluated"


def _transcript_final_state(snapshot: ResponseCreatePreparedSnapshot) -> NormalizedTranscriptFinalState:
    if snapshot.transcript_upgrade_replacement:
        return "replacement_upgrade"
    if snapshot.awaiting_transcript_final:
        return "awaiting_transcript_final"
    if snapshot.normalized_origin == "server_auto":
        return "partial_only"
    return "independent"


def _parent_coverage_state(
    snapshot: ResponseCreatePreparedSnapshot,
    candidate_id: NormalizedCandidateId,
) -> NormalizedParentCoverageState:
    if candidate_id == "awaiting_transcript_final":
        return "coverage_pending"
    if snapshot.same_turn_owner_present:
        return "covered_canonical"
    if snapshot.terminal_state_blocked or snapshot.already_delivered:
        return "covered_terminal_selection"
    if snapshot.tool_followup:
        return "covered_canonical" if snapshot.tool_followup_release else "unknown"
    return "not_applicable"


def _disposition_from_native(action: str | None) -> NormalizedDecisionDisposition:
    return {
        "SEND": "allow_now",
        "SCHEDULE": "defer",
        "BLOCK": "block",
        "DROP": "drop",
    }.get(action, "observe_only")


def _candidate_native_action(
    candidate_id: NormalizedCandidateId,
    lifecycle_decision: ResponseCreateDecision | None,
    execution_decision: ResponseCreateExecutionDecision,
) -> str | None:
    if candidate_id == execution_decision.selected_candidate_id:
        return execution_decision.action.value
    if lifecycle_decision is None:
        return None
    if candidate_id == lifecycle_decision.selected_candidate_id:
        return lifecycle_decision.action.value
    return None


def _candidate_native_reason(
    candidate_id: NormalizedCandidateId,
    lifecycle_decision: ResponseCreateDecision | None,
    execution_decision: ResponseCreateExecutionDecision,
) -> str | None:
    if candidate_id == execution_decision.selected_candidate_id:
        return execution_decision.reason_code
    if lifecycle_decision is None:
        return None
    if candidate_id == lifecycle_decision.selected_candidate_id:
        return lifecycle_decision.reason_code
    return None


def _observed_candidates(
    snapshot: ResponseCreatePreparedSnapshot,
    canonical_audio_started: bool | None,
) -> tuple[NormalizedCandidateId, ...]:
    ordered: list[tuple[NormalizedCandidateId, bool]] = [
        ("same_turn_owner", snapshot.same_turn_owner_present),
        ("tool_lineage_guard", not snapshot.lineage_allowed),
        ("canonical_terminal_state", snapshot.terminal_state_blocked),
        ("active_response", snapshot.response_in_flight),
        ("audio_playback_busy", snapshot.audio_playback_busy),
        ("canonical_audio_already_started", bool(canonical_audio_started)),
        ("single_flight_block", bool(snapshot.single_flight_block_reason)),
        ("already_delivered", snapshot.already_delivered),
        ("preference_recall_lock_blocked", snapshot.preference_recall_lock_blocked),
        ("canonical_response_already_created", snapshot.already_created_for_canonical_key),
        (
            "preference_recall_suppressed",
            snapshot.preference_recall_suppression_active and snapshot.normalized_origin == "server_auto",
        ),
        ("awaiting_transcript_final", snapshot.awaiting_transcript_final),
        ("direct_send", True),
    ]
    return tuple(candidate_id for candidate_id, observed in ordered if observed)




def _normalize_selected_candidate_id(candidate_id: str) -> NormalizedCandidateId:
    if candidate_id == "canonical_audio_started":
        return "canonical_audio_already_started"
    if candidate_id in {
        "same_turn_owner",
        "tool_lineage_guard",
        "canonical_terminal_state",
        "active_response",
        "audio_playback_busy",
        "canonical_audio_already_started",
        "single_flight_block",
        "already_delivered",
        "preference_recall_lock_blocked",
        "canonical_response_already_created",
        "preference_recall_suppressed",
        "awaiting_transcript_final",
        "direct_send",
    }:
        return candidate_id
    raise ValueError(f"Unsupported normalized candidate id: {candidate_id}")

def build_response_create_observation(
    *,
    snapshot: ResponseCreatePreparedSnapshot,
    execution_decision: ResponseCreateExecutionDecision,
    lifecycle_decision: ResponseCreateDecision | None,
    same_turn_owner_reason: str | None,
    canonical_audio_started: bool | None,
) -> NormalizedArbitrationObservation:
    warnings: list[str] = []
    if lifecycle_decision is None:
        warnings.append("lifecycle_decision_unavailable")
    if canonical_audio_started is None:
        warnings.append("canonical_audio_started_unavailable")
    if snapshot.same_turn_owner_present and not same_turn_owner_reason:
        warnings.append("same_turn_owner_reason_unavailable")

    candidates: list[NormalizedDecisionCandidate] = []
    for candidate_id in _observed_candidates(snapshot, canonical_audio_started):
        owner_scope, owner_warnings = _owner_scope_for_candidate(
            candidate_id,
            same_turn_owner_reason=same_turn_owner_reason,
        )
        native_action = _candidate_native_action(candidate_id, lifecycle_decision, execution_decision)
        native_reason = _candidate_native_reason(candidate_id, lifecycle_decision, execution_decision)
        candidate_warnings = list(owner_warnings)
        if native_action is None and candidate_id != "direct_send":
            candidate_warnings.append("native_mapping_unavailable")
        candidates.append(
            NormalizedDecisionCandidate(
                candidate_id=candidate_id,
                observed=True,
                native_action=native_action,
                native_reason_code=native_reason,
                owner_scope=owner_scope,
                deliverable_status=_deliverable_status(candidate_id, snapshot),
                decision_disposition=_disposition_from_native(native_action),
                confirmation_hold_state=_confirmation_hold_state(snapshot),
                transcript_final_state=_transcript_final_state(snapshot),
                parent_coverage_state=_parent_coverage_state(snapshot, candidate_id),
                authority_seam=_AUTHORITY_SEAM,
                normalization_warnings=tuple(candidate_warnings),
            )
        )

    selected_candidate_id: NormalizedCandidateId = _normalize_selected_candidate_id(
        execution_decision.selected_candidate_id
    )
    decision_owner_scope, decision_owner_warnings = _owner_scope_for_candidate(
        selected_candidate_id,
        same_turn_owner_reason=same_turn_owner_reason,
    )
    decision_warnings = list(decision_owner_warnings)
    if selected_candidate_id == "canonical_audio_started":
        decision_warnings.append("selected_candidate_id_normalized")
    decision = NormalizedDecisionArtifact(
        native_outcome_action=execution_decision.action.value,
        native_reason_code=execution_decision.reason_code,
        selected_candidate_id=selected_candidate_id,
        decision_disposition=_disposition_from_native(execution_decision.action.value),
        owner_scope=decision_owner_scope,
        deliverable_status=_deliverable_status(selected_candidate_id, snapshot),
        confirmation_hold_state=_confirmation_hold_state(snapshot),
        transcript_final_state=_transcript_final_state(snapshot),
        parent_coverage_state=_parent_coverage_state(snapshot, selected_candidate_id),
        authority_retained_by=_AUTHORITY_SEAM,
        observational_only=True,
        normalization_warnings=tuple(decision_warnings),
    )
    context = NormalizedArbitrationContext(
        run_id=snapshot.run_id,
        turn_id=snapshot.turn_id,
        input_event_key=snapshot.input_event_key or "unknown",
        canonical_key=snapshot.canonical_key,
        policy_domain=_POLICY_DOMAIN,
        origin=snapshot.normalized_origin,
        authority_retained_by=_AUTHORITY_SEAM,
        observational_only=True,
    )
    return NormalizedArbitrationObservation(
        context=context,
        candidates=tuple(candidates),
        decision=decision,
        normalization_warnings=tuple(warnings),
    )
