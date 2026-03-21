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
NormalizedToolFollowupOutcomePosture: TypeAlias = Literal[
    "released",
    "suppressed",
    "pruned",
    "stale",
    "pending",
    "observe_only",
]
NormalizedToolFollowupDistinctness: TypeAlias = Literal[
    "not_applicable",
    "distinct",
    "redundant",
    "stale",
    "unknown",
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
    "terminal_selected",
    "cancelled",
    "micro_ack_non_deliverable",
    "empty_tool_followup_non_deliverable",
    "provisional_empty_non_deliverable",
    "provisional_server_auto_awaiting_transcript_final",
    "tool_followup_precedence",
    "exact_phrase_obligation_open",
    "tool_output_descriptive_gesture_only",
    "semantic_owner_execution",
    "semantic_owner_parent",
    "tool_followup_uncovered",
    "tool_followup_parent_covered",
    "tool_followup_parent_pending",
    "tool_followup_parent_unknown",
    "tool_followup_pruned",
    "tool_followup_stale",
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
    followup_outcome_posture: NormalizedToolFollowupOutcomePosture | None
    followup_distinctness: NormalizedToolFollowupDistinctness | None
    child_canonical_key: str | None
    parent_canonical_key: str | None
    parent_semantic_owner_key: str | None
    blocked_by_parent_final_coverage: bool | None
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
    followup_outcome_posture: NormalizedToolFollowupOutcomePosture | None = None
    followup_distinctness: NormalizedToolFollowupDistinctness | None = None
    child_canonical_key: str | None = None
    parent_canonical_key: str | None = None
    parent_semantic_owner_key: str | None = None
    blocked_by_parent_final_coverage: bool | None = None
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
    followup_outcome_posture: NormalizedToolFollowupOutcomePosture | None = None
    followup_distinctness: NormalizedToolFollowupDistinctness | None = None
    child_canonical_key: str | None = None
    parent_canonical_key: str | None = None
    parent_semantic_owner_key: str | None = None
    blocked_by_parent_final_coverage: bool | None = None
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
            "followup_outcome_posture": self.decision.followup_outcome_posture,
            "followup_distinctness": self.decision.followup_distinctness,
            "child_canonical_key": self.decision.child_canonical_key,
            "parent_canonical_key": self.decision.parent_canonical_key,
            "parent_semantic_owner_key": self.decision.parent_semantic_owner_key,
            "blocked_by_parent_final_coverage": self.decision.blocked_by_parent_final_coverage,
            "authority_retained_by": self.decision.authority_retained_by,
            "candidate_count": len(self.candidates),
            "observational_only": self.context.observational_only,
            "normalization_warnings": self.normalization_warnings,
        }




class TurnArbitrationDiagnosticsLogPayload(TypedDict):
    run_id: str
    turn_id: str
    diagnostic_codes: tuple[str, ...]
    severity: str
    trace_complete: bool
    trace_partial: bool
    suspicious_mismatch_count: int
    repeated_warning_count: int
    vocabulary_alias_seen: bool
    observational_only: bool
    summary: str


@dataclass(frozen=True)
class TurnArbitrationDiagnostics:
    run_id: str
    turn_id: str
    diagnostic_codes: tuple[str, ...]
    severity: Literal["none", "info", "warning", "error"]
    trace_complete: bool
    trace_partial: bool
    suspicious_mismatch_count: int
    repeated_warning_count: int
    vocabulary_alias_seen: bool
    summary: str
    observational_only: bool = True

    def to_log_payload(self) -> TurnArbitrationDiagnosticsLogPayload:
        return {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "diagnostic_codes": self.diagnostic_codes,
            "severity": self.severity,
            "trace_complete": self.trace_complete,
            "trace_partial": self.trace_partial,
            "suspicious_mismatch_count": self.suspicious_mismatch_count,
            "repeated_warning_count": self.repeated_warning_count,
            "vocabulary_alias_seen": self.vocabulary_alias_seen,
            "observational_only": self.observational_only,
            "summary": self.summary,
        }

class TurnArbitrationTraceLogPayload(TypedDict):
    run_id: str
    turn_id: str
    initial_response_create_selected_candidate_id: str | None
    initial_response_create_disposition: str | None
    initial_response_create_reason_code: str | None
    terminal_selected_candidate_id: str | None
    terminal_deliverable_status: str | None
    terminal_selection_reason_code: str | None
    execution_canonical_key: str | None
    semantic_owner_canonical_key: str | None
    semantic_owner_reason_code: str | None
    semantic_owner_diverged: bool | None
    latest_tool_followup_selected_candidate_id: str | None
    latest_tool_followup_reason_code: str | None
    latest_tool_followup_parent_coverage_state: str | None
    latest_tool_followup_outcome_posture: str | None
    latest_tool_followup_distinctness: str | None
    transcript_final_state: str | None
    normalized_warning_count: int
    warning_codes: tuple[str, ...]
    authority_seams_seen: tuple[str, ...]
    trace_complete: bool
    trace_partial: bool


TurnReviewBucket: TypeAlias = Literal[
    "coherent",
    "partial_expected",
    "suspicious",
    "needs_review",
]


class TurnArbitrationReviewLogPayload(TypedDict):
    run_id: str
    turn_id: str
    review_bucket: TurnReviewBucket
    review_priority: str
    overall_verdict: str
    overall_summary: str
    response_create_summary: str
    terminal_summary: str
    semantic_owner_summary: str
    tool_followup_summary: str
    top_reasons: tuple[str, ...]
    notable_mismatches: tuple[str, ...]
    explainability_gaps: tuple[str, ...]
    suspicious_signals: tuple[str, ...]
    warning_codes: tuple[str, ...]
    diagnostic_codes: tuple[str, ...]
    trace_complete: bool
    trace_partial: bool
    observational_only: bool


@dataclass(frozen=True)
class TurnArbitrationTrace:
    run_id: str
    turn_id: str
    execution_canonical_key: str | None
    semantic_owner_canonical_key: str | None
    response_create_observation: NormalizedArbitrationObservation | None = None
    terminal_selection_observation: NormalizedArbitrationObservation | None = None
    semantic_owner_observation: NormalizedArbitrationObservation | None = None
    tool_followup_observations: tuple[NormalizedArbitrationObservation, ...] = ()
    warning_codes: tuple[str, ...] = ()
    authority_seams_seen: tuple[str, ...] = ()
    trace_complete: bool = False
    trace_partial: bool = True
    diagnostics: TurnArbitrationDiagnostics | None = None
    review_summary: TurnArbitrationReviewSummary | None = None

    @property
    def semantic_owner_diverged(self) -> bool | None:
        if (
            self.semantic_owner_observation is None
            or not self.execution_canonical_key
            or not self.semantic_owner_canonical_key
        ):
            return None
        return self.semantic_owner_canonical_key != self.execution_canonical_key

    @property
    def normalized_warning_count(self) -> int:
        return len(self.warning_codes)

    def to_log_payload(self) -> TurnArbitrationTraceLogPayload:
        response_create = self.response_create_observation.decision if self.response_create_observation else None
        terminal_selection = self.terminal_selection_observation.decision if self.terminal_selection_observation else None
        semantic_owner = self.semantic_owner_observation.decision if self.semantic_owner_observation else None
        tool_followup = self.tool_followup_observations[-1].decision if self.tool_followup_observations else None
        return {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "initial_response_create_selected_candidate_id": (
                response_create.selected_candidate_id if response_create else None
            ),
            "initial_response_create_disposition": (
                response_create.decision_disposition if response_create else None
            ),
            "initial_response_create_reason_code": (
                response_create.native_reason_code if response_create else None
            ),
            "terminal_selected_candidate_id": (
                terminal_selection.selected_candidate_id if terminal_selection else None
            ),
            "terminal_deliverable_status": (
                terminal_selection.deliverable_status if terminal_selection else None
            ),
            "terminal_selection_reason_code": (
                terminal_selection.native_reason_code if terminal_selection else None
            ),
            "execution_canonical_key": self.execution_canonical_key,
            "semantic_owner_canonical_key": self.semantic_owner_canonical_key,
            "semantic_owner_reason_code": (
                semantic_owner.native_reason_code if semantic_owner else None
            ),
            "semantic_owner_diverged": self.semantic_owner_diverged,
            "latest_tool_followup_selected_candidate_id": (
                tool_followup.selected_candidate_id if tool_followup else None
            ),
            "latest_tool_followup_reason_code": (
                tool_followup.native_reason_code if tool_followup else None
            ),
            "latest_tool_followup_parent_coverage_state": (
                tool_followup.parent_coverage_state if tool_followup else None
            ),
            "latest_tool_followup_outcome_posture": (
                tool_followup.followup_outcome_posture if tool_followup else None
            ),
            "latest_tool_followup_distinctness": (
                tool_followup.followup_distinctness if tool_followup else None
            ),
            "transcript_final_state": _trace_transcript_final_state(
                self.response_create_observation,
                self.terminal_selection_observation,
            ),
            "normalized_warning_count": self.normalized_warning_count,
            "warning_codes": self.warning_codes,
            "authority_seams_seen": self.authority_seams_seen,
            "trace_complete": self.trace_complete,
            "trace_partial": self.trace_partial,
        }


@dataclass(frozen=True)
class TurnArbitrationReviewSummary:
    run_id: str
    turn_id: str
    review_bucket: TurnReviewBucket
    review_priority: Literal["low", "medium", "high"]
    overall_verdict: str
    overall_summary: str
    response_create_summary: str
    terminal_summary: str
    semantic_owner_summary: str
    tool_followup_summary: str
    top_reasons: tuple[str, ...]
    notable_mismatches: tuple[str, ...]
    explainability_gaps: tuple[str, ...]
    suspicious_signals: tuple[str, ...]
    warning_codes: tuple[str, ...]
    diagnostic_codes: tuple[str, ...]
    trace_complete: bool
    trace_partial: bool
    observational_only: bool = True

    def to_log_payload(self) -> TurnArbitrationReviewLogPayload:
        return {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "review_bucket": self.review_bucket,
            "review_priority": self.review_priority,
            "overall_verdict": self.overall_verdict,
            "overall_summary": self.overall_summary,
            "response_create_summary": self.response_create_summary,
            "terminal_summary": self.terminal_summary,
            "semantic_owner_summary": self.semantic_owner_summary,
            "tool_followup_summary": self.tool_followup_summary,
            "top_reasons": self.top_reasons,
            "notable_mismatches": self.notable_mismatches,
            "explainability_gaps": self.explainability_gaps,
            "suspicious_signals": self.suspicious_signals,
            "warning_codes": self.warning_codes,
            "diagnostic_codes": self.diagnostic_codes,
            "trace_complete": self.trace_complete,
            "trace_partial": self.trace_partial,
            "observational_only": self.observational_only,
        }


_AUTHORITY_SEAM = "ai.realtime.response_create_runtime"
_POLICY_DOMAIN = "response_create"


_TERMINAL_SELECTION_AUTHORITY_SEAM = "ai.terminal_deliverable_arbitration.arbitrate_terminal_deliverable_selection"
_TERMINAL_SELECTION_POLICY_DOMAIN = "response_terminal_selection"
_SEMANTIC_OWNER_AUTHORITY_SEAM = "ai.semantic_owner_arbitration.decide_semantic_owner"
_SEMANTIC_OWNER_POLICY_DOMAIN = "response_semantic_owner"
_TOOL_FOLLOWUP_AUTHORITY_SEAM = "ai.realtime_api.tool_followup_observation"
_TOOL_FOLLOWUP_POLICY_DOMAIN = "tool_followup_release"


def _terminal_selection_candidate_id(selection_reason: str, *, selected: bool) -> NormalizedCandidateId:
    normalized_reason = str(selection_reason or "").strip().lower()
    mapping: dict[str, NormalizedCandidateId] = {
        "cancelled": "cancelled",
        "micro_ack_non_deliverable": "micro_ack_non_deliverable",
        "empty_tool_followup_non_deliverable": "empty_tool_followup_non_deliverable",
        "provisional_empty_non_deliverable": "provisional_empty_non_deliverable",
        "provisional_server_auto_awaiting_transcript_final": "provisional_server_auto_awaiting_transcript_final",
        "tool_followup_precedence": "tool_followup_precedence",
        "exact_phrase_obligation_open": "exact_phrase_obligation_open",
        "tool_output_descriptive_gesture_only": "tool_output_descriptive_gesture_only",
        "normal": "terminal_selected",
    }
    if normalized_reason in mapping:
        return mapping[normalized_reason]
    if selected:
        return "terminal_selected"
    raise ValueError(f"Unsupported terminal selection reason: {selection_reason}")


def _terminal_selection_deliverable_status(
    *,
    selected: bool,
    selection_reason: str,
    active_response_was_provisional: bool,
) -> tuple[NormalizedDeliverableStatus, tuple[str, ...]]:
    normalized_reason = str(selection_reason or "").strip().lower()
    warnings: list[str] = []
    if selected:
        return "final_observed", tuple(warnings)
    if normalized_reason in {"micro_ack_non_deliverable", "empty_tool_followup_non_deliverable", "provisional_empty_non_deliverable", "tool_output_descriptive_gesture_only", "cancelled"}:
        return "non_deliverable", tuple(warnings)
    if normalized_reason in {"provisional_server_auto_awaiting_transcript_final"}:
        return "provisional_only", tuple(warnings)
    if normalized_reason in {"exact_phrase_obligation_open", "tool_followup_precedence"}:
        return "blocked_terminal", tuple(warnings)
    warnings.append("deliverable_status_conservative_default")
    return ("provisional_only" if active_response_was_provisional else "none_observed"), tuple(warnings)


def _terminal_selection_transcript_state(
    *,
    transcript_final_seen: bool,
    selection_reason: str,
    active_response_was_provisional: bool,
) -> tuple[NormalizedTranscriptFinalState, tuple[str, ...]]:
    normalized_reason = str(selection_reason or "").strip().lower()
    warnings: list[str] = []
    if transcript_final_seen:
        return "transcript_final_linked", tuple(warnings)
    if normalized_reason == "provisional_server_auto_awaiting_transcript_final":
        return "awaiting_transcript_final", tuple(warnings)
    if active_response_was_provisional:
        return "partial_only", tuple(warnings)
    return "not_applicable", tuple(warnings)


def build_terminal_selection_observation(
    *,
    run_id: str,
    turn_id: str,
    input_event_key: str | None,
    canonical_key: str,
    origin: str,
    selected: bool,
    selection_reason: str,
    selected_candidate_id: str | None = None,
    transcript_final_seen: bool,
    active_response_was_provisional: bool,
) -> NormalizedArbitrationObservation:
    expected_candidate_id = _terminal_selection_candidate_id(selection_reason, selected=selected)
    decision_warnings: list[str] = []
    if selected_candidate_id is None:
        candidate_id = expected_candidate_id
    else:
        normalized_candidate_id = _normalize_selected_candidate_id(selected_candidate_id)
        if normalized_candidate_id != expected_candidate_id:
            candidate_id = expected_candidate_id
            decision_warnings.append(
                f"terminal_selected_candidate_id_mismatch_ignored:{normalized_candidate_id}->{expected_candidate_id}"
            )
        else:
            candidate_id = normalized_candidate_id
    deliverable_status, deliverable_warnings = _terminal_selection_deliverable_status(
        selected=selected,
        selection_reason=selection_reason,
        active_response_was_provisional=active_response_was_provisional,
    )
    transcript_final_state, transcript_warnings = _terminal_selection_transcript_state(
        transcript_final_seen=transcript_final_seen,
        selection_reason=selection_reason,
        active_response_was_provisional=active_response_was_provisional,
    )
    decision_disposition: NormalizedDecisionDisposition = "allow_now" if selected else "observe_only"
    owner_scope: NormalizedOwnerScope = "terminal_deliverable" if selected else "subsystem_local"
    decision_warnings = tuple([*decision_warnings, *deliverable_warnings, *transcript_warnings])
    decision = NormalizedDecisionArtifact(
        native_outcome_action="SELECT" if selected else "REJECT",
        native_reason_code=str(selection_reason or ""),
        selected_candidate_id=candidate_id,
        decision_disposition=decision_disposition,
        owner_scope=owner_scope,
        deliverable_status=deliverable_status,
        confirmation_hold_state="not_evaluated",
        transcript_final_state=transcript_final_state,
        parent_coverage_state="not_applicable",
        authority_retained_by=_TERMINAL_SELECTION_AUTHORITY_SEAM,
        observational_only=True,
        normalization_warnings=decision_warnings,
    )
    context = NormalizedArbitrationContext(
        run_id=run_id,
        turn_id=turn_id,
        input_event_key=str(input_event_key or "unknown"),
        canonical_key=canonical_key,
        policy_domain=_TERMINAL_SELECTION_POLICY_DOMAIN,
        origin=str(origin or "").strip().lower(),
        authority_retained_by=_TERMINAL_SELECTION_AUTHORITY_SEAM,
        observational_only=True,
    )
    return NormalizedArbitrationObservation(
        context=context,
        candidates=(NormalizedDecisionCandidate(
            candidate_id=candidate_id,
            observed=True,
            native_action=decision.native_outcome_action,
            native_reason_code=decision.native_reason_code,
            owner_scope=decision.owner_scope,
            deliverable_status=decision.deliverable_status,
            decision_disposition=decision.decision_disposition,
            confirmation_hold_state=decision.confirmation_hold_state,
            transcript_final_state=decision.transcript_final_state,
            parent_coverage_state=decision.parent_coverage_state,
            authority_seam=_TERMINAL_SELECTION_AUTHORITY_SEAM,
            normalization_warnings=decision_warnings,
        ),),
        decision=decision,
        normalization_warnings=decision_warnings,
    )


def build_semantic_owner_observation(
    *,
    run_id: str,
    turn_id: str,
    input_event_key: str | None,
    execution_canonical_key: str,
    semantic_owner_canonical_key: str,
    origin: str,
    selected: bool,
    selection_reason: str,
    parent_turn_id: str | None = None,
    parent_input_event_key: str | None = None,
    selected_candidate_id: str | None = None,
    native_reason_code: str | None = None,
) -> NormalizedArbitrationObservation:
    warnings: list[str] = []
    normalized_reason = str(selection_reason or "").strip().lower()
    normalized_native_reason = str(native_reason_code if native_reason_code is not None else selection_reason or "").strip().lower()
    expected_candidate_id: NormalizedCandidateId
    owner_scope: NormalizedOwnerScope
    parent_coverage_state: NormalizedParentCoverageState
    if semantic_owner_canonical_key == execution_canonical_key:
        expected_candidate_id = "semantic_owner_execution"
        owner_scope = "none" if selected else "subsystem_local"
        parent_coverage_state = "not_applicable"
        if (
            selected
            and normalized_reason == "normal"
            and str(origin or "").strip().lower() == "tool_output"
            and normalized_native_reason in {"", "normal"}
        ):
            warnings.append("semantic_owner_parent_not_promoted")
    else:
        expected_candidate_id = "semantic_owner_parent"
        owner_scope = "semantic_parent"
        parent_coverage_state = "covered_canonical"
    if selected_candidate_id is None:
        candidate_id = expected_candidate_id
    else:
        normalized_candidate_id = _normalize_selected_candidate_id(selected_candidate_id)
        if normalized_candidate_id != expected_candidate_id:
            candidate_id = expected_candidate_id
            warnings.append(
                f"semantic_owner_selected_candidate_id_mismatch_ignored:{normalized_candidate_id}->{expected_candidate_id}"
            )
        else:
            candidate_id = normalized_candidate_id
    if not selected:
        warnings.append("semantic_owner_resolution_without_terminal_selection")
    if not semantic_owner_canonical_key:
        warnings.append("semantic_owner_canonical_key_unavailable")
    if candidate_id == "semantic_owner_parent" and (not parent_turn_id or not parent_input_event_key):
        warnings.append("semantic_parent_lineage_unavailable")
    decision = NormalizedDecisionArtifact(
        native_outcome_action="RETAIN" if candidate_id == "semantic_owner_execution" else "REASSIGN",
        native_reason_code=str(native_reason_code if native_reason_code is not None else selection_reason or ""),
        selected_candidate_id=candidate_id,
        decision_disposition="observe_only",
        owner_scope=owner_scope,
        deliverable_status="final_observed" if selected else "none_observed",
        confirmation_hold_state="not_evaluated",
        transcript_final_state="not_applicable",
        parent_coverage_state=parent_coverage_state,
        authority_retained_by=_SEMANTIC_OWNER_AUTHORITY_SEAM,
        observational_only=True,
        normalization_warnings=tuple(warnings),
    )
    context = NormalizedArbitrationContext(
        run_id=run_id,
        turn_id=turn_id,
        input_event_key=str(input_event_key or "unknown"),
        canonical_key=execution_canonical_key,
        policy_domain=_SEMANTIC_OWNER_POLICY_DOMAIN,
        origin=str(origin or "").strip().lower(),
        authority_retained_by=_SEMANTIC_OWNER_AUTHORITY_SEAM,
        observational_only=True,
    )
    return NormalizedArbitrationObservation(
        context=context,
        candidates=(NormalizedDecisionCandidate(
            candidate_id=candidate_id,
            observed=True,
            native_action=decision.native_outcome_action,
            native_reason_code=decision.native_reason_code,
            owner_scope=decision.owner_scope,
            deliverable_status=decision.deliverable_status,
            decision_disposition=decision.decision_disposition,
            confirmation_hold_state=decision.confirmation_hold_state,
            transcript_final_state=decision.transcript_final_state,
            parent_coverage_state=decision.parent_coverage_state,
            authority_seam=_SEMANTIC_OWNER_AUTHORITY_SEAM,
            normalization_warnings=tuple(warnings),
        ),),
        decision=decision,
        normalization_warnings=tuple(warnings),
    )


def _tool_followup_candidate_id(
    *,
    parent_coverage_state: NormalizedParentCoverageState,
    followup_outcome_posture: NormalizedToolFollowupOutcomePosture,
) -> NormalizedCandidateId:
    if followup_outcome_posture == "pruned":
        return "tool_followup_pruned"
    if followup_outcome_posture == "stale":
        return "tool_followup_stale"
    if parent_coverage_state == "coverage_pending":
        return "tool_followup_parent_pending"
    if parent_coverage_state in {"covered_canonical", "covered_terminal_selection"}:
        return "tool_followup_parent_covered"
    if parent_coverage_state == "uncovered":
        return "tool_followup_uncovered"
    return "tool_followup_parent_unknown"


def build_tool_followup_observation(
    *,
    run_id: str,
    turn_id: str,
    input_event_key: str | None,
    canonical_key: str,
    origin: str,
    parent_coverage_state: NormalizedParentCoverageState,
    followup_outcome_posture: NormalizedToolFollowupOutcomePosture,
    native_reason_code: str,
    native_outcome_action: str,
    followup_distinctness: NormalizedToolFollowupDistinctness = "unknown",
    parent_canonical_key: str | None = None,
    parent_semantic_owner_key: str | None = None,
    blocked_by_parent_final_coverage: bool | None = None,
    authority_seam: str = _TOOL_FOLLOWUP_AUTHORITY_SEAM,
    normalization_warnings: tuple[str, ...] = (),
) -> NormalizedArbitrationObservation:
    candidate_id = _tool_followup_candidate_id(
        parent_coverage_state=parent_coverage_state,
        followup_outcome_posture=followup_outcome_posture,
    )
    decision_disposition: NormalizedDecisionDisposition = (
        "allow_now" if followup_outcome_posture == "released"
        else "defer" if followup_outcome_posture == "pending"
        else "observe_only"
    )
    decision = NormalizedDecisionArtifact(
        native_outcome_action=native_outcome_action,
        native_reason_code=str(native_reason_code or ""),
        selected_candidate_id=candidate_id,
        decision_disposition=decision_disposition,
        owner_scope="pending_tool_followup",
        deliverable_status="none_observed",
        confirmation_hold_state="not_evaluated",
        transcript_final_state="not_applicable",
        parent_coverage_state=parent_coverage_state,
        followup_outcome_posture=followup_outcome_posture,
        followup_distinctness=followup_distinctness,
        child_canonical_key=canonical_key,
        parent_canonical_key=str(parent_canonical_key or "").strip() or None,
        parent_semantic_owner_key=str(parent_semantic_owner_key or "").strip() or None,
        blocked_by_parent_final_coverage=blocked_by_parent_final_coverage,
        authority_retained_by=authority_seam,
        observational_only=True,
        normalization_warnings=normalization_warnings,
    )
    context = NormalizedArbitrationContext(
        run_id=run_id,
        turn_id=turn_id,
        input_event_key=str(input_event_key or "unknown"),
        canonical_key=canonical_key,
        policy_domain=_TOOL_FOLLOWUP_POLICY_DOMAIN,
        origin=str(origin or "").strip().lower(),
        authority_retained_by=authority_seam,
        observational_only=True,
    )
    candidate = NormalizedDecisionCandidate(
        candidate_id=candidate_id,
        observed=True,
        native_action=native_outcome_action,
        native_reason_code=str(native_reason_code or ""),
        owner_scope=decision.owner_scope,
        deliverable_status=decision.deliverable_status,
        decision_disposition=decision.decision_disposition,
        confirmation_hold_state=decision.confirmation_hold_state,
        transcript_final_state=decision.transcript_final_state,
        parent_coverage_state=decision.parent_coverage_state,
        followup_outcome_posture=decision.followup_outcome_posture,
        followup_distinctness=decision.followup_distinctness,
        child_canonical_key=decision.child_canonical_key,
        parent_canonical_key=decision.parent_canonical_key,
        parent_semantic_owner_key=decision.parent_semantic_owner_key,
        blocked_by_parent_final_coverage=decision.blocked_by_parent_final_coverage,
        authority_seam=authority_seam,
        normalization_warnings=normalization_warnings,
    )
    return NormalizedArbitrationObservation(
        context=context,
        candidates=(candidate,),
        decision=decision,
        normalization_warnings=normalization_warnings,
    )


def _trace_transcript_final_state(
    response_create_observation: NormalizedArbitrationObservation | None,
    terminal_selection_observation: NormalizedArbitrationObservation | None,
) -> NormalizedTranscriptFinalState | None:
    if terminal_selection_observation is not None:
        return terminal_selection_observation.decision.transcript_final_state
    if response_create_observation is not None:
        return response_create_observation.decision.transcript_final_state
    return None


def _merge_warning_codes(
    *observations: NormalizedArbitrationObservation | None,
) -> tuple[str, ...]:
    merged: dict[str, None] = {}
    for observation in observations:
        if observation is None:
            continue
        for warning_code in (*observation.normalization_warnings, *observation.decision.normalization_warnings):
            merged.setdefault(warning_code, None)
    return tuple(merged)


def _merge_authority_seams(
    *observations: NormalizedArbitrationObservation | None,
) -> tuple[str, ...]:
    merged: dict[str, None] = {}
    for observation in observations:
        if observation is None:
            continue
        merged.setdefault(observation.context.authority_retained_by, None)
    return tuple(merged)


def merge_arbitration_observations_for_turn(
    *,
    existing_trace: TurnArbitrationTrace | None = None,
    response_create_observation: NormalizedArbitrationObservation | None = None,
    terminal_selection_observation: NormalizedArbitrationObservation | None = None,
    semantic_owner_observation: NormalizedArbitrationObservation | None = None,
    tool_followup_observation: NormalizedArbitrationObservation | None = None,
    semantic_owner_canonical_key: str | None = None,
) -> TurnArbitrationTrace:
    merged_response_create = response_create_observation or (
        existing_trace.response_create_observation if existing_trace is not None else None
    )
    merged_terminal_selection = terminal_selection_observation or (
        existing_trace.terminal_selection_observation if existing_trace is not None else None
    )
    merged_semantic_owner = semantic_owner_observation or (
        existing_trace.semantic_owner_observation if existing_trace is not None else None
    )
    existing_tool_followups = getattr(existing_trace, "tool_followup_observations", ()) if existing_trace is not None else ()
    merged_tool_followups = existing_tool_followups + ((tool_followup_observation,) if tool_followup_observation is not None else ())
    observations = tuple(
        observation
        for observation in (
            merged_response_create,
            merged_terminal_selection,
            merged_semantic_owner,
            *merged_tool_followups,
        )
        if observation is not None
    )
    if not observations:
        raise ValueError("At least one arbitration observation is required to build a turn trace.")
    run_id = next((observation.context.run_id for observation in observations if observation.context.run_id), "")
    turn_id = next((observation.context.turn_id for observation in observations if observation.context.turn_id), "")
    execution_canonical_key = next(
        (
            observation.context.canonical_key
            for observation in (
                merged_response_create,
                merged_terminal_selection,
                merged_semantic_owner,
            )
            if observation is not None and observation.context.canonical_key
        ),
        existing_trace.execution_canonical_key if existing_trace is not None else None,
    )
    if semantic_owner_canonical_key is None and existing_trace is not None:
        semantic_owner_canonical_key = existing_trace.semantic_owner_canonical_key
    if semantic_owner_canonical_key is None and merged_semantic_owner is not None:
        if merged_semantic_owner.decision.selected_candidate_id == "semantic_owner_execution":
            semantic_owner_canonical_key = execution_canonical_key
    warning_codes = _merge_warning_codes(
        merged_response_create,
        merged_terminal_selection,
        merged_semantic_owner,
        *merged_tool_followups,
    )
    authority_seams_seen = _merge_authority_seams(
        merged_response_create,
        merged_terminal_selection,
        merged_semantic_owner,
        *merged_tool_followups,
    )
    trace_complete = all(
        observation is not None
        for observation in (
            merged_response_create,
            merged_terminal_selection,
            merged_semantic_owner,
        )
    )
    trace = TurnArbitrationTrace(
        run_id=run_id,
        turn_id=turn_id,
        execution_canonical_key=execution_canonical_key,
        semantic_owner_canonical_key=semantic_owner_canonical_key,
        response_create_observation=merged_response_create,
        terminal_selection_observation=merged_terminal_selection,
        semantic_owner_observation=merged_semantic_owner,
        tool_followup_observations=merged_tool_followups,
        warning_codes=warning_codes,
        authority_seams_seen=authority_seams_seen,
        trace_complete=trace_complete,
        trace_partial=not trace_complete,
    )
    diagnostics = build_turn_arbitration_diagnostics(trace)
    review_summary = build_turn_review_summary(
        TurnArbitrationTrace(
            **{
                **trace.__dict__,
                "diagnostics": diagnostics,
            }
        )
    )
    return TurnArbitrationTrace(
        **{
            **trace.__dict__,
            "diagnostics": diagnostics,
            "review_summary": review_summary,
        }
    )


def build_turn_arbitration_diagnostics(trace: TurnArbitrationTrace) -> TurnArbitrationDiagnostics:
    diagnostic_codes: list[str] = []
    suspicious_mismatch_count = 0
    repeated_warning_count = 0

    if trace.trace_partial:
        diagnostic_codes.append("trace_partial")
        if trace.response_create_observation is not None and trace.terminal_selection_observation is None:
            diagnostic_codes.append("expected_terminal_selection_missing")
        if trace.terminal_selection_observation is not None and trace.semantic_owner_observation is None:
            diagnostic_codes.append("expected_semantic_owner_missing")

    response_create = trace.response_create_observation.decision if trace.response_create_observation else None
    terminal = trace.terminal_selection_observation.decision if trace.terminal_selection_observation else None
    semantic_owner = trace.semantic_owner_observation.decision if trace.semantic_owner_observation else None

    if terminal is not None and semantic_owner is None and terminal.deliverable_status == "final_observed":
        diagnostic_codes.append("terminal_selected_without_semantic_owner")
        suspicious_mismatch_count += 1
    if semantic_owner is not None and _semantic_owner_has_explicit_parent_promotion(trace):
        if _semantic_owner_parent_promotion_is_expected(trace):
            diagnostic_codes.append("semantic_owner_parent_promotion_expected")
        else:
            diagnostic_codes.append("semantic_owner_diverged")
            suspicious_mismatch_count += 1
    if terminal is not None and terminal.selected_candidate_id == "terminal_selected" and terminal.deliverable_status != "final_observed":
        diagnostic_codes.append("terminal_selected_non_final_status")
        suspicious_mismatch_count += 1
    if response_create is not None and terminal is not None:
        if response_create.decision_disposition == "block" and terminal.deliverable_status == "final_observed":
            diagnostic_codes.append("response_create_blocked_but_terminal_final")
            suspicious_mismatch_count += 1
        if response_create.decision_disposition == "drop" and terminal.selected_candidate_id == "terminal_selected":
            diagnostic_codes.append("response_create_dropped_but_terminal_selected")
            suspicious_mismatch_count += 1
        if response_create.decision_disposition == "defer" and terminal.deliverable_status == "final_observed" and response_create.transcript_final_state == "awaiting_transcript_final":
            diagnostic_codes.append("deferred_create_then_final_terminal")

    tool_followups = [observation.decision for observation in trace.tool_followup_observations]
    if tool_followups:
        if any(
            observation.followup_outcome_posture == "suppressed"
            and observation.parent_coverage_state == "unknown"
            for observation in tool_followups
        ):
            diagnostic_codes.append("tool_followup_suppressed_with_unknown_parent_coverage")
            suspicious_mismatch_count += 1
        if any(
            observation.followup_outcome_posture == "released"
            and observation.parent_coverage_state in {"covered_canonical", "covered_terminal_selection"}
            and observation.blocked_by_parent_final_coverage is True
            for observation in tool_followups
        ):
            diagnostic_codes.append("tool_followup_released_despite_parent_final_coverage")
            suspicious_mismatch_count += 1
        if any(observation.parent_coverage_state == "coverage_pending" for observation in tool_followups):
            diagnostic_codes.append("tool_followup_coverage_pending_observed")
        if any(
            observation.followup_outcome_posture in {"pruned", "stale"}
            and observation.native_reason_code in {"", "unknown"}
            for observation in tool_followups
        ):
            diagnostic_codes.append("tool_followup_pruned_with_weak_reason")
        if any(
            observation.parent_coverage_state == "covered_terminal_selection"
            and observation.followup_outcome_posture == "released"
            for observation in tool_followups
        ):
            diagnostic_codes.append("tool_followup_terminal_selection_disagreement")

    conservative_warning_count = sum(1 for code in trace.warning_codes if "conservative" in code or "ambiguous" in code)
    if conservative_warning_count:
        diagnostic_codes.append("conservative_mapping_present")
    if trace.normalized_warning_count > 1:
        repeated_warning_count = trace.normalized_warning_count
        diagnostic_codes.append("repeated_normalization_warnings")
    if any("alias" in code or "normalized_from_" in code for code in trace.warning_codes):
        diagnostic_codes.append("vocabulary_alias_detected")

    if terminal is not None and terminal.deliverable_status == "final_observed" and semantic_owner is None:
        diagnostic_codes.append("explainability_gap_final_without_owner")
    if response_create is not None and response_create.decision_disposition == "allow_now" and terminal is None:
        diagnostic_codes.append("explainability_gap_runtime_resolved_without_terminal")

    if not diagnostic_codes:
        diagnostic_codes.append("trace_coherent")

    if suspicious_mismatch_count:
        severity = "warning" if suspicious_mismatch_count == 1 else "error"
    elif trace.trace_partial or repeated_warning_count or conservative_warning_count:
        severity = "info" if not trace.trace_complete else "warning"
    else:
        severity = "none"

    summary = (
        "coherent" if diagnostic_codes == ["trace_coherent"] else ",".join(diagnostic_codes)
    )
    return TurnArbitrationDiagnostics(
        run_id=trace.run_id,
        turn_id=trace.turn_id,
        diagnostic_codes=tuple(dict.fromkeys(diagnostic_codes)),
        severity=severity,
        trace_complete=trace.trace_complete,
        trace_partial=trace.trace_partial,
        suspicious_mismatch_count=suspicious_mismatch_count,
        repeated_warning_count=repeated_warning_count,
        vocabulary_alias_seen="vocabulary_alias_detected" in diagnostic_codes,
        summary=summary,
    )


def summarize_turn_arbitration_diagnostics(diagnostics: TurnArbitrationDiagnostics) -> TurnArbitrationDiagnosticsLogPayload:
    return diagnostics.to_log_payload()


def summarize_turn_arbitration_trace(trace: TurnArbitrationTrace) -> TurnArbitrationTraceLogPayload:
    return trace.to_log_payload()


def _semantic_owner_has_explicit_parent_promotion(trace: TurnArbitrationTrace) -> bool:
    decision = trace.semantic_owner_observation.decision if trace.semantic_owner_observation else None
    if decision is None:
        return False
    if decision.selected_candidate_id != "semantic_owner_parent":
        return False
    return str(decision.native_reason_code or "").strip().lower() == "parent_promoted_from_tool_output"


def _semantic_owner_parent_promotion_is_expected(trace: TurnArbitrationTrace) -> bool:
    if not _semantic_owner_has_explicit_parent_promotion(trace):
        return False
    terminal_observation = trace.terminal_selection_observation
    terminal = terminal_observation.decision if terminal_observation else None
    if terminal is None:
        return False
    if terminal.selected_candidate_id != "terminal_selected":
        return False
    if str(terminal.native_reason_code or "").strip().lower() != "normal":
        return False
    if str(terminal_observation.context.origin or "").strip().lower() != "tool_output":
        return False
    promoted_parent_key = str(trace.semantic_owner_canonical_key or "").strip()
    if not promoted_parent_key:
        return False
    for observation in reversed(trace.tool_followup_observations):
        decision = observation.decision
        if decision.followup_outcome_posture != "released":
            continue
        distinctness = str(decision.followup_distinctness or "").strip().lower()
        native_reason = str(decision.native_reason_code or "").strip().lower()
        parent_coverage_state = str(decision.parent_coverage_state or "").strip().lower()
        if parent_coverage_state == "uncovered":
            if distinctness == "unknown":
                if native_reason != "released_for_followup_delivery":
                    continue
            elif distinctness == "redundant":
                if native_reason != "parent_not_deliverable":
                    continue
            elif distinctness != "distinct":
                continue
        elif parent_coverage_state == "not_applicable":
            if native_reason not in {"not_suppressible", "non_gesture_tool", "distinct_info"}:
                continue
            if distinctness not in {"unknown", "distinct"}:
                continue
        else:
            continue
        parent_semantic_owner_key = str(decision.parent_semantic_owner_key or "").strip()
        parent_canonical_key = str(decision.parent_canonical_key or "").strip()
        if promoted_parent_key and promoted_parent_key in {parent_semantic_owner_key, parent_canonical_key}:
            return True
    return False


def _review_bucket_for_trace(trace: TurnArbitrationTrace) -> TurnReviewBucket:
    diagnostics = trace.diagnostics
    if diagnostics is None:
        return "needs_review"
    if "explainability_gap_final_without_owner" in diagnostics.diagnostic_codes:
        return "needs_review"
    if trace.trace_partial and diagnostics.suspicious_mismatch_count == 0:
        return "partial_expected"
    if diagnostics.suspicious_mismatch_count > 0:
        return "suspicious"
    if any(code.startswith("explainability_gap_") for code in diagnostics.diagnostic_codes):
        return "needs_review"
    return "coherent"


def _review_priority_for_bucket(bucket: TurnReviewBucket) -> Literal["low", "medium", "high"]:
    if bucket == "coherent":
        return "low"
    if bucket == "partial_expected":
        return "medium"
    return "high"


def _response_create_summary(trace: TurnArbitrationTrace) -> str:
    decision = trace.response_create_observation.decision if trace.response_create_observation else None
    if decision is None:
        return "response.create seam missing"
    if decision.decision_disposition == "allow_now":
        return f"response.create allowed ({decision.selected_candidate_id})"
    if decision.decision_disposition == "defer":
        return f"response.create deferred ({decision.native_reason_code or decision.selected_candidate_id})"
    if decision.decision_disposition == "block":
        return f"response.create blocked ({decision.native_reason_code or decision.selected_candidate_id})"
    if decision.decision_disposition == "drop":
        return f"response.create dropped ({decision.native_reason_code or decision.selected_candidate_id})"
    return f"response.create observed ({decision.selected_candidate_id})"


def _terminal_summary(trace: TurnArbitrationTrace) -> str:
    decision = trace.terminal_selection_observation.decision if trace.terminal_selection_observation else None
    if decision is None:
        return "terminal seam missing"
    if decision.selected_candidate_id == "terminal_selected":
        return "terminal deliverable selected"
    return f"terminal not selected ({decision.native_reason_code or decision.selected_candidate_id})"


def _semantic_owner_summary(trace: TurnArbitrationTrace) -> str:
    decision = trace.semantic_owner_observation.decision if trace.semantic_owner_observation else None
    if decision is None:
        return "semantic owner seam missing"
    if _semantic_owner_parent_promotion_is_expected(trace):
        return "semantic owner promoted to parent after tool followup delivery"
    if _semantic_owner_has_explicit_parent_promotion(trace):
        return "semantic owner diverged to parent"
    if decision.selected_candidate_id == "semantic_owner_execution":
        reason_code = str(decision.native_reason_code or "").strip().lower()
        if reason_code == "terminal_not_selected":
            return "semantic owner remained execution-scoped pending terminal selection"
        return "semantic owner stayed on execution canonical"
    return f"semantic owner observed ({decision.selected_candidate_id})"


def _tool_followup_summary(trace: TurnArbitrationTrace) -> str:
    if not trace.tool_followup_observations:
        return "no tool followup observations"
    latest = trace.tool_followup_observations[-1].decision
    return (
        f"tool followup {latest.followup_outcome_posture} "
        f"(parent={latest.parent_coverage_state}, distinctness={latest.followup_distinctness})"
    )


def _collect_explainability_gaps(trace: TurnArbitrationTrace) -> tuple[str, ...]:
    diagnostics = trace.diagnostics
    if diagnostics is None:
        return ("diagnostics_unavailable",)
    gaps: list[str] = []
    if "expected_terminal_selection_missing" in diagnostics.diagnostic_codes:
        gaps.append("terminal selection seam missing after response.create observation")
    if "expected_semantic_owner_missing" in diagnostics.diagnostic_codes:
        gaps.append("semantic owner seam missing after terminal observation")
    if "explainability_gap_final_without_owner" in diagnostics.diagnostic_codes:
        gaps.append("final terminal selected without semantic owner explanation")
    if "explainability_gap_runtime_resolved_without_terminal" in diagnostics.diagnostic_codes:
        gaps.append("runtime resolved response.create without terminal explanation")
    return tuple(gaps)


def _collect_notable_mismatches(trace: TurnArbitrationTrace) -> tuple[str, ...]:
    diagnostics = trace.diagnostics
    if diagnostics is None:
        return ()
    mismatches: list[str] = []
    if "semantic_owner_diverged" in diagnostics.diagnostic_codes:
        mismatches.append("semantic owner diverged from execution canonical")
    if "response_create_blocked_but_terminal_final" in diagnostics.diagnostic_codes:
        mismatches.append("blocked response.create still ended with final terminal")
    if "response_create_dropped_but_terminal_selected" in diagnostics.diagnostic_codes:
        mismatches.append("dropped response.create still ended with terminal selection")
    if "tool_followup_terminal_selection_disagreement" in diagnostics.diagnostic_codes:
        mismatches.append("tool followup released despite terminal-coverage disagreement")
    return tuple(mismatches)


def _collect_suspicious_signals(trace: TurnArbitrationTrace) -> tuple[str, ...]:
    diagnostics = trace.diagnostics
    if diagnostics is None:
        return ("diagnostics_unavailable",)
    suspicious: list[str] = []
    if "tool_followup_suppressed_with_unknown_parent_coverage" in diagnostics.diagnostic_codes:
        suspicious.append("tool followup suppressed while parent coverage stayed unknown")
    if "tool_followup_released_despite_parent_final_coverage" in diagnostics.diagnostic_codes:
        suspicious.append("tool followup released despite parent final coverage")
    if "terminal_selected_without_semantic_owner" in diagnostics.diagnostic_codes:
        suspicious.append("terminal selected without semantic owner correlation")
    if "conservative_mapping_present" in diagnostics.diagnostic_codes:
        suspicious.append("conservative normalization warning present")
    return tuple(suspicious)


def _top_reasons(trace: TurnArbitrationTrace) -> tuple[str, ...]:
    reasons: list[str] = []
    for decision in (
        trace.response_create_observation.decision if trace.response_create_observation else None,
        trace.terminal_selection_observation.decision if trace.terminal_selection_observation else None,
        trace.semantic_owner_observation.decision if trace.semantic_owner_observation else None,
        trace.tool_followup_observations[-1].decision if trace.tool_followup_observations else None,
    ):
        if decision is None:
            continue
        reason = str(decision.native_reason_code or decision.selected_candidate_id).strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    return tuple(reasons[:4])


def build_turn_review_summary(trace: TurnArbitrationTrace) -> TurnArbitrationReviewSummary:
    bucket = _review_bucket_for_trace(trace)
    explainability_gaps = _collect_explainability_gaps(trace)
    notable_mismatches = _collect_notable_mismatches(trace)
    suspicious_signals = _collect_suspicious_signals(trace)
    response_summary = _response_create_summary(trace)
    terminal_summary = _terminal_summary(trace)
    semantic_summary = _semantic_owner_summary(trace)
    tool_summary = _tool_followup_summary(trace)
    overall_verdict_map: dict[TurnReviewBucket, str] = {
        "coherent": "coherent turn trace",
        "partial_expected": "partial observational trace",
        "suspicious": "suspicious arbitration trace",
        "needs_review": "review required",
    }
    overall_summary_parts = [
        response_summary,
        terminal_summary,
        semantic_summary,
        tool_summary,
    ]
    if suspicious_signals:
        overall_summary_parts.append(f"suspicious signals: {suspicious_signals[0]}")
    elif explainability_gaps:
        if bucket == "partial_expected":
            overall_summary_parts.append(f"expected gaps remain: {explainability_gaps[0]}")
        elif bucket == "needs_review":
            overall_summary_parts.append(f"review needed: {explainability_gaps[0]}")
        else:
            overall_summary_parts.append(f"explainability gap: {explainability_gaps[0]}")
    elif bucket == "coherent":
        overall_summary_parts.append("no suspicious signals")
    elif bucket == "partial_expected":
        overall_summary_parts.append("expected gaps remain")
    elif bucket == "needs_review":
        overall_summary_parts.append("review needed")
    else:
        overall_summary_parts.append("suspicious signals require review")
    diagnostics = trace.diagnostics
    return TurnArbitrationReviewSummary(
        run_id=trace.run_id,
        turn_id=trace.turn_id,
        review_bucket=bucket,
        review_priority=_review_priority_for_bucket(bucket),
        overall_verdict=overall_verdict_map[bucket],
        overall_summary="; ".join(overall_summary_parts),
        response_create_summary=response_summary,
        terminal_summary=terminal_summary,
        semantic_owner_summary=semantic_summary,
        tool_followup_summary=tool_summary,
        top_reasons=_top_reasons(trace),
        notable_mismatches=notable_mismatches,
        explainability_gaps=explainability_gaps,
        suspicious_signals=suspicious_signals,
        warning_codes=trace.warning_codes,
        diagnostic_codes=diagnostics.diagnostic_codes if diagnostics is not None else ("diagnostics_unavailable",),
        trace_complete=trace.trace_complete,
        trace_partial=trace.trace_partial,
    )


def get_latest_turn_review_summary(trace: TurnArbitrationTrace | None) -> TurnArbitrationReviewSummary | None:
    if trace is None:
        return None
    attached_summary = trace.review_summary
    if attached_summary is not None:
        return attached_summary
    return build_turn_review_summary(trace)


def summarize_turn_arbitration_for_review(trace: TurnArbitrationTrace) -> TurnArbitrationReviewLogPayload:
    review_summary = get_latest_turn_review_summary(trace)
    if review_summary is None:
        raise ValueError("A turn arbitration trace is required to summarize review output.")
    return review_summary.to_log_payload()


def should_emit_turn_review_summary_info(trace: TurnArbitrationTrace) -> bool:
    review_summary = get_latest_turn_review_summary(trace)
    if review_summary is None:
        return False
    if trace.trace_complete:
        return True
    return review_summary.review_bucket in {"suspicious", "needs_review"}


def turn_review_summary_info_fingerprint(
    trace: TurnArbitrationTrace,
    review_summary: TurnArbitrationReviewSummary | None = None,
) -> tuple[str, ...] | None:
    summary = review_summary or get_latest_turn_review_summary(trace)
    if summary is None:
        return None
    return (
        summary.review_bucket,
        summary.overall_verdict,
        summary.overall_summary,
        summary.response_create_summary,
        summary.terminal_summary,
        summary.semantic_owner_summary,
        summary.tool_followup_summary,
        str(summary.trace_complete).lower(),
    )


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
    if candidate_id == "canonical_key_already_created":
        return "canonical_response_already_created"
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
        "terminal_selected",
        "cancelled",
        "micro_ack_non_deliverable",
        "empty_tool_followup_non_deliverable",
        "provisional_empty_non_deliverable",
        "provisional_server_auto_awaiting_transcript_final",
        "tool_followup_precedence",
        "exact_phrase_obligation_open",
        "tool_output_descriptive_gesture_only",
        "semantic_owner_execution",
        "semantic_owner_parent",
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

    raw_selected_candidate_id = execution_decision.selected_candidate_id
    selected_candidate_id: NormalizedCandidateId = _normalize_selected_candidate_id(
        raw_selected_candidate_id
    )
    decision_owner_scope, decision_owner_warnings = _owner_scope_for_candidate(
        selected_candidate_id,
        same_turn_owner_reason=same_turn_owner_reason,
    )
    decision_warnings = list(decision_owner_warnings)
    if raw_selected_candidate_id != selected_candidate_id:
        decision_warnings.append(f"selected_candidate_id_normalized_from_{raw_selected_candidate_id}")
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
