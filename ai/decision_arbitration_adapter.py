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
    "terminal_selected",
    "cancelled",
    "micro_ack_non_deliverable",
    "provisional_empty_non_deliverable",
    "provisional_server_auto_awaiting_transcript_final",
    "tool_followup_precedence",
    "exact_phrase_obligation_open",
    "tool_output_descriptive_gesture_only",
    "semantic_owner_execution",
    "semantic_owner_parent",
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
    transcript_final_state: str | None
    normalized_warning_count: int
    warning_codes: tuple[str, ...]
    authority_seams_seen: tuple[str, ...]
    trace_complete: bool
    trace_partial: bool


@dataclass(frozen=True)
class TurnArbitrationTrace:
    run_id: str
    turn_id: str
    execution_canonical_key: str | None
    semantic_owner_canonical_key: str | None
    response_create_observation: NormalizedArbitrationObservation | None = None
    terminal_selection_observation: NormalizedArbitrationObservation | None = None
    semantic_owner_observation: NormalizedArbitrationObservation | None = None
    warning_codes: tuple[str, ...] = ()
    authority_seams_seen: tuple[str, ...] = ()
    trace_complete: bool = False
    trace_partial: bool = True

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


_AUTHORITY_SEAM = "ai.realtime.response_create_runtime"
_POLICY_DOMAIN = "response_create"


_TERMINAL_SELECTION_AUTHORITY_SEAM = "ai.realtime_api._response_done_deliverable_decision"
_TERMINAL_SELECTION_POLICY_DOMAIN = "response_terminal_selection"
_SEMANTIC_OWNER_AUTHORITY_SEAM = "ai.realtime_api._resolve_semantic_answer_owner_for_response"
_SEMANTIC_OWNER_POLICY_DOMAIN = "response_semantic_owner"


def _terminal_selection_candidate_id(selection_reason: str, *, selected: bool) -> NormalizedCandidateId:
    normalized_reason = str(selection_reason or "").strip().lower()
    mapping: dict[str, NormalizedCandidateId] = {
        "cancelled": "cancelled",
        "micro_ack_non_deliverable": "micro_ack_non_deliverable",
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
    if normalized_reason in {"micro_ack_non_deliverable", "provisional_empty_non_deliverable", "tool_output_descriptive_gesture_only", "cancelled"}:
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
    transcript_final_seen: bool,
    active_response_was_provisional: bool,
) -> NormalizedArbitrationObservation:
    candidate_id = _terminal_selection_candidate_id(selection_reason, selected=selected)
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
    decision_warnings = tuple([*deliverable_warnings, *transcript_warnings])
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
) -> NormalizedArbitrationObservation:
    warnings: list[str] = []
    normalized_reason = str(selection_reason or "").strip().lower()
    candidate_id: NormalizedCandidateId
    owner_scope: NormalizedOwnerScope
    parent_coverage_state: NormalizedParentCoverageState
    if semantic_owner_canonical_key == execution_canonical_key:
        candidate_id = "semantic_owner_execution"
        owner_scope = "none" if selected else "subsystem_local"
        parent_coverage_state = "not_applicable"
        if selected and normalized_reason == "normal" and str(origin or "").strip().lower() == "tool_output":
            warnings.append("semantic_owner_parent_not_promoted")
    else:
        candidate_id = "semantic_owner_parent"
        owner_scope = "semantic_parent"
        parent_coverage_state = "covered_canonical"
    if not selected:
        warnings.append("semantic_owner_resolution_without_terminal_selection")
    if not semantic_owner_canonical_key:
        warnings.append("semantic_owner_canonical_key_unavailable")
    if candidate_id == "semantic_owner_parent" and (not parent_turn_id or not parent_input_event_key):
        warnings.append("semantic_parent_lineage_unavailable")
    decision = NormalizedDecisionArtifact(
        native_outcome_action="RETAIN" if candidate_id == "semantic_owner_execution" else "REASSIGN",
        native_reason_code=str(selection_reason or ""),
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
    observations = tuple(
        observation
        for observation in (
            merged_response_create,
            merged_terminal_selection,
            merged_semantic_owner,
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
    )
    authority_seams_seen = _merge_authority_seams(
        merged_response_create,
        merged_terminal_selection,
        merged_semantic_owner,
    )
    trace_complete = all(
        observation is not None
        for observation in (
            merged_response_create,
            merged_terminal_selection,
            merged_semantic_owner,
        )
    )
    return TurnArbitrationTrace(
        run_id=run_id,
        turn_id=turn_id,
        execution_canonical_key=execution_canonical_key,
        semantic_owner_canonical_key=semantic_owner_canonical_key,
        response_create_observation=merged_response_create,
        terminal_selection_observation=merged_terminal_selection,
        semantic_owner_observation=merged_semantic_owner,
        warning_codes=warning_codes,
        authority_seams_seen=authority_seams_seen,
        trace_complete=trace_complete,
        trace_partial=not trace_complete,
    )


def summarize_turn_arbitration_trace(trace: TurnArbitrationTrace) -> TurnArbitrationTraceLogPayload:
    return trace.to_log_payload()


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
