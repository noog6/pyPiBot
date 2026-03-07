"""Deterministic decision policy for realtime interaction lifecycle gates.

Migration note:
    Phase 1 keeps call-site signatures in ``RealtimeAPI`` stable while moving
    inline branching into policy methods that return structured reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ai.decision_arbitration import (
    ArbitrationAction,
    ArbitrationCandidate,
    decide_arbitration,
)


class ResponseCreateDecisionAction(str, Enum):
    SEND = "send"
    SCHEDULE = "schedule"
    BLOCK = "block"


class ServerAutoCreatedDecisionAction(str, Enum):
    ALLOW = "allow"
    CANCEL_PRE_AUDIO = "cancel_pre_audio"
    DEFER = "defer"


@dataclass(frozen=True)
class ResponseCreateDecision:
    action: ResponseCreateDecisionAction
    reason_code: str
    queue_reason: str | None = None


@dataclass(frozen=True)
class ServerAutoCreatedDecision:
    action: ServerAutoCreatedDecisionAction
    reason_code: str


@dataclass(frozen=True)
class WatchdogTimeoutDecision:
    reason_code: str
    details: str
    should_schedule_micro_ack: bool


class InteractionLifecyclePolicy:
    """Pure, deterministic decision adapter for response lifecycle transitions."""

    def decide_response_create(
        self,
        *,
        response_in_flight: bool,
        audio_playback_busy: bool,
        consumes_canonical_slot: bool,
        canonical_audio_started: bool,
        explicit_multipart: bool,
        single_flight_block_reason: str,
        already_delivered: bool,
        preference_recall_lock_blocked: bool,
        canonical_key_already_created: bool,
        has_safety_override: bool,
        suppression_active: bool,
        normalized_origin: str,
        awaiting_transcript_final: bool,
    ) -> ResponseCreateDecision:
        candidates: list[ArbitrationCandidate] = []
        if response_in_flight:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="active_response",
                    action=ArbitrationAction.DEFER,
                    reason_code="active_response",
                    priority=100,
                )
            )
        if audio_playback_busy:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="audio_playback_busy",
                    action=ArbitrationAction.DEFER,
                    reason_code="audio_playback_busy",
                    priority=90,
                )
            )
        if consumes_canonical_slot and canonical_audio_started and not explicit_multipart:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="canonical_audio_started",
                    action=ArbitrationAction.REFUSE,
                    reason_code="canonical_audio_already_started",
                    priority=80,
                )
            )
        if consumes_canonical_slot and single_flight_block_reason:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="single_flight_block",
                    action=ArbitrationAction.REFUSE,
                    reason_code=single_flight_block_reason,
                    priority=70,
                )
            )
        if already_delivered:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="already_delivered",
                    action=ArbitrationAction.REFUSE,
                    reason_code="already_delivered",
                    priority=60,
                )
            )
        if preference_recall_lock_blocked:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="preference_recall_lock_blocked",
                    action=ArbitrationAction.REFUSE,
                    reason_code="preference_recall_lock_blocked",
                    priority=50,
                )
            )
        if consumes_canonical_slot and canonical_key_already_created and not has_safety_override:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="canonical_key_already_created",
                    action=ArbitrationAction.REFUSE,
                    reason_code="canonical_response_already_created",
                    priority=40,
                )
            )
        if suppression_active and normalized_origin == "server_auto":
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="preference_recall_suppressed",
                    action=ArbitrationAction.REFUSE,
                    reason_code="preference_recall_suppressed",
                    priority=30,
                )
            )
        if awaiting_transcript_final and normalized_origin == "server_auto":
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="awaiting_transcript_final",
                    action=ArbitrationAction.DEFER,
                    reason_code="awaiting_transcript_final",
                    priority=20,
                )
            )

        arbitration_decision = decide_arbitration(
            policy_name="response_create",
            candidates=candidates,
            default_candidate=ArbitrationCandidate(
                candidate_id="direct_send",
                action=ArbitrationAction.DO_NOW,
                reason_code="direct_send",
                priority=0,
            ),
        )
        if arbitration_decision.action is ArbitrationAction.DEFER:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.SCHEDULE,
                reason_code=arbitration_decision.reason_code,
                queue_reason=arbitration_decision.reason_code,
            )
        if arbitration_decision.action is ArbitrationAction.REFUSE:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.BLOCK,
                reason_code=arbitration_decision.reason_code,
            )
        return ResponseCreateDecision(
            action=ResponseCreateDecisionAction.SEND,
            reason_code=arbitration_decision.reason_code,
        )

    def decide_server_auto_created(
        self,
        *,
        normalized_origin: str,
        has_turn_id: bool,
        has_canonical_key: bool,
        suppression_by_turn: bool,
        suppression_window_active: bool,
        suppression_by_input_event: bool,
        obligation_replacement: bool,
    ) -> ServerAutoCreatedDecision:
        candidates: list[ArbitrationCandidate] = []
        if normalized_origin != "server_auto":
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="origin_not_server_auto",
                    action=ArbitrationAction.DO_NOW,
                    reason_code="origin_not_server_auto",
                    priority=100,
                )
            )
        if not has_turn_id or not has_canonical_key:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="missing_turn_or_canonical",
                    action=ArbitrationAction.DEFER,
                    reason_code="missing_turn_or_canonical",
                    priority=90,
                )
            )
        if obligation_replacement:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="response_obligation_replacement",
                    action=ArbitrationAction.REFUSE,
                    reason_code="response_obligation_replacement",
                    priority=80,
                )
            )
        if suppression_by_turn or suppression_window_active or suppression_by_input_event:
            candidates.append(
                ArbitrationCandidate(
                    candidate_id="preference_recall_suppressed",
                    action=ArbitrationAction.REFUSE,
                    reason_code="preference_recall_suppressed",
                    priority=70,
                )
            )

        arbitration_decision = decide_arbitration(
            policy_name="server_auto_created",
            candidates=candidates,
            default_candidate=ArbitrationCandidate(
                candidate_id="allow",
                action=ArbitrationAction.DO_NOW,
                reason_code="allow",
                priority=0,
            ),
        )
        if arbitration_decision.action is ArbitrationAction.DEFER:
            return ServerAutoCreatedDecision(
                action=ServerAutoCreatedDecisionAction.DEFER,
                reason_code=arbitration_decision.reason_code,
            )
        if arbitration_decision.action is ArbitrationAction.REFUSE:
            return ServerAutoCreatedDecision(
                action=ServerAutoCreatedDecisionAction.CANCEL_PRE_AUDIO,
                reason_code=arbitration_decision.reason_code,
            )
        return ServerAutoCreatedDecision(
            action=ServerAutoCreatedDecisionAction.ALLOW,
            reason_code=arbitration_decision.reason_code,
        )

    def decide_watchdog_timeout(
        self,
        *,
        suppressed_by_turn: bool,
        suppressed_by_input_event: bool,
        suppression_window_active: bool,
        response_in_flight: bool,
        active_response_origin: str,
        active_response_id: str,
        delivery_state_terminal: bool,
        audio_playback_busy: bool,
        has_pending_response_create: bool,
        pending_origin: str,
        pending_reason: str,
        listening_state_gate: bool,
    ) -> WatchdogTimeoutDecision:
        if suppressed_by_turn:
            return WatchdogTimeoutDecision(
                reason_code="preference_recall_suppressed",
                details="turn suppression active",
                should_schedule_micro_ack=False,
            )
        if suppressed_by_input_event:
            return WatchdogTimeoutDecision(
                reason_code="preference_recall_suppressed",
                details="input_event suppression active",
                should_schedule_micro_ack=False,
            )
        if suppression_window_active:
            return WatchdogTimeoutDecision(
                reason_code="preference_recall_suppressed",
                details="suppression window active",
                should_schedule_micro_ack=False,
            )
        if response_in_flight:
            return WatchdogTimeoutDecision(
                reason_code="active_response_in_flight",
                details=f"active_origin={active_response_origin} response_id={active_response_id}",
                should_schedule_micro_ack=False,
            )
        if delivery_state_terminal:
            return WatchdogTimeoutDecision(
                reason_code="already_handled",
                details="canonical delivery terminal state",
                should_schedule_micro_ack=False,
            )
        if audio_playback_busy:
            return WatchdogTimeoutDecision(
                reason_code="audio_playback_busy",
                details="audio playback active",
                should_schedule_micro_ack=True,
            )
        if has_pending_response_create:
            return WatchdogTimeoutDecision(
                reason_code="response_create_queued",
                details=f"pending_origin={pending_origin} pending_reason={pending_reason}",
                should_schedule_micro_ack=True,
            )
        if listening_state_gate:
            return WatchdogTimeoutDecision(
                reason_code="listening_state_gate",
                details="state_manager is LISTENING",
                should_schedule_micro_ack=False,
            )
        return WatchdogTimeoutDecision(
            reason_code="timeout",
            details="response.created missing before timeout",
            should_schedule_micro_ack=True,
        )
