"""Deterministic decision policy for realtime interaction lifecycle gates.

Migration note:
    Phase 1 keeps call-site signatures in ``RealtimeAPI`` stable while moving
    inline branching into policy methods that return structured reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    ) -> ResponseCreateDecision:
        if response_in_flight:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.SCHEDULE,
                reason_code="active_response",
                queue_reason="active_response",
            )
        if audio_playback_busy:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.SCHEDULE,
                reason_code="audio_playback_busy",
                queue_reason="audio_playback_busy",
            )
        if consumes_canonical_slot and canonical_audio_started and not explicit_multipart:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.BLOCK,
                reason_code="canonical_audio_already_started",
            )
        if consumes_canonical_slot and single_flight_block_reason:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.BLOCK,
                reason_code=single_flight_block_reason,
            )
        if already_delivered:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.BLOCK,
                reason_code="already_delivered",
            )
        if preference_recall_lock_blocked:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.BLOCK,
                reason_code="preference_recall_lock_blocked",
            )
        if consumes_canonical_slot and canonical_key_already_created and not has_safety_override:
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.BLOCK,
                reason_code="canonical_response_already_created",
            )
        if suppression_active and normalized_origin == "server_auto":
            return ResponseCreateDecision(
                action=ResponseCreateDecisionAction.BLOCK,
                reason_code="preference_recall_suppressed",
            )
        return ResponseCreateDecision(
            action=ResponseCreateDecisionAction.SEND,
            reason_code="direct_send",
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
        if normalized_origin != "server_auto":
            return ServerAutoCreatedDecision(
                action=ServerAutoCreatedDecisionAction.ALLOW,
                reason_code="origin_not_server_auto",
            )
        if not has_turn_id or not has_canonical_key:
            return ServerAutoCreatedDecision(
                action=ServerAutoCreatedDecisionAction.DEFER,
                reason_code="missing_turn_or_canonical",
            )
        if obligation_replacement:
            return ServerAutoCreatedDecision(
                action=ServerAutoCreatedDecisionAction.CANCEL_PRE_AUDIO,
                reason_code="response_obligation_replacement",
            )
        if suppression_by_turn or suppression_window_active or suppression_by_input_event:
            return ServerAutoCreatedDecision(
                action=ServerAutoCreatedDecisionAction.CANCEL_PRE_AUDIO,
                reason_code="preference_recall_suppressed",
            )
        return ServerAutoCreatedDecision(
            action=ServerAutoCreatedDecisionAction.ALLOW,
            reason_code="allow",
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

