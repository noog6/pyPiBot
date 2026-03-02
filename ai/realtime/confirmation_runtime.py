"""Runtime service for tool-governance confirmation handling."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

from ai.realtime.confirmation import ConfirmationState
from ai.orchestration import OrchestrationPhase
from core.logging import logger


class ConfirmationRuntimeAPI(Protocol):
    ...


@dataclass
class ConfirmationRuntime:
    api: ConfirmationRuntimeAPI

    async def maybe_handle_approval_response(
        self, text: str, websocket: Any
    ) -> bool:
        api = self.api
        transition_lock = await api._confirmation_transition_guard()
        async with transition_lock:
            token = getattr(api, "_pending_confirmation_token", None)
            if token is not None and (token.kind != "tool_governance" or token.pending_action is None):
                return False
            pending = token.pending_action if token is not None else api._pending_action
            if pending is None:
                return False
            if token is not None:
                api._mark_confirmation_activity(reason="approval_transcript")
    
            action = pending.action
            if await api._handle_stop_word(text, websocket, source="approval_response"):
                return True
            cooldown_remaining = api._tool_execution_cooldown_remaining()
            if cooldown_remaining > 0:
                if token is not None:
                    api._set_confirmation_state(ConfirmationState.RESOLVING, reason="stop_word_cooldown")
                await api._reject_tool_call(
                    action,
                    f"Tool execution paused for {cooldown_remaining:.0f}s due to stop word.",
                    websocket,
                    staging=pending.staging,
                    status="cancelled",
                )
                if token is not None:
                    api._close_confirmation_token(outcome="cancelled_by_stop_word")
                else:
                    api._clear_pending_action()
                api._awaiting_confirmation_completion = False
                api.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation cancelled by stop word",
                )
                return True
    
            normalized = text.strip()
            if not normalized:
                return False
    
            now = time.monotonic()
            service = getattr(api, "_confirmation_service", None)
            service_decision = (
                service.handle_user_text(normalized, now=now)
                if service is not None and token is not None
                else None
            )
            decision = service_decision.parsed_decision if service_decision is not None else api._parse_confirmation_decision(normalized)
            if api._approval_expired_for_action(action, now=now, decision=decision):
                if token is not None:
                    api._set_confirmation_state(ConfirmationState.RESOLVING, reason="approval_expired")
                logger.info("CONFIRMATION_TIMEOUT tool=%s cause=expiry", action.tool_name)
                api._record_confirmation_timeout(action, cause="expiry")
                api._log_structured_noop_event(
                    outcome="timeout",
                    reason="approval_expired",
                    tool_name=action.tool_name,
                    action_id=action.id,
                    token_id=getattr(token, "id", None),
                    idempotency_key=pending.idempotency_key,
                )
                if token is not None:
                    api._close_confirmation_token(outcome="expired")
                else:
                    api._clear_pending_action()
                await api._emit_final_noop_user_text(
                    websocket,
                    outcome="timeout",
                    reason="approval_expired",
                )
                api._awaiting_confirmation_completion = False
                api.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation timeout",
                )
                return True
    
            logger.info(
                'CONFIRMATION_CANDIDATE transcript="%s" len=%d decision=%s',
                api._clip_text(normalized, limit=100),
                len(normalized),
                decision,
            )
    
            if decision == "yes":
                api._clear_micro_ack_decline_guard()
                if token is not None:
                    api._set_confirmation_state(ConfirmationState.RESOLVING, reason="tool_confirmation_accepted")
                logger.info("CONFIRMATION_ACCEPTED tool=%s", action.tool_name)
                staging = pending.staging or api._stage_action(action)
                api._awaiting_confirmation_completion = True
                await api._execute_action(
                    action,
                    staging,
                    websocket,
                    idempotency_key=pending.idempotency_key,
                    force_no_tools_followup=True,
                    inject_no_tools_instruction=True,
                )
                if token is not None:
                    api._close_confirmation_token(outcome="accepted")
                else:
                    api._clear_pending_action()
                return True
    
            if decision in {"no", "cancel"}:
                api._extend_micro_ack_decline_guard(now=now)
                api._cancel_micro_ack(
                    turn_id=api._current_turn_id_or_unknown(),
                    reason="confirmation_declined",
                )
                if token is not None:
                    api._set_confirmation_state(ConfirmationState.RESOLVING, reason="tool_confirmation_rejected")
                logger.info("CONFIRMATION_REJECTED tool=%s", action.tool_name)
                await api._reject_tool_call(
                    action,
                    "User declined approval.",
                    websocket,
                    status="cancelled",
                    include_assistant_message=False,
                )
                if token is not None:
                    api._close_confirmation_token(outcome="rejected")
                else:
                    api._clear_pending_action()
                await api._emit_final_noop_user_text(
                    websocket,
                    outcome="cancelled",
                    reason="user_declined_approval",
                )
                api._awaiting_confirmation_completion = False
                api.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation rejected",
                )
                return True
    
            if decision == "unclear" and api._detect_alternate_intent_while_confirmation_pending(
                normalized,
                pending_action=pending,
            ):
                api._pending_alternate_intent_override = {
                    "detected_at": time.monotonic(),
                    "token_id": token.id if token is not None else "",
                    "pending_tool": action.tool_name,
                    "idempotency_key": pending.idempotency_key,
                }
                logger.info(
                    "CONFIRMATION_REMINDER_SUPPRESSED reason=%s suppress_reason=alternate_intent_detected idempotency_key=%s run_id=%s",
                    "non_decision_input",
                    pending.idempotency_key or "none",
                    api._current_run_id() or "none",
                )
                return False
    
            if service_decision is not None:
                retry_count = service_decision.retry_count
                max_retries = token.max_retries if token is not None else pending.max_retries
            elif token is not None:
                token.retry_count += 1
                retry_count = token.retry_count
                max_retries = token.max_retries
            else:
                pending.retry_count += 1
                retry_count = pending.retry_count
                max_retries = pending.max_retries
    
            if (service_decision is not None and service_decision.retry_exhausted) or retry_count > max_retries:
                if token is not None:
                    api._set_confirmation_state(ConfirmationState.RESOLVING, reason="tool_confirmation_retry_exhausted")
                logger.info("CONFIRMATION_TIMEOUT tool=%s cause=retry_exhausted", action.tool_name)
                api._record_confirmation_timeout(action, cause="retry_exhausted")
                api._log_structured_noop_event(
                    outcome="timeout",
                    reason="retry_exhausted",
                    tool_name=action.tool_name,
                    action_id=action.id,
                    token_id=getattr(token, "id", None),
                    idempotency_key=pending.idempotency_key,
                )
                if token is not None:
                    api._close_confirmation_token(outcome="retry_exhausted")
                else:
                    api._clear_pending_action()
                await api._emit_final_noop_user_text(
                    websocket,
                    outcome="timeout",
                    reason="retry_exhausted",
                )
                api._awaiting_confirmation_completion = False
                api.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation timeout",
                )
                return True
    
            parsed_intent_category, parsed_intent_idempotency_key = api._classify_pending_confirmation_intent(
                normalized,
                pending_action=pending,
            )
            if token is not None:
                await api._maybe_emit_confirmation_reminder(
                    websocket,
                    reason="non_decision_input",
                    parsed_intent_category=parsed_intent_category,
                    parsed_intent_idempotency_key=parsed_intent_idempotency_key,
                )
            else:
                await api._maybe_emit_confirmation_reminder(
                    websocket,
                    reason="non_decision_input_legacy",
                    parsed_intent_category=parsed_intent_category,
                    parsed_intent_idempotency_key=parsed_intent_idempotency_key,
                )
            return True
    

