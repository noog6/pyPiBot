"""Runtime service for tool-governance confirmation handling."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
import uuid
from typing import Any, Protocol

from ai.realtime.confirmation import ConfirmationState, ConfirmationTransitionDecision
from ai.orchestration import OrchestrationPhase
from core.logging import logger


class ConfirmationRuntimeAPI(Protocol):
    ...


@dataclass
class ConfirmationRuntime:
    api: ConfirmationRuntimeAPI

    def build_confirmation_transition_decision(
        self,
        *,
        reason: str,
        include_reminder_gate: bool = False,
        was_confirmation_guarded: bool = False,
    ) -> ConfirmationTransitionDecision:
        api = self.api
        token = getattr(api, "_pending_confirmation_token", None)
        token_active, awaiting_phase, hold_active, phase_is_awaiting = api._confirmation_hold_components()
        phase = getattr(api.orchestration_state, "phase", None)
        should_close_follow_up = (
            phase_is_awaiting
            and not token_active
            and bool(getattr(api, "_awaiting_confirmation_completion", False))
        )
        close_reason = "confirmation_follow_up_completed" if should_close_follow_up else None
        emit_reminder = bool(include_reminder_gate and token_active and awaiting_phase)
        recover_mic = bool(was_confirmation_guarded and hold_active)
        logger.debug(
            "CONFIRMATION_TRANSITION_DECISION reason=%s allow_response_transition=%s emit_reminder=%s recover_mic=%s close_reason=%s token_active=%s awaiting_phase=%s phase=%s",
            reason,
            not hold_active,
            emit_reminder,
            recover_mic,
            close_reason or "none",
            token_active,
            awaiting_phase,
            phase,
        )
        return ConfirmationTransitionDecision(
            allow_response_transition=not hold_active,
            emit_reminder=emit_reminder,
            recover_mic=recover_mic,
            close_reason=close_reason,
        )

    async def maybe_handle_confirmation_decision_timeout(
        self,
        websocket: Any,
        *,
        source_event: str,
    ) -> bool:
        api = self.api
        transition_lock = await self.confirmation_transition_guard()
        async with transition_lock:
            expired_token = self.expire_confirmation_awaiting_decision_timeout()
        if expired_token is None:
            return False
        logger.info(
            "CONFIRMATION_TIMEOUT token=%s kind=%s cause=awaiting_decision_timeout source_event=%s",
            expired_token.id,
            expired_token.kind,
            source_event,
        )
        action = expired_token.pending_action.action if expired_token.pending_action is not None else None
        api._log_structured_noop_event(
            outcome="timeout",
            reason="awaiting_decision_timeout",
            tool_name=getattr(action, "tool_name", None),
            action_id=getattr(action, "id", None),
            token_id=expired_token.id,
            idempotency_key=getattr(getattr(expired_token, "pending_action", None), "idempotency_key", None),
        )
        await api._emit_final_noop_user_text(
            websocket,
            outcome="timeout",
            reason="awaiting_decision_timeout",
        )
        return True

    def expire_confirmation_awaiting_decision_timeout(self) -> Any | None:
        api = self.api
        token = getattr(api, "_pending_confirmation_token", None)
        if token is None:
            return None
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.pending_token = token
            coordinator.state = getattr(api, "_confirmation_state", ConfirmationState.IDLE)
            coordinator.token_created_at = getattr(api, "_confirmation_token_created_at", None)
            coordinator.pause_started_at = getattr(api, "_confirmation_pause_started_at", None)
            coordinator.paused_accum_s = float(getattr(api, "_confirmation_paused_accum_s", 0.0))
            coordinator.speech_active = bool(getattr(api, "_confirmation_speech_active", False))
            coordinator.asr_pending = bool(getattr(api, "_confirmation_asr_pending", False))
            timeout_decision = coordinator.check_timeout(time.monotonic())
            api._confirmation_pause_started_at = coordinator.pause_started_at
            api._confirmation_paused_accum_s = coordinator.paused_accum_s
            if not timeout_decision.expired:
                return None
        else:
            if getattr(api, "_confirmation_state", ConfirmationState.IDLE) != ConfirmationState.AWAITING_DECISION:
                return None
            timeout_s = self.get_confirmation_timeout_s(token)
            if timeout_s <= 0.0:
                return None
            self.refresh_confirmation_pause()
            pause_reason = self.confirmation_pause_reason()
            remaining_s = self.confirmation_remaining_seconds()
            api._log_confirmation_timeout_check(token, remaining_s=remaining_s, pause_reason=pause_reason)
            if pause_reason is not None or remaining_s > 0.0:
                return None
        self.set_confirmation_state(ConfirmationState.RESOLVING, reason="awaiting_decision_timeout")
        self.close_confirmation_token(outcome="awaiting_decision_timeout")
        api._awaiting_confirmation_completion = False
        api.orchestration_state.transition(
            OrchestrationPhase.IDLE,
            reason="confirmation timeout",
        )
        return token

    def set_confirmation_state(self, state: ConfirmationState, *, reason: str) -> None:
        api = self.api
        previous = getattr(api, "_confirmation_state", ConfirmationState.IDLE)
        token = getattr(api, "_pending_confirmation_token", None)
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.pending_token = token
            coordinator.awaiting_confirmation_completion = bool(
                getattr(api, "_awaiting_confirmation_completion", False)
            )
            coordinator.state = previous
            coordinator.transition(reason, {"state": state, "now": time.monotonic()})
            api._confirmation_state = coordinator.state
            return
        if previous != state:
            api._log_confirmation_transition(previous, state, reason=reason, token=token)
        api._confirmation_state = state
        if state == ConfirmationState.AWAITING_DECISION and token is not None:
            metadata = token.metadata if isinstance(token.metadata, dict) else {}
            metadata.setdefault("awaiting_decision_since", time.monotonic())
            token.metadata = metadata

    async def confirmation_transition_guard(self) -> asyncio.Lock:
        api = self.api
        lock = getattr(api, "_confirmation_transition_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            api._confirmation_transition_lock = lock
        return lock

    def get_confirmation_timeout_s(self, token: Any | None) -> float:
        api = self.api
        if token is not None and token.kind == "research_permission":
            return max(0.0, float(getattr(api, "_research_permission_awaiting_decision_timeout_s", 60.0)))
        return max(0.0, float(getattr(api, "_confirmation_awaiting_decision_timeout_s", 20.0)))

    def confirmation_pause_reason(self) -> str | None:
        api = self.api
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.speech_active = bool(getattr(api, "_confirmation_speech_active", False))
            coordinator.asr_pending = bool(getattr(api, "_confirmation_asr_pending", False))
            return coordinator._pause_reason()
        if getattr(api, "_confirmation_speech_active", False):
            return "speech_active"
        if getattr(api, "_confirmation_asr_pending", False):
            return "asr_pending"
        return None

    def is_confirmation_ttl_paused(self) -> bool:
        return self.confirmation_pause_reason() in {"speech_active", "asr_pending"}

    def refresh_confirmation_pause(self) -> None:
        api = self.api
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is None:
            token = getattr(api, "_pending_confirmation_token", None)
            if token is None:
                api._confirmation_pause_started_at = None
                return
            now = time.monotonic()
            if self.confirmation_pause_reason() is None:
                started = getattr(api, "_confirmation_pause_started_at", None)
                if isinstance(started, (int, float)):
                    api._confirmation_paused_accum_s += max(0.0, now - float(started))
                api._confirmation_pause_started_at = None
                return
            if api._confirmation_pause_started_at is None:
                api._confirmation_pause_started_at = now
            return
        coordinator.pending_token = getattr(api, "_pending_confirmation_token", None)
        coordinator.speech_active = bool(getattr(api, "_confirmation_speech_active", False))
        coordinator.asr_pending = bool(getattr(api, "_confirmation_asr_pending", False))
        coordinator.pause_started_at = getattr(api, "_confirmation_pause_started_at", None)
        coordinator.paused_accum_s = float(getattr(api, "_confirmation_paused_accum_s", 0.0))
        coordinator._refresh_pause(now=time.monotonic())
        api._confirmation_pause_started_at = coordinator.pause_started_at
        api._confirmation_paused_accum_s = coordinator.paused_accum_s

    def mark_confirmation_activity(self, *, reason: str, now: float | None = None) -> None:
        api = self.api
        token = getattr(api, "_pending_confirmation_token", None)
        if token is None:
            return
        current = time.monotonic() if now is None else float(now)
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.pending_token = token
            coordinator.speech_active = bool(getattr(api, "_confirmation_speech_active", False))
            coordinator.asr_pending = bool(getattr(api, "_confirmation_asr_pending", False))
            coordinator.on_user_activity(reason, current)
            api._confirmation_last_activity_at = coordinator.last_activity_at
            api._confirmation_pause_started_at = coordinator.pause_started_at
            api._confirmation_paused_accum_s = coordinator.paused_accum_s
        else:
            api._confirmation_last_activity_at = current
            self.refresh_confirmation_pause()
        remaining_s = self.confirmation_remaining_seconds(now=current)
        logger.info(
            "CONFIRMATION_ACTIVITY token=%s kind=%s reason=%s remaining_s=%.1f paused_reason=%s",
            token.id,
            token.kind,
            reason,
            remaining_s,
            self.confirmation_pause_reason() or "none",
        )

    def confirmation_effective_elapsed_s(self, now: float | None = None) -> float:
        api = self.api
        token = getattr(api, "_pending_confirmation_token", None)
        if token is None:
            return 0.0
        current = time.monotonic() if now is None else float(now)
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.pending_token = token
            coordinator.token_created_at = getattr(api, "_confirmation_token_created_at", None)
            coordinator.pause_started_at = getattr(api, "_confirmation_pause_started_at", None)
            coordinator.paused_accum_s = float(getattr(api, "_confirmation_paused_accum_s", 0.0))
            coordinator.speech_active = bool(getattr(api, "_confirmation_speech_active", False))
            coordinator.asr_pending = bool(getattr(api, "_confirmation_asr_pending", False))
            return coordinator.effective_elapsed_s(now=current)
        started_at = api._confirmation_token_created_at
        if not isinstance(started_at, (int, float)):
            started_at = token.created_at
        paused_s = float(getattr(api, "_confirmation_paused_accum_s", 0.0))
        pause_started = getattr(api, "_confirmation_pause_started_at", None)
        if isinstance(pause_started, (int, float)) and self.confirmation_pause_reason() is not None:
            paused_s += max(0.0, current - float(pause_started))
        return max(0.0, current - float(started_at) - paused_s)

    def confirmation_remaining_seconds(self, *, now: float | None = None) -> float:
        api = self.api
        token = getattr(api, "_pending_confirmation_token", None)
        if token is None:
            return 0.0
        current = time.monotonic() if now is None else float(now)
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.pending_token = token
            coordinator.token_created_at = getattr(api, "_confirmation_token_created_at", None)
            coordinator.pause_started_at = getattr(api, "_confirmation_pause_started_at", None)
            coordinator.paused_accum_s = float(getattr(api, "_confirmation_paused_accum_s", 0.0))
            coordinator.speech_active = bool(getattr(api, "_confirmation_speech_active", False))
            coordinator.asr_pending = bool(getattr(api, "_confirmation_asr_pending", False))
            return coordinator.remaining_seconds(now=current)
        timeout_s = self.get_confirmation_timeout_s(token)
        return max(0.0, timeout_s - self.confirmation_effective_elapsed_s(now=now))

    def create_confirmation_token(
        self,
        *,
        token_cls: Any,
        kind: str,
        tool_name: str | None,
        request: Any = None,
        pending_action: Any = None,
        expiry_ts: float | None = None,
        max_retries: int = 2,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        api = self.api
        token = token_cls(
            id=f"confirm_{uuid.uuid4().hex}",
            kind=kind,
            tool_name=tool_name,
            request=request,
            pending_action=pending_action,
            created_at=time.monotonic(),
            expiry_ts=expiry_ts,
            max_retries=max_retries,
            metadata=metadata or {},
        )
        now = token.created_at
        api._pending_confirmation_token = token
        service = getattr(api, "_confirmation_service", None)
        if service is not None:
            service.start_pending(token, pending_action, now)
        api._confirmation_token_created_at = now
        api._confirmation_last_activity_at = now
        api._confirmation_speech_active = False
        api._confirmation_asr_pending = False
        api._confirmation_pause_started_at = None
        api._confirmation_paused_accum_s = 0.0
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.on_token_started(token, now)
        self.set_confirmation_state(ConfirmationState.PENDING_PROMPT, reason=f"token_created:{kind}")
        logger.info(
            "CONFIRMATION_TOKEN_CREATED token=%s kind=%s tool=%s",
            token.id,
            kind,
            tool_name or "none",
        )
        return token

    def sync_confirmation_legacy_fields(self) -> None:
        api = self.api
        token = getattr(api, "_pending_confirmation_token", None)
        if token is None:
            api._pending_action = None
            api._pending_research_request = None
            if getattr(api, "_confirmation_state", ConfirmationState.IDLE) != ConfirmationState.IDLE:
                self.set_confirmation_state(ConfirmationState.IDLE, reason="token_cleared")
            return
        api._pending_action = token.pending_action
        api._pending_research_request = token.request if token.kind in {"research_permission", "research_budget"} else None

    def close_confirmation_token(self, *, outcome: str) -> None:
        api = self.api
        token = getattr(api, "_pending_confirmation_token", None)
        if token is None:
            self.sync_confirmation_legacy_fields()
            return
        self.set_confirmation_state(ConfirmationState.COMPLETED, reason=f"outcome:{outcome}")
        logger.info(
            "CONFIRMATION_TOKEN_CLOSED token=%s kind=%s outcome=%s",
            token.id,
            token.kind,
            outcome,
        )
        if token.pending_action is not None:
            action = token.pending_action.action
            idempotency_key = token.pending_action.idempotency_key
            if outcome in {"accepted", "approved"}:
                api._record_intent_state(action.tool_name, action.tool_args, "approved", idempotency_key=idempotency_key)
            elif outcome in {"rejected", "cancelled_by_stop_word", "cleared_pending_action"}:
                api._record_intent_state(action.tool_name, action.tool_args, "denied", idempotency_key=idempotency_key)
            elif outcome in {"awaiting_decision_timeout", "expired", "retry_exhausted", "unclear_cancelled"}:
                api._record_intent_state(action.tool_name, action.tool_args, "timeout", idempotency_key=idempotency_key)
            elif outcome == "replaced":
                api._record_intent_state(action.tool_name, action.tool_args, "replaced", idempotency_key=idempotency_key)
            api._record_recent_confirmation_outcome(idempotency_key, outcome)
        pending_deferred_call = api._deferred_research_tool_call
        deferred_call_id = None
        deferred_action = "none"
        if (
            token.kind in {"research_permission", "research_budget"}
            and isinstance(pending_deferred_call, dict)
            and str(pending_deferred_call.get("token_id") or "") == token.id
        ):
            deferred_call_id = str(pending_deferred_call.get("call_id") or "") or "unknown"
            if outcome in {"accepted", "approved"}:
                deferred_action = "retain_for_dispatch"
            else:
                api._deferred_research_tool_call = None
                deferred_action = "cleared"
        logger.debug(
            "CONFIRMATION_TOKEN_CLOSE_PENDING_TOOL token=%s kind=%s outcome=%s call_id=%s action=%s",
            token.id,
            token.kind,
            outcome,
            deferred_call_id or "none",
            deferred_action,
        )
        api._research_pending_call_ids.clear()
        closed_at = time.monotonic()
        api._confirmation_last_closed_token = {
            "token": token,
            "kind": token.kind,
            "closed_at": closed_at,
            "outcome": outcome,
        }
        service = getattr(api, "_confirmation_service", None)
        if service is not None:
            service.close(outcome=outcome, now=closed_at)
        if token.pending_action:
            api._presented_actions.discard(token.pending_action.action.id)
        api._confirmation_timeout_check_last_logged_at.pop(token.id, None)
        api._confirmation_timeout_check_last_pause_reason.pop(token.id, None)
        reminder_key = api._confirmation_reminder_key(token)
        reminder_tracker = getattr(api, "_confirmation_reminder_tracker", None)
        if not isinstance(reminder_tracker, dict):
            reminder_tracker = {}
            api._confirmation_reminder_tracker = reminder_tracker
        if reminder_key is not None:
            reminder_tracker.pop(reminder_key, None)
        if outcome in {"accepted", "rejected", "awaiting_decision_timeout", "replaced"}:
            latch_key = api._confirmation_prompt_latch_key(token)
            if latch_key is not None:
                prompt_latches = getattr(api, "_pending_confirmation_prompt_latches", None)
                if not isinstance(prompt_latches, set):
                    prompt_latches = set()
                    api._pending_confirmation_prompt_latches = prompt_latches
                prompt_latches.discard(latch_key)
        api._clear_queued_confirmation_reminder_markers(token)
        api._pending_confirmation_token = None
        coordinator = getattr(api, "_confirmation_coordinator", None)
        if coordinator is not None:
            coordinator.pending_token = None
            coordinator.token_created_at = None
        api._confirmation_token_created_at = None
        api._confirmation_last_activity_at = None
        api._confirmation_speech_active = False
        api._confirmation_asr_pending = False
        api._confirmation_pause_started_at = None
        api._confirmation_paused_accum_s = 0.0
        self.sync_confirmation_legacy_fields()

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
