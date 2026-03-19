"""Runtime service for queued response.create arbitration."""

from __future__ import annotations

import time
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ai.decision_arbitration_adapter import build_response_create_observation
from ai.interaction_lifecycle_policy import ResponseCreateDecision, ResponseCreateDecisionAction
from ai.realtime.types import PendingResponseCreate
from core.logging import logger, log_ws_event
from interaction import InteractionState
from ai.realtime.types import UtteranceContext


class ResponseCreateRuntimeAPI(Protocol):
    ...


class ResponseCreateOutcomeAction(str, Enum):
    # Semantics:
    # - BLOCK: a hard guard/policy/state stop prevents execution for this attempt.
    # - DROP: the contender intentionally loses because another owner/path/state
    #   already covers the turn or makes this attempt non-winning.
    SEND = "SEND"
    SCHEDULE = "SCHEDULE"
    BLOCK = "BLOCK"
    DROP = "DROP"


@dataclass(frozen=True)
class ResponseCreatePreparedSnapshot:
    now: float
    run_id: str
    origin: str
    normalized_origin: str
    response_create_event: dict[str, Any]
    response_metadata: dict[str, Any]
    turn_id: str
    input_event_key: str
    canonical_key: str
    effective_memory_note: str | None
    had_pending_preference_context: bool
    preference_note: str
    tool_followup: bool
    tool_call_id: str
    tool_followup_release: bool
    tool_followup_state: str
    consumes_canonical_slot: bool
    explicit_multipart: bool
    transcript_upgrade_replacement: bool
    allow_audio_started_upgrade: bool
    suppression_turns: set[str]
    created_keys: set[str]
    single_flight_block_reason: str
    suppression_active: bool
    awaiting_transcript_final: bool
    preference_recall_lock_blocked: bool
    preference_recall_suppression_active: bool
    preference_recall_lock_active: bool
    response_in_flight: bool
    active_response_present: bool
    already_delivered: bool
    already_created_for_canonical_key: bool
    same_turn_owner_reason: str | None
    same_turn_owner_present: bool
    pending_server_auto_present: bool
    has_safety_override: bool
    audio_playback_busy: bool
    terminal_state_blocked: bool
    lineage_allowed: bool
    lineage_reason: str


@dataclass(frozen=True)
class ResponseCreateExecutionDecision:
    """Canonical response.create arbitration result.

    `action` is the operator-facing outcome surface:
    - SEND / SCHEDULE are executable winners.
    - BLOCK is a guarded refusal/hard stop for this attempt.
    - DROP is an intentional contender discard because another path/state wins.
    """

    action: ResponseCreateOutcomeAction
    reason_code: str
    explanation: str
    selected_candidate_id: str
    queue_reason: str | None = None
    blocked_by_terminal_state: bool = False
    should_log_arbitration: bool = True


@dataclass
class ResponseCreateRuntime:
    api: ResponseCreateRuntimeAPI

    def _apply_memory_intent_instruction_guardrail(
        self,
        *,
        response_create_event: dict[str, Any],
        response_metadata: dict[str, Any],
    ) -> None:
        memory_intent_subtype = str(response_metadata.get("memory_intent_subtype") or "").strip().lower()
        if memory_intent_subtype not in {"preference_recall", "general_memory", "topic_recall"}:
            return
        response_payload = response_create_event.setdefault("response", {})
        if not isinstance(response_payload, dict):
            return
        existing_instructions = str(response_payload.get("instructions") or "").strip()
        guardrail = (
            "Memory-intent response mode: answer the user's memory question directly and prioritize memory/tool "
            "results for this turn. Do not add passive scene or environment narration unless the user explicitly "
            "asks a visual question."
        )
        if guardrail in existing_instructions:
            return
        response_payload["instructions"] = f"{existing_instructions}\n\n{guardrail}" if existing_instructions else guardrail

    def _note_response_create_blocked(self, *, canonical_key: str, reason: str) -> None:
        api = self.api
        api._response_create_queued_creates_total = int(getattr(api, "_response_create_queued_creates_total", 0) or 0) + 1
        api._sync_pending_response_create_queue()
        if str(reason or "").strip().lower() == "active_response":
            logger.debug(
                "response_create_blocked_active canonical_key=%s active_response_id=%s qsize=%s reason=%s",
                canonical_key,
                str(getattr(api, "_active_response_id", "") or "").strip() or "pending_create_ack",
                len(getattr(api, "_response_create_queue", deque()) or ()),
                reason,
            )

    def _build_tool_output_followup_event(
        self,
        *,
        response_create_event: dict[str, Any],
        turn_id: str,
        input_event_key: str,
    ) -> dict[str, Any]:
        followup_event = deepcopy(response_create_event)
        response_payload = followup_event.setdefault("response", {})
        if not isinstance(response_payload, dict):
            response_payload = {}
            followup_event["response"] = response_payload
        metadata = response_payload.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            response_payload["metadata"] = metadata
        followup_input_event_key = f"{input_event_key}:tool_followup"
        metadata["turn_id"] = turn_id
        metadata["input_event_key"] = followup_input_event_key
        metadata["consumes_canonical_slot"] = "false"
        metadata["explicit_multipart"] = "true"
        metadata["tool_followup"] = "true"
        metadata["tool_followup_reason"] = "already_created"
        return followup_event

    def _build_execution_decision(
        self,
        *,
        action: ResponseCreateOutcomeAction,
        reason_code: str,
        explanation: str,
        selected_candidate_id: str,
        queue_reason: str | None = None,
        blocked_by_terminal_state: bool = False,
        should_log_arbitration: bool = True,
    ) -> ResponseCreateExecutionDecision:
        return ResponseCreateExecutionDecision(
            action=action,
            reason_code=reason_code,
            explanation=explanation,
            selected_candidate_id=selected_candidate_id,
            queue_reason=queue_reason,
            blocked_by_terminal_state=blocked_by_terminal_state,
            should_log_arbitration=should_log_arbitration,
        )

    def _normalize_execution_decision(
        self,
        *,
        prepared_snapshot: ResponseCreatePreparedSnapshot,
        decision: ResponseCreateExecutionDecision,
    ) -> ResponseCreateExecutionDecision:
        preferred_reason_code = decision.reason_code
        preferred_explanation = decision.explanation
        if preferred_reason_code == "already_done" and prepared_snapshot.already_delivered:
            preferred_reason_code = "already_delivered"
            preferred_explanation = "Response.create blocked: already_delivered."
        if preferred_reason_code == decision.reason_code and preferred_explanation == decision.explanation:
            return decision
        return self._build_execution_decision(
            action=decision.action,
            reason_code=preferred_reason_code,
            explanation=preferred_explanation,
            selected_candidate_id=decision.selected_candidate_id,
            queue_reason=decision.queue_reason,
            blocked_by_terminal_state=decision.blocked_by_terminal_state,
            should_log_arbitration=decision.should_log_arbitration,
        )

    def _schedule_path_admission_override(
        self,
        *,
        prepared_snapshot: ResponseCreatePreparedSnapshot,
        decision: ResponseCreateExecutionDecision,
    ) -> ResponseCreateExecutionDecision:
        """Apply queue-admission-only guards after core arbitration.

        These are intentionally path-local because they manage queue ownership /
        side effects (mainly tool-followup suppression), not the core shared
        response.create arbitration outcome.
        """
        if decision.action is not ResponseCreateOutcomeAction.SCHEDULE:
            return decision
        api = self.api
        if not prepared_snapshot.tool_followup:
            return decision
        parent_turn_id = str(prepared_snapshot.response_metadata.get("parent_turn_id") or prepared_snapshot.turn_id or "").strip() or prepared_snapshot.turn_id
        parent_input_event_key = str(prepared_snapshot.response_metadata.get("parent_input_event_key") or "").strip() or None
        if hasattr(api, "_should_suppress_tool_followup_after_turn_deliverable") and api._should_suppress_tool_followup_after_turn_deliverable(
            turn_id=parent_turn_id,
            parent_input_event_key=parent_input_event_key,
        ):
            return self._build_execution_decision(
                action=ResponseCreateOutcomeAction.DROP,
                reason_code="tool_followup_final_deliverable_already_sent",
                explanation="Tool followup dropped because the parent turn already has a final deliverable.",
                selected_candidate_id="tool_followup_final_deliverable",
            )
        current_state = self.api._tool_followup_state(canonical_key=prepared_snapshot.canonical_key)
        if current_state in {"creating", "created", "done", "dropped", "scheduled", "blocked_active_response", "released_on_response_done"}:
            return self._build_execution_decision(
                action=ResponseCreateOutcomeAction.DROP,
                reason_code=f"already_{current_state}",
                explanation=f"Tool followup dropped because it was already {current_state}.",
                selected_candidate_id="tool_followup_existing_state",
            )
        return decision

    def _log_response_create_outcome(
        self,
        *,
        snapshot: ResponseCreatePreparedSnapshot,
        decision: ResponseCreateExecutionDecision,
    ) -> None:
        logger.info(
            "response_create_outcome run_id=%s turn_id=%s input_event_key=%s canonical_key=%s origin=%s "
            "action=%s reason_code=%s explanation=%s awaiting_transcript_final=%s response_in_flight=%s "
            "already_delivered=%s already_created=%s same_turn_owner_present=%s suppression_active=%s "
            "terminal_state_blocked=%s pending_server_auto_present=%s tool_followup_state=%s",
            snapshot.run_id or "",
            snapshot.turn_id,
            snapshot.input_event_key or "unknown",
            snapshot.canonical_key,
            snapshot.normalized_origin,
            decision.action.value,
            decision.reason_code,
            decision.explanation,
            snapshot.awaiting_transcript_final,
            snapshot.response_in_flight,
            snapshot.already_delivered,
            snapshot.already_created_for_canonical_key,
            snapshot.same_turn_owner_present,
            snapshot.preference_recall_suppression_active,
            snapshot.terminal_state_blocked,
            snapshot.pending_server_auto_present,
            snapshot.tool_followup_state,
        )

    def prepare_response_create_snapshot(
        self,
        *,
        response_create_event: dict[str, Any],
        origin: str,
        utterance_context: UtteranceContext | None,
        memory_brief_note: str | None,
        now: float,
    ) -> ResponseCreatePreparedSnapshot:
        """Prepare mutable request state and snapshot decision inputs for response.create."""
        api = self.api
        context_hint = utterance_context or getattr(api, "_utterance_context", None)
        if context_hint is not None:
            metadata = api._extract_response_create_metadata(response_create_event)
            metadata.setdefault("turn_id", context_hint.turn_id)
            if context_hint.input_event_key:
                metadata.setdefault("input_event_key", context_hint.input_event_key)
        turn_id = api._resolve_response_create_turn_id(origin=origin, response_create_event=response_create_event)
        current_input_event_key = api._ensure_response_create_correlation(
            response_create_event=response_create_event,
            origin=origin,
            turn_id=turn_id,
        )
        response_metadata = api._extract_response_create_metadata(response_create_event)
        canonical_origin = api._canonical_response_create_origin(
            origin=origin,
            response_metadata=response_metadata,
        )

        preference_payload = None
        if hasattr(api, "_peek_pending_preference_memory_context_payload"):
            preference_payload = api._peek_pending_preference_memory_context_payload(
                turn_id=turn_id,
                input_event_key=current_input_event_key,
            )
        had_pending_preference_context = isinstance(preference_payload, dict)
        preference_note = (
            str(preference_payload.get("prompt_note") or "").strip()
            if isinstance(preference_payload, dict)
            else ""
        )
        effective_memory_note = memory_brief_note or preference_note or None
        api._bind_active_input_event_key_for_turn(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            allow_tool_rebind=(
                str(response_metadata.get("tool_followup_release", "")).strip().lower() in {"true", "1", "yes"}
            ),
            cause="prepare_response_create",
            origin=canonical_origin,
        )
        with api._utterance_context_scope(turn_id=turn_id, input_event_key=current_input_event_key) as resolved_context:
            turn_id = resolved_context.turn_id
            current_input_event_key = resolved_context.input_event_key
            canonical_key = resolved_context.canonical_key

        self._apply_memory_intent_instruction_guardrail(
            response_create_event=response_create_event,
            response_metadata=response_metadata,
        )
        normalized_origin = canonical_origin
        lineage_allowed = True
        lineage_reason = ""
        lineage_guard = getattr(api, "_evaluate_tool_lineage_guard", None)
        if callable(lineage_guard):
            lineage_allowed, lineage_reason, _lineage_canonical_key, _lineage_parent_state, _lineage_call_id = lineage_guard(
                origin=normalized_origin,
                turn_id=turn_id,
                input_event_key=current_input_event_key,
                response_metadata=response_metadata,
            )
        tool_followup = str(response_metadata.get("tool_followup", "")).strip().lower() in {"true", "1", "yes"}
        tool_call_id = str(response_metadata.get("tool_call_id") or "").strip()
        tool_followup_release = str(response_metadata.get("tool_followup_release", "")).strip().lower() in {"true", "1", "yes"}
        tool_followup_state = api._tool_followup_state(canonical_key=canonical_key) if tool_followup else "new"
        consumes_canonical_slot = api._response_consumes_canonical_slot(response_metadata)
        explicit_multipart = api._response_is_explicit_multipart(response_metadata)
        transcript_upgrade_replacement = str(response_metadata.get("transcript_upgrade_replacement", "")).strip().lower() in {"true", "1", "yes"}
        allow_audio_started_upgrade = api._should_upgrade_with_audio_started(
            response_metadata,
            origin=normalized_origin,
        )
        suppression_turns = getattr(api, "_preference_recall_suppressed_turns", set())
        created_keys = getattr(api, "_response_created_canonical_keys", set())
        single_flight_block_reason = (
            ""
            if (not consumes_canonical_slot or transcript_upgrade_replacement)
            else api._single_flight_block_reason(
                turn_id=turn_id,
                input_event_key=current_input_event_key,
            )
        )
        suppression_active = (turn_id in suppression_turns) and (not str(current_input_event_key or "").strip())
        awaiting_transcript_final = False
        if normalized_origin == "server_auto":
            missing_for_turn = getattr(api, "_transcript_final_missing_for_turn", None)
            if callable(missing_for_turn):
                awaiting_transcript_final = bool(
                    missing_for_turn(turn_id=turn_id, input_event_key=current_input_event_key)
                )
            if not awaiting_transcript_final:
                gating_verdict = getattr(api, "_get_response_gating_verdict", None)
                if callable(gating_verdict):
                    verdict = gating_verdict(turn_id=turn_id, input_event_key=current_input_event_key)
                    awaiting_transcript_final = (
                        str(getattr(verdict, "reason", "") or "").strip().lower() == "awaiting_transcript_final"
                    )
            if not awaiting_transcript_final:
                awaiting_transcript_final = str(current_input_event_key or "").strip().startswith("synthetic_server_auto_")
        preference_recall_lock_blocked = api._is_preference_recall_lock_blocked(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            normalized_origin=normalized_origin,
            response_metadata=response_metadata,
        )
        same_turn_owner_reason = None
        if normalized_origin == "assistant_message":
            owner_fn = getattr(api, "_assistant_message_same_turn_owner_reason", None)
            if callable(owner_fn):
                same_turn_owner_reason = str(
                    owner_fn(
                        turn_id=turn_id,
                        input_event_key=current_input_event_key,
                        canonical_key=canonical_key,
                    )
                    or ""
                ).strip() or None
        already_delivered = api._is_response_already_delivered(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        )
        has_safety_override = api._response_has_safety_override(response_create_event)
        terminal_state_blocked = api._drop_response_create_for_terminal_state(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            origin=origin,
            response_metadata=response_metadata,
        )
        pending_server_auto_present = False
        pending_server_auto_for_turn = getattr(api, "_pending_server_auto_response_for_turn", None)
        if callable(pending_server_auto_for_turn):
            pending_server_auto_present = pending_server_auto_for_turn(turn_id=turn_id) is not None
        response_in_flight = api._is_active_response_blocking()
        created_for_key = canonical_key in created_keys
        return ResponseCreatePreparedSnapshot(
            now=now,
            run_id=str(api._current_run_id() or "").strip(),
            origin=origin,
            normalized_origin=normalized_origin,
            response_create_event=response_create_event,
            response_metadata=response_metadata,
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            canonical_key=canonical_key,
            effective_memory_note=effective_memory_note,
            had_pending_preference_context=had_pending_preference_context,
            preference_note=preference_note,
            tool_followup=tool_followup,
            tool_call_id=tool_call_id,
            tool_followup_release=tool_followup_release,
            tool_followup_state=tool_followup_state,
            consumes_canonical_slot=consumes_canonical_slot,
            explicit_multipart=explicit_multipart,
            transcript_upgrade_replacement=transcript_upgrade_replacement,
            allow_audio_started_upgrade=allow_audio_started_upgrade,
            suppression_turns=suppression_turns,
            created_keys=created_keys,
            single_flight_block_reason=single_flight_block_reason,
            suppression_active=suppression_active,
            awaiting_transcript_final=awaiting_transcript_final,
            preference_recall_lock_blocked=preference_recall_lock_blocked,
            preference_recall_suppression_active=suppression_active,
            preference_recall_lock_active=preference_recall_lock_blocked,
            response_in_flight=response_in_flight,
            active_response_present=response_in_flight,
            already_delivered=already_delivered,
            already_created_for_canonical_key=created_for_key,
            same_turn_owner_reason=same_turn_owner_reason,
            same_turn_owner_present=same_turn_owner_reason is not None,
            pending_server_auto_present=pending_server_auto_present,
            has_safety_override=has_safety_override,
            audio_playback_busy=bool(api._audio_playback_busy),
            terminal_state_blocked=terminal_state_blocked,
            lineage_allowed=bool(lineage_allowed),
            lineage_reason=lineage_reason,
        )

    def _decide_response_create_action_with_lifecycle(
        self,
        prepared_snapshot: ResponseCreatePreparedSnapshot,
    ) -> tuple[ResponseCreateExecutionDecision, ResponseCreateDecision | None]:
        """Private observation seam helper for response.create decisions.

        This returns the same authoritative execution decision used by the
        runtime plus the underlying lifecycle-policy decision when one exists
        so the observational adapter can mirror, not steer, the seam. BLOCK is
        reserved for hard runtime/policy guards. DROP is reserved for
        non-winning contenders already covered by another owner/path/state.
        """
        api = self.api
        if not prepared_snapshot.lineage_allowed:
            return self._normalize_execution_decision(
                prepared_snapshot=prepared_snapshot,
                decision=self._build_execution_decision(
                    action=ResponseCreateOutcomeAction.BLOCK,
                    reason_code=prepared_snapshot.lineage_reason or "lineage_blocked",
                    explanation="Tool lineage guard blocked response.create.",
                    selected_candidate_id="tool_lineage_guard",
                    should_log_arbitration=False,
                ),
            ), None
        if prepared_snapshot.terminal_state_blocked:
            return self._normalize_execution_decision(
                prepared_snapshot=prepared_snapshot,
                decision=self._build_execution_decision(
                    action=ResponseCreateOutcomeAction.BLOCK,
                    reason_code="canonical_terminal_state",
                    explanation="Canonical turn is already terminal.",
                    selected_candidate_id="canonical_terminal_state",
                    blocked_by_terminal_state=True,
                    should_log_arbitration=False,
                ),
            ), None
        if prepared_snapshot.same_turn_owner_present:
            return self._normalize_execution_decision(
                prepared_snapshot=prepared_snapshot,
                decision=self._build_execution_decision(
                    action=ResponseCreateOutcomeAction.DROP,
                    reason_code="same_turn_already_owned",
                    explanation=f"Assistant message suppressed by same-turn owner: {prepared_snapshot.same_turn_owner_reason}.",
                    selected_candidate_id="same_turn_owner",
                ),
            ), None
        canonical_audio_started = (
            api._canonical_first_audio_started(prepared_snapshot.canonical_key)
            and not prepared_snapshot.allow_audio_started_upgrade
        )
        policy_decision: ResponseCreateDecision = api._lifecycle_policy().decide_response_create(
            response_in_flight=prepared_snapshot.response_in_flight,
            audio_playback_busy=prepared_snapshot.audio_playback_busy,
            consumes_canonical_slot=prepared_snapshot.consumes_canonical_slot,
            canonical_audio_started=canonical_audio_started,
            explicit_multipart=prepared_snapshot.explicit_multipart,
            single_flight_block_reason=prepared_snapshot.single_flight_block_reason,
            already_delivered=prepared_snapshot.already_delivered,
            preference_recall_lock_blocked=prepared_snapshot.preference_recall_lock_blocked,
            canonical_key_already_created=prepared_snapshot.already_created_for_canonical_key,
            has_safety_override=prepared_snapshot.has_safety_override,
            suppression_active=prepared_snapshot.suppression_active,
            normalized_origin=prepared_snapshot.normalized_origin,
            awaiting_transcript_final=prepared_snapshot.awaiting_transcript_final,
        )
        if policy_decision.action is ResponseCreateDecisionAction.SCHEDULE:
            return self._normalize_execution_decision(
                prepared_snapshot=prepared_snapshot,
                decision=self._build_execution_decision(
                    action=ResponseCreateOutcomeAction.SCHEDULE,
                    reason_code=policy_decision.reason_code,
                    explanation=f"Response.create deferred: {policy_decision.reason_code}.",
                    selected_candidate_id=policy_decision.selected_candidate_id,
                    queue_reason=policy_decision.queue_reason,
                ),
            ), policy_decision
        if policy_decision.action is ResponseCreateDecisionAction.BLOCK:
            return self._normalize_execution_decision(
                prepared_snapshot=prepared_snapshot,
                decision=self._build_execution_decision(
                    action=ResponseCreateOutcomeAction.BLOCK,
                    reason_code=policy_decision.reason_code,
                    explanation=f"Response.create blocked: {policy_decision.reason_code}.",
                    selected_candidate_id=policy_decision.selected_candidate_id,
                ),
            ), policy_decision
        return self._normalize_execution_decision(
            prepared_snapshot=prepared_snapshot,
            decision=self._build_execution_decision(
                action=ResponseCreateOutcomeAction.SEND,
                reason_code=policy_decision.reason_code,
                explanation="Response.create allowed for immediate send.",
                selected_candidate_id=policy_decision.selected_candidate_id,
            ),
        ), policy_decision

    def decide_response_create_action(
        self,
        prepared_snapshot: ResponseCreatePreparedSnapshot,
    ) -> ResponseCreateExecutionDecision:
        decision, _lifecycle_decision = self._decide_response_create_action_with_lifecycle(prepared_snapshot)
        return decision

    def evaluate_response_create_attempt(
        self,
        *,
        response_create_event: dict[str, Any],
        origin: str,
        utterance_context: UtteranceContext | None,
        memory_brief_note: str | None,
        now: float | None = None,
    ) -> tuple[ResponseCreatePreparedSnapshot, ResponseCreateExecutionDecision]:
        prepared_snapshot = self.prepare_response_create_snapshot(
            response_create_event=response_create_event,
            origin=origin,
            utterance_context=utterance_context,
            memory_brief_note=memory_brief_note,
            now=time.monotonic() if now is None else now,
        )
        decision, lifecycle_decision = self._decide_response_create_action_with_lifecycle(prepared_snapshot)
        canonical_audio_started: bool | None = None
        if lifecycle_decision is not None:
            canonical_audio_started = bool(
                self.api._canonical_first_audio_started(prepared_snapshot.canonical_key)
                and not prepared_snapshot.allow_audio_started_upgrade
            )
        observation = build_response_create_observation(
            snapshot=prepared_snapshot,
            execution_decision=decision,
            lifecycle_decision=lifecycle_decision,
            same_turn_owner_reason=prepared_snapshot.same_turn_owner_reason,
            canonical_audio_started=canonical_audio_started,
        )
        logger.debug("decision_adapter_observation payload=%s", observation.to_log_payload())
        return prepared_snapshot, decision

    def schedule_pending_response_create(
        self,
        *,
        websocket: Any,
        response_create_event: dict[str, Any],
        origin: str,
        reason: str,
        record_ai_call: bool,
        debug_context: dict[str, Any] | None,
        memory_brief_note: str | None,
        emit_outcome_log: bool = True,
    ) -> bool:
        api = self.api
        prepared_snapshot, decision = self.evaluate_response_create_attempt(
            response_create_event=response_create_event,
            origin=origin,
            utterance_context=None,
            memory_brief_note=memory_brief_note,
        )
        turn_id = prepared_snapshot.turn_id
        current_input_event_key = prepared_snapshot.input_event_key
        canonical_key = prepared_snapshot.canonical_key
        normalized_origin = prepared_snapshot.normalized_origin
        response_metadata = prepared_snapshot.response_metadata
        decision = self._schedule_path_admission_override(
            prepared_snapshot=prepared_snapshot,
            decision=decision,
        )
        if emit_outcome_log:
            self._log_response_create_outcome(snapshot=prepared_snapshot, decision=decision)
        if decision.action is not ResponseCreateOutcomeAction.SCHEDULE:
            if decision.action is ResponseCreateOutcomeAction.DROP and prepared_snapshot.tool_followup:
                api._set_tool_followup_state(
                    canonical_key=canonical_key,
                    state="dropped",
                    reason=decision.reason_code,
                )
            if decision.action is ResponseCreateOutcomeAction.DROP and decision.reason_code == "same_turn_already_owned":
                api._mark_transcript_response_outcome(
                    input_event_key=current_input_event_key,
                    turn_id=turn_id,
                    outcome="response_not_scheduled",
                    reason="same_turn_already_owned",
                    details=f"owner={prepared_snapshot.same_turn_owner_reason} canonical_key={canonical_key} origin={normalized_origin}",
                )
            return False
        tool_followup = str(response_metadata.get("tool_followup", "")).strip().lower() in {"true", "1", "yes"}
        tool_call_id = str(response_metadata.get("tool_call_id") or "").strip()
        if tool_followup:
            current_state = api._tool_followup_state(canonical_key=canonical_key)
            if current_state in {"creating", "created", "done", "dropped"}:
                logger.info(
                    "tool_followup_arbitration outcome=deny reason=already_%s call_id=%s canonical_key=%s",
                    current_state,
                    tool_call_id or "unknown",
                    canonical_key,
                )
                logger.info(
                    "tool_followup_create_suppressed canonical_key=%s reason=already_%s prior_state=%s",
                    canonical_key,
                    current_state,
                    current_state,
                )
                return False
        consumes_canonical_slot = api._response_consumes_canonical_slot(response_metadata)
        explicit_multipart = api._response_is_explicit_multipart(response_metadata)
        transcript_upgrade_replacement = str(response_metadata.get("transcript_upgrade_replacement", "")).strip().lower() in {"true", "1", "yes"}
        allow_audio_started_upgrade = api._should_upgrade_with_audio_started(
            response_metadata,
            origin=normalized_origin,
        )
        if consumes_canonical_slot and api._canonical_first_audio_started(canonical_key) and not explicit_multipart and not allow_audio_started_upgrade:
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=direct reason=canonical_audio_already_started canonical_key=%s",
                api._current_run_id() or "",
                turn_id,
                normalized_origin,
                canonical_key,
            )
            return False
        suppression_turns = getattr(api, "_preference_recall_suppressed_turns", set())
        obligation_present = api._response_obligation_key(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        ) in getattr(api, "_response_obligations", {})
        pending_server_auto_len = len(getattr(api, "_pending_server_auto_input_event_keys", deque()) or ())
        logger.debug(
            "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
            "canonical_key=%s sent_now=false scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
            "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
            api._current_run_id() or "",
            turn_id,
            normalized_origin,
            reason,
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(api, "_response_create_queue", deque()) or ()),
            pending_server_auto_len,
            bool(turn_id in suppression_turns),
            api._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
            obligation_present,
            getattr(api, "_response_done_serial", 0),
        )
        suppression_active = (turn_id in suppression_turns) and (not str(current_input_event_key or "").strip())
        schedule_logged_turn_ids = getattr(api, "_response_schedule_logged_turn_ids", None)
        if not isinstance(schedule_logged_turn_ids, set):
            schedule_logged_turn_ids = set()
            api._response_schedule_logged_turn_ids = schedule_logged_turn_ids
        if turn_id not in schedule_logged_turn_ids:
            schedule_logged_turn_ids.add(turn_id)
            turn_timestamps_store = getattr(api, "_turn_diagnostic_timestamps", None)
            if not isinstance(turn_timestamps_store, dict):
                turn_timestamps_store = {}
                api._turn_diagnostic_timestamps = turn_timestamps_store
            turn_timestamps = turn_timestamps_store.setdefault(turn_id, {})
            turn_timestamps["response_schedule"] = time.monotonic()
            logger.info(
                "response_schedule_marker run_id=%s turn_id=%s origin=%s input_event_key=%s canonical_key=%s suppression_active=%s mode=queued",
                api._current_run_id() or "",
                turn_id,
                origin,
                current_input_event_key or "unknown",
                canonical_key,
                suppression_active,
            )
            logger.debug(
                "turn_diagnostic_timestamps run_id=%s turn_id=%s transcript_final_ts=%s preference_recall_start_ts=%s preference_recall_end_ts=%s response_schedule_ts=%s",
                api._current_run_id() or "",
                turn_id,
                turn_timestamps.get("transcript_final"),
                turn_timestamps.get("preference_recall_start"),
                turn_timestamps.get("preference_recall_end"),
                turn_timestamps.get("response_schedule"),
            )
        reminder_key = api._extract_confirmation_reminder_dedupe_key(response_create_event)
        candidate = PendingResponseCreate(
            websocket=websocket,
            event=response_create_event,
            origin=origin,
            turn_id=turn_id,
            created_at=time.monotonic(),
            reason=reason,
            record_ai_call=record_ai_call,
            debug_context=debug_context,
            memory_brief_note=memory_brief_note,
            queued_reminder_key=reminder_key,
            enqueued_done_serial=api._response_done_serial,
            enqueue_seq=api._next_response_create_enqueue_seq(),
        )
        if tool_followup:
            response_metadata["tool_followup_release"] = "true"
            scheduled_state = "scheduled"
            scheduled_reason = f"{reason or 'queued'}"
            if str(reason or "").strip().lower() == "active_response":
                active_response_id = str(getattr(api, "_active_response_id", "") or "").strip() or "unknown"
                active_origin = str(getattr(api, "_active_response_origin", "unknown") or "unknown").strip() or "unknown"
                response_metadata["blocked_by_response_id"] = active_response_id
                scheduled_state = "blocked_active_response"
                scheduled_reason = f"active_response response_id={active_response_id} origin={active_origin}"
            api._set_tool_followup_state(
                canonical_key=canonical_key,
                state=scheduled_state,
                reason=scheduled_reason,
            )

        previous = api._pending_response_create
        if previous is not None:
            previous_metadata = api._extract_response_create_metadata(previous.event)
            previous_input_event_key = str(previous_metadata.get("input_event_key") or "").strip()
            previous_turn_id = str(previous_metadata.get("turn_id") or previous.turn_id or "").strip() or previous.turn_id
            previous_canonical_key = api._canonical_utterance_key(
                turn_id=previous_turn_id,
                input_event_key=previous_input_event_key,
            )
            if canonical_key == previous_canonical_key:
                logger.info("response_create_queue_deduped canonical_key=%s", canonical_key)
                return False

        for queued in list(getattr(api, "_response_create_queue", deque()) or ()):
            metadata = api._extract_response_create_metadata(queued.get("event") or {})
            queued_canonical_key = api._canonical_utterance_key(
                turn_id=str(queued.get("turn_id") or "").strip(),
                input_event_key=str(metadata.get("input_event_key") or "").strip(),
            )
            if queued_canonical_key == canonical_key:
                logger.info("response_create_queue_deduped canonical_key=%s", canonical_key)
                return False

        if previous is None:
            api._pending_response_create = candidate
        else:
            api._response_create_queue.append(
                {
                    "websocket": candidate.websocket,
                    "event": candidate.event,
                    "origin": candidate.origin,
                    "turn_id": candidate.turn_id,
                    "record_ai_call": candidate.record_ai_call,
                    "debug_context": candidate.debug_context,
                    "memory_brief_note": candidate.memory_brief_note,
                    "queued_reminder_key": candidate.queued_reminder_key,
                    "enqueued_done_serial": candidate.enqueued_done_serial,
                    "enqueue_seq": candidate.enqueue_seq,
                }
            )
        api._response_create_queued_creates_total = int(getattr(api, "_response_create_queued_creates_total", 0) or 0) + 1
        api._sync_pending_response_create_queue()
        if str(reason or "").strip().lower() == "active_response":
            logger.debug(
                "response_create_blocked_active canonical_key=%s active_response_id=%s qsize=%s reason=%s",
                canonical_key,
                str(getattr(api, "_active_response_id", "") or "").strip() or "pending_create_ack",
                len(getattr(api, "_response_create_queue", deque()) or ()),
                "active_response",
            )
        if reason == "audio_playback_busy":
            logger.debug(
                "playback_busy_retry_enqueued run_id=%s input_event_key=%s reason=audio_playback_busy",
                api._current_run_id() or "",
                current_input_event_key or "unknown",
            )
        schedule_log_method = logger.info
        if str(reason or "").strip().lower() == "active_response":
            schedule_log_method = logger.debug
        schedule_log_method(
                "response_create_scheduled turn_id=%s origin=%s reason=%s",
            candidate.turn_id,
            candidate.origin,
            decision.reason_code,
        )
        api._log_response_site_debug(
            site="response_create_scheduled",
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            canonical_key=canonical_key,
            origin=normalized_origin,
            trigger=reason,
        )
        logger.debug(
            "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
            "canonical_key=%s sent_now=false scheduled=true replaced=false queue_len=%s pending_server_auto_len=%s "
            "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
            api._current_run_id() or "",
            turn_id,
            normalized_origin,
            decision.reason_code,
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(api, "_response_create_queue", deque()) or ()),
            len(getattr(api, "_pending_server_auto_input_event_keys", deque()) or ()),
            bool(turn_id in suppression_turns),
            api._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
            obligation_present,
            getattr(api, "_response_done_serial", 0),
        )
        return False

    async def send_response_create(
        self,
        websocket: Any,
        response_create_event: dict[str, Any],
        *,
        origin: str,
        utterance_context: UtteranceContext | None = None,
        record_ai_call: bool = False,
        debug_context: dict[str, Any] | None = None,
        memory_brief_note: str | None = None,
    ) -> bool:
        api = self.api
        now = time.monotonic()
        delta_ms = None
        if api._last_response_create_ts is not None:
            delta_ms = (now - api._last_response_create_ts) * 1000.0
        if api._response_create_debug_trace:
            ctx = debug_context or {}
            logger.info(
                "[Debug][response.create] active_response_id=%s origin=%s tool=%s call_id=%s research_id=%s delta_ms=%s",
                api._active_response_id,
                origin,
                ctx.get("tool_name"),
                ctx.get("call_id"),
                ctx.get("research_id"),
                f"{delta_ms:.1f}" if delta_ms is not None else "n/a",
            )

        prepared_snapshot, decision = self.evaluate_response_create_attempt(
            response_create_event=response_create_event,
            origin=origin,
            utterance_context=utterance_context,
            memory_brief_note=memory_brief_note,
        )
        turn_id = prepared_snapshot.turn_id
        current_input_event_key = prepared_snapshot.input_event_key
        canonical_key = prepared_snapshot.canonical_key
        normalized_origin = prepared_snapshot.normalized_origin
        response_metadata = prepared_snapshot.response_metadata
        pref_len = len(prepared_snapshot.preference_note)
        compile_keys_considered = [
            api._canonical_utterance_key(turn_id=turn_id, input_event_key=current_input_event_key),
            api._canonical_utterance_key(turn_id=turn_id, input_event_key=api._active_input_event_key_for_turn(turn_id)),
            api._canonical_utterance_key(turn_id=turn_id, input_event_key=str(getattr(api, "_active_server_auto_input_event_key", "") or "").strip()),
        ]
        deduped_compile_keys: list[str] = []
        for key in compile_keys_considered:
            if key not in deduped_compile_keys:
                deduped_compile_keys.append(key)
        logger.debug(
            "response_prompt_compile_trace response_id=%s run_id=%s turn_id=%s origin=%s input_event_key=%s compile_keys_considered=%s pref_recall_found=%s pref_key_used=%s pref_len=%s",
            str(getattr(api, "_active_response_id", "") or ""),
            api._current_run_id() or "",
            turn_id,
            origin,
            current_input_event_key or "unknown",
            deduped_compile_keys,
            str(pref_len > 0).lower(),
            canonical_key if pref_len > 0 else "",
            pref_len,
            )
        logger.debug(
            "response_prompt_compiled canonical_key=%s includes_pref_recall=%s pref_len=%s",
            canonical_key,
            str(pref_len > 0).lower(),
            pref_len,
        )
        if str(origin or "").strip().lower() == "server_auto":
            preference_payload = None
            if prepared_snapshot.had_pending_preference_context and hasattr(api, "_peek_pending_preference_memory_context_payload"):
                preference_payload = api._peek_pending_preference_memory_context_payload(
                    turn_id=turn_id,
                    input_event_key=current_input_event_key,
                )
            pref_hit = bool(isinstance(preference_payload, dict) and preference_payload.get("hit", False))
            attached_sources = []
            if isinstance(preference_payload, dict):
                source = str(preference_payload.get("source") or "").strip()
                if source:
                    attached_sources.append(source)
            logger.info(
                "response_context_snapshot run_id=%s turn_id=%s input_event_key=%s canonical_key=%s origin=%s pref_ctx_present=%s pref_hit=%s pref_text_len=%s attached_sources=%s",
                api._current_run_id() or "",
                turn_id,
                current_input_event_key or "unknown",
                canonical_key,
                str(origin or "").strip().lower(),
                str(pref_len > 0).lower(),
                str(pref_hit).lower(),
                pref_len,
                    ",".join(attached_sources) if attached_sources else "none",
                )
        if pref_len == 0 and prepared_snapshot.had_pending_preference_context:
            logger.debug(
                "response_prompt_compile_missing_pref_recall response_id=%s run_id=%s turn_id=%s origin=%s input_event_key=%s reason=no_alias",
                str(getattr(api, "_active_response_id", "") or ""),
                api._current_run_id() or "",
                turn_id,
                origin,
                current_input_event_key or "unknown",
            )
            logger.debug(
                "pref_recall_excluded canonical_key=%s reason=missing_prompt_note",
                canonical_key,
            )

        if prepared_snapshot.tool_followup:
            parent_turn_id = str(response_metadata.get("parent_turn_id") or turn_id or "").strip() or turn_id
            parent_input_event_key = str(response_metadata.get("parent_input_event_key") or "").strip() or None
            if hasattr(api, "_should_suppress_tool_followup_after_turn_deliverable") and api._should_suppress_tool_followup_after_turn_deliverable(
                turn_id=parent_turn_id,
                parent_input_event_key=parent_input_event_key,
            ):
                logger.info(
                    "tool_followup_create_suppressed canonical_key=%s reason=final_deliverable_already_sent parent_turn_id=%s",
                    canonical_key,
                    parent_turn_id,
                )
                api._set_tool_followup_state(
                    canonical_key=canonical_key,
                    state="dropped",
                    reason="final_deliverable_already_sent",
                )
                self._log_response_create_outcome(
                    snapshot=prepared_snapshot,
                    decision=self._build_execution_decision(
                        action=ResponseCreateOutcomeAction.DROP,
                        reason_code="tool_followup_final_deliverable_already_sent",
                        explanation="Tool followup dropped because the parent turn already has a final deliverable.",
                        selected_candidate_id="tool_followup_final_deliverable",
                    ),
                )
                return False
            if prepared_snapshot.tool_followup_release and prepared_snapshot.tool_followup_state in {"scheduled", "scheduled_release", "blocked_active_response", "released_on_response_done"}:
                pass
            elif prepared_snapshot.tool_followup_state != "new":
                deny_reason = f"already_{prepared_snapshot.tool_followup_state}"
                logger.info(
                    "tool_followup_arbitration outcome=deny reason=%s call_id=%s canonical_key=%s",
                    deny_reason,
                    prepared_snapshot.tool_call_id or "unknown",
                    canonical_key,
                )
                logger.info(
                    "tool_followup_create_suppressed canonical_key=%s reason=%s prior_state=%s",
                    canonical_key,
                    deny_reason,
                    prepared_snapshot.tool_followup_state,
                )
                self._log_response_create_outcome(
                    snapshot=prepared_snapshot,
                    decision=self._build_execution_decision(
                        action=ResponseCreateOutcomeAction.DROP,
                        reason_code=deny_reason,
                        explanation=f"Tool followup dropped because it was already {prepared_snapshot.tool_followup_state}.",
                        selected_candidate_id="tool_followup_existing_state",
                    ),
                )
                return False

        self._log_response_create_outcome(snapshot=prepared_snapshot, decision=decision)
        if decision.should_log_arbitration:
            logger.debug(
                "arbitration_decision surface=response_create action=%s reason_code=%s selected_candidate_id=%s turn_id=%s canonical_key=%s",
                decision.action.value.lower(),
                decision.reason_code,
                decision.selected_candidate_id,
                turn_id,
                canonical_key,
            )
        if prepared_snapshot.tool_followup:
            arbitration_outcome = (
                "deny" if decision.action in {ResponseCreateOutcomeAction.BLOCK, ResponseCreateOutcomeAction.DROP} else "allow"
            )
            logger.info(
                "tool_followup_arbitration outcome=%s reason=%s call_id=%s canonical_key=%s",
                arbitration_outcome,
                decision.reason_code,
                prepared_snapshot.tool_call_id or "unknown",
                canonical_key,
            )

        if decision.action is ResponseCreateOutcomeAction.SCHEDULE:
            return api._schedule_pending_response_create(
                websocket=websocket,
                response_create_event=response_create_event,
                origin=origin,
                reason=str(decision.queue_reason or decision.reason_code),
                record_ai_call=record_ai_call,
                debug_context=debug_context,
                memory_brief_note=prepared_snapshot.effective_memory_note,
                emit_outcome_log=False,
            )
        if decision.action is ResponseCreateOutcomeAction.DROP:
            if prepared_snapshot.tool_followup:
                api._set_tool_followup_state(
                    canonical_key=canonical_key,
                    state="dropped",
                    reason=decision.reason_code,
                )
            return False
        if decision.action is ResponseCreateOutcomeAction.BLOCK:
            if decision.reason_code == "already_delivered":
                api._record_duplicate_create_attempt(
                    turn_id=turn_id,
                    canonical_key=canonical_key,
                    reason="already_delivered",
                )
                logger.debug(
                    "duplicate_response_prevented run_id=%s turn_id=%s input_event_key=%s reason=already_delivered origin=%s",
                    api._current_run_id() or "",
                    turn_id,
                    current_input_event_key or "unknown",
                    normalized_origin,
                )
                return False
            if decision.reason_code == "preference_recall_lock_blocked":
                return False
            if prepared_snapshot.single_flight_block_reason and decision.reason_code == prepared_snapshot.single_flight_block_reason:
                self._note_response_create_blocked(
                    canonical_key=canonical_key,
                    reason=decision.reason_code,
                )
                if normalized_origin == "tool_output" and decision.reason_code == "already_created":
                    followup_event = self._build_tool_output_followup_event(
                        response_create_event=response_create_event,
                        turn_id=turn_id,
                        input_event_key=current_input_event_key or "unknown",
                    )
                    logger.info(
                        "response_schedule_followup run_id=%s turn_id=%s origin=%s reason=already_created mode=tool_output_followup",
                        api._current_run_id() or "",
                        turn_id,
                        normalized_origin,
                    )
                    return api._schedule_pending_response_create(
                        websocket=websocket,
                        response_create_event=followup_event,
                        origin=origin,
                        reason="tool_output_followup_already_created",
                        record_ai_call=record_ai_call,
                        debug_context=debug_context,
                        memory_brief_note=memory_brief_note,
                    )
                api._log_response_create_blocked(
                    turn_id=turn_id,
                    origin=normalized_origin,
                    input_event_key=current_input_event_key,
                    canonical_key=canonical_key,
                    block_reason=decision.reason_code,
                )
                if decision.reason_code == "canonical_response_already_created":
                    api._record_duplicate_create_attempt(
                        turn_id=turn_id,
                        canonical_key=canonical_key,
                        reason=decision.reason_code,
                    )
                return False
            if decision.reason_code == "preference_recall_suppressed":
                self._note_response_create_blocked(
                    canonical_key=canonical_key,
                    reason=decision.reason_code,
                )
                api._drop_suppressed_scheduled_response_creates(turn_id=turn_id, origin=normalized_origin)
                api._mark_transcript_response_outcome(
                    input_event_key=current_input_event_key,
                    turn_id=turn_id,
                    outcome="response_not_scheduled",
                    reason=decision.reason_code,
                    details="direct response.create blocked",
                )
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=direct reason=%s canonical_key=%s",
                api._current_run_id() or "",
                turn_id,
                normalized_origin,
                (decision.reason_code if decision is not None else "scheduled_release"),
                canonical_key,
            )
            return False

        memory_note_to_send = memory_brief_note
        if not memory_note_to_send and hasattr(api, "_consume_pending_preference_memory_context_note"):
            memory_note_to_send = api._consume_pending_preference_memory_context_note(
                turn_id=turn_id,
                input_event_key=current_input_event_key,
            )
        if memory_note_to_send:
            try:
                await api._send_memory_brief_note(websocket, memory_note_to_send)
            except Exception as exc:  # pragma: no cover - defensive fail-open
                logger.warning("Memory brief injection skipped due to error: %s", exc)

        obligation_present = api._response_obligation_key(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        ) in getattr(api, "_response_obligations", {})
        pending_server_auto_len = len(getattr(api, "_pending_server_auto_input_event_keys", deque()) or ())
        logger.debug(
            "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
            "canonical_key=%s sent_now=false scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
            "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
            api._current_run_id() or "",
            turn_id,
            normalized_origin,
            decision.reason_code,
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(api, "_response_create_queue", deque()) or ()),
            pending_server_auto_len,
            bool(turn_id in prepared_snapshot.suppression_turns),
            api._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
            obligation_present,
            getattr(api, "_response_done_serial", 0),
        )
        schedule_logged_turn_ids = getattr(api, "_response_schedule_logged_turn_ids", None)
        if not isinstance(schedule_logged_turn_ids, set):
            schedule_logged_turn_ids = set()
            api._response_schedule_logged_turn_ids = schedule_logged_turn_ids
        if turn_id not in schedule_logged_turn_ids:
            schedule_logged_turn_ids.add(turn_id)
            turn_timestamps_store = getattr(api, "_turn_diagnostic_timestamps", None)
            if not isinstance(turn_timestamps_store, dict):
                turn_timestamps_store = {}
                api._turn_diagnostic_timestamps = turn_timestamps_store
            turn_timestamps = turn_timestamps_store.setdefault(turn_id, {})
            turn_timestamps["response_schedule"] = now
            logger.info(
                "response_schedule_marker run_id=%s turn_id=%s origin=%s input_event_key=%s canonical_key=%s suppression_active=%s mode=direct",
                api._current_run_id() or "",
                turn_id,
                origin,
                current_input_event_key or "unknown",
                canonical_key,
                prepared_snapshot.suppression_active,
            )
            logger.debug(
                "turn_diagnostic_timestamps run_id=%s turn_id=%s transcript_final_ts=%s preference_recall_start_ts=%s preference_recall_end_ts=%s response_schedule_ts=%s",
                api._current_run_id() or "",
                turn_id,
                turn_timestamps.get("transcript_final"),
                turn_timestamps.get("preference_recall_start"),
                turn_timestamps.get("preference_recall_end"),
                turn_timestamps.get("response_schedule"),
            )
        if prepared_snapshot.tool_followup and not prepared_snapshot.tool_followup_release:
            api._set_tool_followup_state(
                canonical_key=canonical_key,
                state="creating",
                reason="direct_send",
            )
        elif prepared_snapshot.tool_followup and prepared_snapshot.tool_followup_release:
            if prepared_snapshot.tool_followup_state in {"scheduled", "scheduled_release", "blocked_active_response", "released_on_response_done"}:
                api._set_tool_followup_state(
                    canonical_key=canonical_key,
                    state="released_on_response_done",
                    reason=f"queue_release trigger={decision.reason_code}",
                )
                api._set_tool_followup_state(
                    canonical_key=canonical_key,
                    state="creating",
                    reason="released_on_response_done",
                )
        log_ws_event("Outgoing", response_create_event)
        api._track_outgoing_event(response_create_event, origin=origin)
        transport = api._get_or_create_transport()
        await transport.send_json(websocket, response_create_event)
        if prepared_snapshot.consumes_canonical_slot:
            api._set_response_delivery_state(
                turn_id=turn_id,
                input_event_key=current_input_event_key,
                state="created",
            )
        api._mark_input_event_key_scheduled(input_event_key=current_input_event_key)
        api._last_response_create_ts = now
        api._response_in_flight = True
        if record_ai_call:
            api._record_ai_call()
        logger.debug(
            "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
            "canonical_key=%s sent_now=true scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
            "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
            api._current_run_id() or "",
            turn_id,
            normalized_origin,
            decision.reason_code,
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(api, "_response_create_queue", deque()) or ()),
            len(getattr(api, "_pending_server_auto_input_event_keys", deque()) or ()),
            bool(turn_id in prepared_snapshot.suppression_turns),
            api._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
            obligation_present,
            getattr(api, "_response_done_serial", 0),
        )
        return True

    async def drain_response_create_queue(self, source_trigger: str | None = None) -> None:
        api = self.api
        source_candidate = source_trigger
        if source_candidate is None:
            source_candidate = getattr(api, "_response_create_queue_drain_source", "explicit_caller")
        api._response_create_queue_drain_source = "explicit_caller"
        normalized_source_trigger = str(source_candidate or "").strip().lower()
        if normalized_source_trigger not in {"response_done", "playback_complete", "websocket_close", "active_cleared"}:
            normalized_source_trigger = "explicit_caller"

        selected_pending_origin = "none"
        selected_pending_turn_id = "none"
        selected_pending_input_event_key = "unknown"
        selected_pending_canonical_key = "unknown"
        selected_pending_trigger = "none"
        selected_pending_enqueued_done_serial = "none"
        selected_pending_serial_relation = "none"
        drain_result = "none"
        skipped_reason = "none"

        def _serial_relation(serial: int | None) -> str:
            if serial is None:
                return "none"
            if serial < api._response_done_serial:
                return "older"
            if serial > api._response_done_serial:
                return "newer"
            return "equal"

        def _emit_drain_trace(*, stage: str, queue_len_before_value: int, queue_len_after_value: int) -> None:
            logger.debug(
                "[RESPTRACE] queue_drain_%s source_trigger=%s run_id=%s turn_id=%s input_event_key=%s canonical_key=%s "
                "queue_len_before=%s queue_len_after=%s picked_origin=%s picked_turn_id=%s "
                "picked_input_event_key=%s picked_canonical_key=%s selected_pending_trigger=%s skipped_reason=%s "
                "enqueued_done_serial=%s enqueued_done_serial_relation=%s response_done_serial=%s drain_result=%s",
                stage,
                normalized_source_trigger,
                api._current_run_id() or "",
                trace_turn_id,
                trace_input_event_key or "unknown",
                trace_canonical_key,
                queue_len_before_value,
                queue_len_after_value,
                selected_pending_origin,
                selected_pending_turn_id,
                selected_pending_input_event_key,
                selected_pending_canonical_key,
                selected_pending_trigger,
                skipped_reason,
                selected_pending_enqueued_done_serial,
                selected_pending_serial_relation,
                getattr(api, "_response_done_serial", 0),
                drain_result,
            )

        current_state = getattr(api.state_manager, "state", InteractionState.IDLE)
        queue_len_before = len(getattr(api, "_response_create_queue", deque()) or ())
        trace_turn_id = api._current_turn_id_or_unknown()
        trace_input_event_key = str(getattr(api, "_current_input_event_key", "") or "").strip()
        trace_canonical_key = api._canonical_utterance_key(
            turn_id=trace_turn_id,
            input_event_key=trace_input_event_key,
        )
        _emit_drain_trace(
            stage="pre",
            queue_len_before_value=queue_len_before,
            queue_len_after_value=queue_len_before,
        )
        if (
            api._is_active_response_blocking()
            or api._audio_playback_busy
            or current_state == InteractionState.LISTENING
        ):
            drain_result = "state_or_flight_gate"
            _emit_drain_trace(
                stage="post",
                queue_len_before_value=queue_len_before,
                queue_len_after_value=queue_len_before,
            )
            return

        if api._pending_response_create is None and api._response_create_queue:
            api._dedupe_queued_confirmation_reminders()
            queued_candidates: list[tuple[int, int, dict[str, Any], dict[str, Any], str, str, str]] = []
            retained_queue_entries: list[dict[str, Any]] = []
            for queued in list(api._response_create_queue):
                metadata = api._extract_response_create_metadata(queued.get("event") or {})
                queued_trigger = api._extract_response_create_trigger(metadata)
                picked_turn_id = str(queued.get("turn_id") or "turn-unknown")
                picked_origin = str(queued.get("origin") or "unknown")
                picked_input_event_key = str(metadata.get("input_event_key") or "").strip()
                canonical_key = api._canonical_utterance_key(
                    turn_id=picked_turn_id,
                    input_event_key=picked_input_event_key,
                )
                if api._drop_response_create_for_terminal_state(
                    turn_id=picked_turn_id,
                    input_event_key=picked_input_event_key,
                    origin=picked_origin,
                    response_metadata=metadata,
                ):
                    skipped_reason = "canonical_terminal_state"
                    continue
                retained_queue_entries.append(queued)
                if not api._can_release_queued_response_create(queued_trigger, metadata):
                    skipped_reason = "release_gate_blocked"
                    continue
                queued_candidates.append(
                    (
                        api._response_create_queue_priority(origin=picked_origin, canonical_key=canonical_key),
                        int(queued.get("enqueue_seq") or 0),
                        queued,
                        metadata,
                        queued_trigger,
                        picked_turn_id,
                        picked_input_event_key,
                    )
                )
            api._response_create_queue = deque(retained_queue_entries)
            if queued_candidates:
                queued_candidates.sort(key=lambda item: (-item[0], item[1]))
                _, _, queued, metadata, queued_trigger, picked_turn_id, picked_input_event_key = queued_candidates[0]
                try:
                    api._response_create_queue.remove(queued)
                except ValueError:
                    pass
                picked_origin = str(queued.get("origin") or "unknown")
                picked_input_event_key = picked_input_event_key or "unknown"
                enqueued_done_serial_value = int(queued.get("enqueued_done_serial") or api._response_done_serial)
                selected_pending_origin = picked_origin
                selected_pending_turn_id = picked_turn_id
                selected_pending_input_event_key = picked_input_event_key
                selected_pending_canonical_key = api._canonical_utterance_key(
                    turn_id=picked_turn_id,
                    input_event_key=str(metadata.get("input_event_key") or "").strip(),
                )
                selected_pending_trigger = queued_trigger
                selected_pending_enqueued_done_serial = str(enqueued_done_serial_value)
                selected_pending_serial_relation = _serial_relation(enqueued_done_serial_value)
                api._pending_response_create = PendingResponseCreate(
                    websocket=queued["websocket"],
                    event=queued["event"],
                    origin=queued["origin"],
                    turn_id=str(queued.get("turn_id") or api._next_response_turn_id()),
                    created_at=time.monotonic(),
                    reason="legacy_queue_hydration",
                    record_ai_call=bool(queued.get("record_ai_call", False)),
                    debug_context=queued.get("debug_context"),
                    memory_brief_note=queued.get("memory_brief_note"),
                    queued_reminder_key=api._queued_response_reminder_key(queued),
                    enqueued_done_serial=enqueued_done_serial_value,
                    enqueue_seq=int(queued.get("enqueue_seq") or 0),
                )
        if api._pending_response_create is None:
            drain_result = skipped_reason if skipped_reason != "none" else "no_pending"
            _emit_drain_trace(
                stage="post",
                queue_len_before_value=queue_len_before,
                queue_len_after_value=len(getattr(api, "_response_create_queue", deque()) or ()),
            )
            return

        pending = api._pending_response_create
        response_metadata = api._extract_response_create_metadata(pending.event)
        queued_trigger = api._extract_response_create_trigger(response_metadata)
        pending_input_event_key = str(response_metadata.get("input_event_key") or "").strip()
        selected_pending_origin = pending.origin
        selected_pending_turn_id = pending.turn_id
        selected_pending_input_event_key = pending_input_event_key or "unknown"
        selected_pending_canonical_key = api._canonical_utterance_key(
            turn_id=pending.turn_id,
            input_event_key=pending_input_event_key,
        )
        selected_pending_trigger = queued_trigger
        selected_pending_enqueued_done_serial = str(pending.enqueued_done_serial)
        selected_pending_serial_relation = _serial_relation(pending.enqueued_done_serial)
        if not api._can_release_queued_response_create(queued_trigger, response_metadata):
            if pending.reason != "legacy_queue_hydration":
                api._sync_pending_response_create_queue()
            logger.info(
                "Deferring queued response.create origin=%s trigger=%s while awaiting confirmation.",
                pending.origin,
                queued_trigger,
            )
            drain_result = "release_gate_blocked"
            _emit_drain_trace(
                stage="post",
                queue_len_before_value=queue_len_before,
                queue_len_after_value=len(getattr(api, "_response_create_queue", deque()) or ()),
            )
            return

        if api._drop_response_create_for_terminal_state(
            turn_id=pending.turn_id,
            input_event_key=pending_input_event_key,
            origin=pending.origin,
            response_metadata=response_metadata,
        ):
            api._pending_response_create = None
            if pending.reason != "legacy_queue_hydration":
                api._sync_pending_response_create_queue()
            drain_result = "dropped_terminal_state"
            skipped_reason = "canonical_terminal_state"
            _emit_drain_trace(
                stage="post",
                queue_len_before_value=queue_len_before,
                queue_len_after_value=len(getattr(api, "_response_create_queue", deque()) or ()),
            )
            return

        pending_canonical_key = api._canonical_utterance_key(
            turn_id=pending.turn_id,
            input_event_key=pending_input_event_key,
        )
        if api._should_drop_tool_followup_at_create_seam(
            turn_id=pending.turn_id,
            response_metadata=response_metadata,
            canonical_key=pending_canonical_key,
            drain_trigger=normalized_source_trigger,
        ):
            api._pending_response_create = None
            if pending.reason != "legacy_queue_hydration":
                queue = getattr(api, "_response_create_queue", None)
                if isinstance(queue, deque):
                    removed = False
                    kept: deque[dict[str, Any]] = deque()
                    for queued in queue:
                        metadata = api._extract_response_create_metadata(queued.get("event") or {})
                        queued_canonical_key = api._canonical_utterance_key(
                            turn_id=str(queued.get("turn_id") or "").strip(),
                            input_event_key=str(metadata.get("input_event_key") or "").strip(),
                        )
                        if not removed and queued_canonical_key == pending_canonical_key:
                            removed = True
                            continue
                        kept.append(queued)
                    api._response_create_queue = kept
                api._sync_pending_response_create_queue()
            drain_result = "dropped_parent_covered"
            skipped_reason = "parent_covered_tool_result"
            _emit_drain_trace(
                stage="post",
                queue_len_before_value=queue_len_before,
                queue_len_after_value=len(getattr(api, "_response_create_queue", deque()) or ()),
            )
            return

        api._pending_response_create = None
        if pending.reason != "legacy_queue_hydration":
            queue = getattr(api, "_response_create_queue", None)
            if isinstance(queue, deque):
                removed = False
                kept: deque[dict[str, Any]] = deque()
                for queued in queue:
                    metadata = api._extract_response_create_metadata(queued.get("event") or {})
                    queued_canonical_key = api._canonical_utterance_key(
                        turn_id=str(queued.get("turn_id") or "").strip(),
                        input_event_key=str(metadata.get("input_event_key") or "").strip(),
                    )
                    if not removed and queued_canonical_key == pending_canonical_key:
                        removed = True
                        continue
                    kept.append(queued)
                api._response_create_queue = kept
            api._sync_pending_response_create_queue()
        await api._send_response_create(
            pending.websocket,
            pending.event,
            origin=pending.origin,
            record_ai_call=pending.record_ai_call,
            debug_context=pending.debug_context,
            memory_brief_note=pending.memory_brief_note,
        )
        api._response_create_drains_total = int(getattr(api, "_response_create_drains_total", 0) or 0) + 1
        logger.info(
            "response_create_queue_drained triggered_by=%s drained_count=%s qsize_after=%s",
            normalized_source_trigger,
            int(getattr(api, "_response_create_drains_total", 0) or 0),
            len(getattr(api, "_response_create_queue", deque()) or ()),
        )
        drain_result = "sent_to_send_response_create"
        _emit_drain_trace(
            stage="post",
            queue_len_before_value=queue_len_before,
            queue_len_after_value=len(getattr(api, "_response_create_queue", deque()) or ()),
        )
