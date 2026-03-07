"""Runtime service for queued response.create arbitration."""

from __future__ import annotations

import time
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from ai.interaction_lifecycle_policy import ResponseCreateDecisionAction
from ai.realtime.types import PendingResponseCreate
from core.logging import logger, log_ws_event
from interaction import InteractionState
from ai.realtime.types import UtteranceContext


class ResponseCreateRuntimeAPI(Protocol):
    ...


@dataclass
class ResponseCreateRuntime:
    api: ResponseCreateRuntimeAPI

    def _note_response_create_blocked(self, *, canonical_key: str, reason: str) -> None:
        api = self.api
        api._response_create_queued_creates_total = int(getattr(api, "_response_create_queued_creates_total", 0) or 0) + 1
        api._sync_pending_response_create_queue()
        if str(reason or "").strip().lower() == "active_response":
            logger.info(
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
    ) -> bool:
        api = self.api
        turn_id = api._resolve_response_create_turn_id(origin=origin, response_create_event=response_create_event)
        current_input_event_key = api._ensure_response_create_correlation(
            response_create_event=response_create_event,
            origin=origin,
            turn_id=turn_id,
        )
        with api._utterance_context_scope(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        ) as resolved_context:
            turn_id = resolved_context.turn_id
            current_input_event_key = resolved_context.input_event_key
            canonical_key = resolved_context.canonical_key
        response_metadata = api._extract_response_create_metadata(response_create_event)
        normalized_origin = str(origin or "").strip().lower()
        tool_followup = str(response_metadata.get("tool_followup", "")).strip().lower() in {"true", "1", "yes"}
        tool_call_id = str(response_metadata.get("tool_call_id") or "").strip()
        if tool_followup:
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
                return False
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
            if current_state == "scheduled":
                logger.info(
                    "tool_followup_create_suppressed canonical_key=%s reason=already_scheduled prior_state=%s",
                    canonical_key,
                    current_state,
                )
                return False
            if current_state in {"blocked_active_response", "released_on_response_done"}:
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
        created_keys = getattr(api, "_response_created_canonical_keys", set())
        if consumes_canonical_slot:
            single_flight_block_reason = "" if transcript_upgrade_replacement else api._single_flight_block_reason(
                turn_id=turn_id,
                input_event_key=current_input_event_key,
            )
            if single_flight_block_reason:
                self._note_response_create_blocked(
                    canonical_key=canonical_key,
                    reason=single_flight_block_reason,
                )
                api._log_response_create_blocked(
                    turn_id=turn_id,
                    origin=normalized_origin,
                    input_event_key=current_input_event_key,
                    canonical_key=canonical_key,
                    block_reason=single_flight_block_reason,
                )
                return False
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
        if api._is_response_already_delivered(turn_id=turn_id, input_event_key=current_input_event_key):
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
            if reason == "audio_playback_busy":
                logger.debug(
                    "playback_busy_retry_dropped run_id=%s input_event_key=%s reason=already_delivered",
                    api._current_run_id() or "",
                    current_input_event_key or "unknown",
                )
            return False
        if api._is_preference_recall_lock_blocked(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            normalized_origin=normalized_origin,
            response_metadata=response_metadata,
        ):
            return False
        if (
            consumes_canonical_slot
            and api._canonical_first_audio_started(canonical_key)
            and not api._response_is_explicit_multipart(response_metadata)
            and not allow_audio_started_upgrade
        ):
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=queued reason=canonical_audio_already_started canonical_key=%s",
                api._current_run_id() or "",
                turn_id,
                normalized_origin,
                canonical_key,
            )
            return False
        if normalized_origin == "assistant_message":
            owner_reason = ""
            owner_fn = getattr(api, "_assistant_message_same_turn_owner_reason", None)
            if callable(owner_fn):
                owner_reason = str(
                    owner_fn(
                        turn_id=turn_id,
                        input_event_key=current_input_event_key,
                        canonical_key=canonical_key,
                    )
                    or ""
                ).strip()
            if owner_reason:
                logger.info(
                    "response_not_scheduled run_id=%s turn_id=%s input_event_key=%s reason=same_turn_already_owned details=owner=%s canonical_key=%s origin=%s",
                    api._current_run_id() or "",
                    turn_id,
                    current_input_event_key or "unknown",
                    owner_reason,
                    canonical_key,
                    normalized_origin,
                )
                api._mark_transcript_response_outcome(
                    input_event_key=current_input_event_key,
                    turn_id=turn_id,
                    outcome="response_not_scheduled",
                    reason="same_turn_already_owned",
                    details=f"owner={owner_reason} canonical_key={canonical_key} origin={normalized_origin}",
                )
                return False

        if (
            consumes_canonical_slot
            and canonical_key in created_keys
            and not api._response_has_safety_override(response_create_event)
        ):
            api._record_duplicate_create_attempt(
                turn_id=turn_id,
                canonical_key=canonical_key,
                reason="canonical_response_already_created",
            )
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=queued reason=canonical_response_already_created canonical_key=%s",
                api._current_run_id() or "",
                turn_id,
                normalized_origin,
                canonical_key,
            )
            return False
        suppression_active = (turn_id in suppression_turns) and (not str(current_input_event_key or "").strip())
        if suppression_active and normalized_origin == "server_auto":
            self._note_response_create_blocked(
                canonical_key=canonical_key,
                reason="preference_recall_suppressed",
            )
            api._drop_suppressed_scheduled_response_creates(turn_id=turn_id, origin=normalized_origin)
            api._mark_transcript_response_outcome(
                input_event_key=current_input_event_key,
                turn_id=turn_id,
                outcome="response_not_scheduled",
                reason="preference_recall_suppressed",
                details="queued response.create blocked",
            )
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=queued reason=preference_recall_suppressed",
                api._current_run_id() or "",
                turn_id,
                normalized_origin,
            )
            return False
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
            logger.info(
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
        logger.info(
            "response_create_scheduled turn_id=%s origin=%s reason=%s",
            candidate.turn_id,
            candidate.origin,
            reason,
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
            reason,
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
        )
        with api._utterance_context_scope(turn_id=turn_id, input_event_key=current_input_event_key) as resolved_context:
            turn_id = resolved_context.turn_id
            current_input_event_key = resolved_context.input_event_key
            canonical_key = resolved_context.canonical_key

        pref_len = len(str(preference_note or ""))
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
        if pref_len == 0 and had_pending_preference_context:
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

        response_metadata = api._extract_response_create_metadata(response_create_event)
        normalized_origin = str(origin or "").strip().lower()
        tool_followup = str(response_metadata.get("tool_followup", "")).strip().lower() in {"true", "1", "yes"}
        tool_call_id = str(response_metadata.get("tool_call_id") or "").strip()
        tool_followup_release = str(response_metadata.get("tool_followup_release", "")).strip().lower() in {"true", "1", "yes"}
        tool_followup_state = "new"
        if tool_followup:
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
                return False
            tool_followup_state = api._tool_followup_state(canonical_key=canonical_key)
            if tool_followup_release and tool_followup_state in {"scheduled", "scheduled_release", "blocked_active_response", "released_on_response_done"}:
                pass
            elif tool_followup_state != "new":
                deny_reason = f"already_{tool_followup_state}"
                logger.info(
                    "tool_followup_arbitration outcome=deny reason=%s call_id=%s canonical_key=%s",
                    deny_reason,
                    tool_call_id or "unknown",
                    canonical_key,
                )
                logger.info(
                    "tool_followup_create_suppressed canonical_key=%s reason=%s prior_state=%s",
                    canonical_key,
                    deny_reason,
                    tool_followup_state,
                )
                return False
        consumes_canonical_slot = api._response_consumes_canonical_slot(response_metadata)
        explicit_multipart = api._response_is_explicit_multipart(response_metadata)
        transcript_upgrade_replacement = str(response_metadata.get("transcript_upgrade_replacement", "")).strip().lower() in {"true", "1", "yes"}
        allow_audio_started_upgrade = api._should_upgrade_with_audio_started(
            response_metadata,
            origin=normalized_origin,
        )
        suppression_turns = getattr(api, "_preference_recall_suppressed_turns", set())
        created_keys = getattr(api, "_response_created_canonical_keys", set())
        single_flight_block_reason = ("" if transcript_upgrade_replacement else api._single_flight_block_reason(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
        )) if consumes_canonical_slot else ""
        suppression_active = (turn_id in suppression_turns) and (not str(current_input_event_key or "").strip())

        awaiting_transcript_final = False
        if normalized_origin == "server_auto":
            missing_for_turn = getattr(api, "_transcript_final_missing_for_turn", None)
            if callable(missing_for_turn):
                awaiting_transcript_final = bool(
                    missing_for_turn(
                        turn_id=turn_id,
                        input_event_key=current_input_event_key,
                    )
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
        if api._drop_response_create_for_terminal_state(
            turn_id=turn_id,
            input_event_key=current_input_event_key,
            origin=origin,
            response_metadata=response_metadata,
        ):
            if tool_followup:
                api._set_tool_followup_state(
                    canonical_key=canonical_key,
                    state="dropped",
                    reason="canonical_terminal_state",
                )
            return False
        decision = api._lifecycle_policy().decide_response_create(
            response_in_flight=api._is_active_response_blocking(),
            audio_playback_busy=bool(api._audio_playback_busy),
            consumes_canonical_slot=consumes_canonical_slot,
            canonical_audio_started=(api._canonical_first_audio_started(canonical_key) and not allow_audio_started_upgrade),
            explicit_multipart=explicit_multipart,
            single_flight_block_reason=single_flight_block_reason,
            already_delivered=api._is_response_already_delivered(turn_id=turn_id, input_event_key=current_input_event_key),
            preference_recall_lock_blocked=preference_recall_lock_blocked,
            canonical_key_already_created=canonical_key in created_keys,
            has_safety_override=api._response_has_safety_override(response_create_event),
            suppression_active=suppression_active,
            normalized_origin=normalized_origin,
            awaiting_transcript_final=awaiting_transcript_final,
        )
        if tool_followup and decision is not None:
            arbitration_outcome = "deny" if decision.action is ResponseCreateDecisionAction.BLOCK else "allow"
            logger.info(
                "tool_followup_arbitration outcome=%s reason=%s call_id=%s canonical_key=%s",
                arbitration_outcome,
                (decision.reason_code if decision is not None else "scheduled_release"),
                tool_call_id or "unknown",
                canonical_key,
            )

        if decision is not None and decision.action is ResponseCreateDecisionAction.SCHEDULE:
            return api._schedule_pending_response_create(
                websocket=websocket,
                response_create_event=response_create_event,
                origin=origin,
                reason=str(decision.queue_reason or decision.reason_code),
                record_ai_call=record_ai_call,
                debug_context=debug_context,
                memory_brief_note=effective_memory_note,
            )
        if decision is not None and decision.action is ResponseCreateDecisionAction.BLOCK:
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
            if single_flight_block_reason and decision.reason_code == single_flight_block_reason:
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
            (decision.reason_code if decision is not None else "scheduled_release"),
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(api, "_response_create_queue", deque()) or ()),
            pending_server_auto_len,
            bool(turn_id in suppression_turns),
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
        if tool_followup and not tool_followup_release:
            api._set_tool_followup_state(
                canonical_key=canonical_key,
                state="creating",
                reason="direct_send",
            )
        elif tool_followup and tool_followup_release:
            if tool_followup_state in {"scheduled", "scheduled_release", "blocked_active_response", "released_on_response_done"}:
                api._set_tool_followup_state(
                    canonical_key=canonical_key,
                    state="released_on_response_done",
                    reason=f"queue_release trigger={(decision.reason_code if decision is not None else 'scheduled_release')}",
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
        if consumes_canonical_slot:
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
            (decision.reason_code if decision is not None else "scheduled_release"),
            current_input_event_key or "unknown",
            canonical_key,
            len(getattr(api, "_response_create_queue", deque()) or ()),
            len(getattr(api, "_pending_server_auto_input_event_keys", deque()) or ()),
            bool(turn_id in suppression_turns),
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

        api._pending_response_create = None
        if pending.reason != "legacy_queue_hydration":
            pending_canonical_key = api._canonical_utterance_key(
                turn_id=pending.turn_id,
                input_event_key=pending_input_event_key,
            )
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
