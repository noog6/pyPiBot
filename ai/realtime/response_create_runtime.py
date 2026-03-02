"""Runtime service for queued response.create arbitration."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

from ai.interaction_lifecycle_policy import ResponseCreateDecisionAction
from ai.realtime.types import PendingResponseCreate
from core.logging import logger


class ResponseCreateRuntimeAPI(Protocol):
    ...


@dataclass
class ResponseCreateRuntime:
    api: ResponseCreateRuntimeAPI

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
        consumes_canonical_slot = api._response_consumes_canonical_slot(response_metadata)
        explicit_multipart = api._response_is_explicit_multipart(response_metadata)
        if consumes_canonical_slot and api._canonical_first_audio_started(canonical_key) and not explicit_multipart:
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
            single_flight_block_reason = api._single_flight_block_reason(
                turn_id=turn_id,
                input_event_key=current_input_event_key,
            )
            if single_flight_block_reason:
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
        ):
            logger.info(
                "response_schedule_blocked run_id=%s turn_id=%s origin=%s mode=queued reason=canonical_audio_already_started canonical_key=%s",
                api._current_run_id() or "",
                turn_id,
                normalized_origin,
                canonical_key,
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
        suppression_active = turn_id in suppression_turns and not current_input_event_key
        if suppression_active and normalized_origin == "server_auto":
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
        )
        previous = api._pending_response_create
        if previous is None:
            api._pending_response_create = candidate
            api._sync_pending_response_create_queue()
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
    
        should_replace = False
        replacement_reason = ""
        if candidate.turn_id != previous.turn_id:
            should_replace = True
            replacement_reason = "newer_turn"
        else:
            candidate_priority = api._response_create_priority(candidate.origin)
            previous_priority = api._response_create_priority(previous.origin)
            if candidate_priority > previous_priority:
                should_replace = True
                replacement_reason = "higher_priority"
            elif candidate_priority == previous_priority:
                should_replace = True
                replacement_reason = "latest_update"
    
        if should_replace:
            api._pending_response_create = candidate
            api._sync_pending_response_create_queue()
            if reason == "audio_playback_busy":
                logger.debug(
                    "playback_busy_retry_enqueued run_id=%s input_event_key=%s reason=audio_playback_busy",
                    api._current_run_id() or "",
                    current_input_event_key or "unknown",
                )
            logger.info(
                "response_create_replaced turn_id=%s old_origin=%s new_origin=%s reason=%s",
                candidate.turn_id,
                previous.origin,
                candidate.origin,
                replacement_reason,
            )
            api._log_response_site_debug(
                site="response_create_replaced",
                turn_id=turn_id,
                input_event_key=current_input_event_key,
                canonical_key=canonical_key,
                origin=normalized_origin,
                trigger=replacement_reason,
            )
            logger.debug(
                "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
                "canonical_key=%s sent_now=false scheduled=true replaced=true queue_len=%s pending_server_auto_len=%s "
                "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
                api._current_run_id() or "",
                turn_id,
                normalized_origin,
                replacement_reason,
                current_input_event_key or "unknown",
                canonical_key,
                len(getattr(api, "_response_create_queue", deque()) or ()),
                len(getattr(api, "_pending_server_auto_input_event_keys", deque()) or ()),
                bool(turn_id in suppression_turns),
                api._resptrace_suppression_reason(turn_id=turn_id, input_event_key=current_input_event_key),
                obligation_present,
                getattr(api, "_response_done_serial", 0),
            )
        else:
            logger.info(
                "response_create_dropped turn_id=%s origin=%s reason=%s",
                candidate.turn_id,
                candidate.origin,
                "lower_priority_than_pending",
            )
            logger.debug(
                "[RESPTRACE] response_create_request run_id=%s turn_id=%s origin=%s trigger_reason=%s input_event_key=%s "
                "canonical_key=%s sent_now=false scheduled=false replaced=false queue_len=%s pending_server_auto_len=%s "
                "suppression_active=%s suppression_reason=%s obligation_present=%s response_done_serial=%s",
                api._current_run_id() or "",
                turn_id,
                normalized_origin,
                "lower_priority_than_pending",
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
    

