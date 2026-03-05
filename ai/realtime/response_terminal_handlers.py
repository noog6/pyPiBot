"""Handlers for terminal realtime response events."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import os
import time
from typing import TYPE_CHECKING, Any

from core.logging import log_info, logger
from interaction import InteractionState
from ai.orchestration import OrchestrationPhase
from ai.realtime.asr_trust import topic_mismatch_detected
from ai.utils import RUN_TIME_TABLE_LOG_JSON

if TYPE_CHECKING:
    from ai.realtime_api import RealtimeAPI


def _log_runtime(function_or_name: str, duration: float) -> None:
    os.makedirs(os.path.dirname(RUN_TIME_TABLE_LOG_JSON), exist_ok=True)
    time_record = {
        "timestamp": datetime.now().isoformat(),
        "function": function_or_name,
        "duration": f"{duration:.4f}",
    }
    with open(RUN_TIME_TABLE_LOG_JSON, "a", encoding="utf-8") as file:
        json.dump(time_record, file)
        file.write("\n")

    logger.info("⏰ %s() took %.4f seconds", function_or_name, duration)


class ResponseTerminalHandlers:
    """Owns response terminal event handlers for :class:`RealtimeAPI`."""

    def __init__(self, api: RealtimeAPI) -> None:
        self._api = api

    async def handle_transcribe_response_done(self) -> None:
        api = self._api
        if (
            api._active_response_confirmation_guarded
            and (api._has_active_confirmation_token() or api._pending_action is not None)
            and api._is_awaiting_confirmation_phase()
        ):
            api.assistant_reply = ""
            api._assistant_reply_accum = ""
            api._assistant_reply_response_id = None
            if api.websocket is not None:
                token = getattr(api, "_pending_confirmation_token", None)
                if token is not None:
                    if (
                        token is not None
                        and api._is_confirmation_prompt_latched(token)
                        and api._is_guarded_server_auto_reminder_allowed(reason="transcribe_response_done")
                    ):
                        await api._maybe_emit_confirmation_reminder(
                            api.websocket,
                            reason="transcribe_response_done",
                        )
                else:
                    if api._is_guarded_server_auto_reminder_allowed(reason="transcribe_response_done_legacy"):
                        await api._maybe_emit_confirmation_reminder(
                            api.websocket,
                            reason="transcribe_response_done_legacy",
                        )
            api._active_response_confirmation_guarded = False
            api.response_in_progress = False
            api._recover_confirmation_guard_microphone("transcribe_response_done")
            api._maybe_enqueue_reflection("response transcript done")
            if api._pending_image_stimulus and not api._pending_image_flush_after_playback:
                await api._flush_pending_image_stimulus("response transcript done")
            logger.info("Finished handle_transcribe_response_done()")
            return

        if api.assistant_reply:
            active_input_event_key = str(getattr(api, "_active_response_input_event_key", "") or "").strip()
            snapshot = getattr(api, "_utterance_trust_snapshot_by_input_event_key", {}).get(active_input_event_key, {})
            transcript_text = str(snapshot.get("transcript_text") or "")
            if (
                transcript_text
                and topic_mismatch_detected(transcript_text, api.assistant_reply)
                and bool(getattr(api, "_asr_verify_on_risk_enabled", False))
            ):
                websocket = getattr(api, "websocket", None)
                if websocket is not None:
                    await api.send_assistant_message(
                        "Did you mean the color of your pants right now, or your favorite color?",
                        websocket,
                        response_metadata={
                            "trigger": "topic_mismatch_detector",
                            "input_event_key": active_input_event_key,
                        },
                    )
                    logger.info(
                        "topic_mismatch_clarify run_id=%s input_event_key=%s transcript_anchors=%s",
                        api._current_run_id() or "",
                        active_input_event_key,
                        ",".join(snapshot.get("topic_anchors") or []),
                    )
                    api.assistant_reply = ""
                    api._assistant_reply_accum = ""
                    api._assistant_reply_response_id = None
                    api.response_in_progress = False
                    return
            api.assistant_reply = api._normalize_memory_recall_answer(api.assistant_reply)
            log_info(f"Assistant Response: {api.assistant_reply}", style="bold blue")
            api.assistant_reply = ""
            api._assistant_reply_response_id = None

        api.response_in_progress = False
        api._maybe_enqueue_reflection("response transcript done")
        if api._pending_image_stimulus and not api._pending_image_flush_after_playback:
            await api._flush_pending_image_stimulus("response transcript done")
        logger.info("Finished handle_transcribe_response_done()")

    async def handle_audio_response_done(self) -> None:
        api = self._api
        if api.response_start_time is not None:
            response_end_time = time.perf_counter()
            response_duration = response_end_time - api.response_start_time
            _log_runtime("realtime_api_response", response_duration)
            total_ms = response_duration * 1000.0
            if total_ms > 2000.0:
                turn_id = api._current_turn_id_or_unknown()
                turn_timestamps_store = getattr(api, "_turn_diagnostic_timestamps", {})
                turn_timestamps = turn_timestamps_store.get(turn_id, {}) if isinstance(turn_timestamps_store, dict) else {}
                transcript_final_ts = turn_timestamps.get("transcript_final")
                preference_recall_start_ts = turn_timestamps.get("preference_recall_start")
                preference_recall_end_ts = turn_timestamps.get("preference_recall_end")
                cancel_replace_ms = float(turn_timestamps.get("cancel_replace_ms") or 0.0)
                t_wait_transcript_final_ms = 0.0
                if isinstance(transcript_final_ts, (int, float)):
                    t_wait_transcript_final_ms = max(0.0, (float(transcript_final_ts) - float(getattr(api, "_response_start_monotonic", 0.0) or 0.0)) * 1000.0)
                t_pref_recall_ms = 0.0
                if isinstance(preference_recall_start_ts, (int, float)) and isinstance(preference_recall_end_ts, (int, float)):
                    t_pref_recall_ms = max(0.0, (float(preference_recall_end_ts) - float(preference_recall_start_ts)) * 1000.0)
                t_wait_response_done_ms = max(0.0, total_ms - t_wait_transcript_final_ms - t_pref_recall_ms - cancel_replace_ms)
                logger.debug(
                    "realtime_api_response_timing_breakdown t_wait_transcript_final_ms=%.1f t_pref_recall_ms=%.1f "
                    "t_cancel_replace_ms=%.1f t_wait_response_done_ms=%.1f total_ms=%.1f",
                    t_wait_transcript_final_ms,
                    t_pref_recall_ms,
                    cancel_replace_ms,
                    t_wait_response_done_ms,
                    total_ms,
                )
            api.response_start_time = None
            api._response_start_monotonic = None

        log_info("Assistant audio response complete.", style="bold blue")
        api.response_in_progress = False
        if api._audio_accum:
            if api.audio_player:
                api.audio_player.play_audio(bytes(api._audio_accum))
            api._audio_accum.clear()
        if api.audio_player:
            api.audio_player.close_response()
        manager = getattr(api, "_micro_ack_manager", None)
        if manager is not None:
            manager.mark_assistant_audio_ended()
        delivered_turn_id = api._current_turn_id_or_unknown()
        delivered_input_event_key = str(getattr(api, "_active_response_input_event_key", "") or "").strip()
        if delivered_input_event_key:
            api._set_response_delivery_state(
                turn_id=delivered_turn_id,
                input_event_key=delivered_input_event_key,
                state="delivered",
            )
            logger.debug(
                "response_delivery_marked run_id=%s input_event_key=%s",
                api._current_run_id() or "",
                delivered_input_event_key,
            )
        if api._pending_image_stimulus:
            if api.audio_player:
                api._pending_image_flush_after_playback = True
            else:
                await api._flush_pending_image_stimulus("audio response done")

    async def handle_response_done(self, event: dict[str, Any] | None = None) -> None:
        api = self._api
        api._mark_utterance_info_summary(response_done_seen=True)
        turn_id = api._current_turn_id_or_unknown()
        done_input_event_key = str(getattr(api, "_active_response_input_event_key", "") or "").strip()
        active_done_canonical_key = str(getattr(api, "_active_response_canonical_key", "") or "").strip()
        done_canonical_key = active_done_canonical_key or api._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=done_input_event_key,
        )
        active_response_id = str(getattr(api, "_active_response_id", "") or "").strip()
        active_response_origin = str(getattr(api, "_active_response_origin", "unknown") or "unknown").strip()
        active_input_event_key = done_input_event_key
        active_canonical_key = str(getattr(api, "_active_response_canonical_key", "") or "").strip()
        delivery_state_before_done = api._response_delivery_state(
            turn_id=turn_id,
            input_event_key=done_input_event_key,
        )
        active_response_id_before_clear = getattr(api, "_active_response_id", None)
        active_response_origin_before_clear = getattr(api, "_active_response_origin", "unknown")
        active_response_input_event_key_before_clear = getattr(api, "_active_response_input_event_key", None)
        active_response_canonical_key_before_clear = getattr(api, "_active_response_canonical_key", None)
        suppressed_turns = getattr(api, "_preference_recall_suppressed_turns", None)
        if not isinstance(suppressed_turns, set):
            suppressed_turns = set()
            api._preference_recall_suppressed_turns = suppressed_turns
        suppressed_input_event_keys = getattr(api, "_preference_recall_suppressed_input_event_keys", None)
        if not isinstance(suppressed_input_event_keys, set):
            suppressed_input_event_keys = set()
            api._preference_recall_suppressed_input_event_keys = suppressed_input_event_keys
        obligations = getattr(api, "_response_obligations", {})
        obligation_count_before_clear = len(obligations) if isinstance(obligations, dict) else 0
        obligation_key = api._response_obligation_key(turn_id=turn_id, input_event_key=done_input_event_key)
        obligation_present_before = isinstance(obligations, dict) and obligation_key in obligations
        pending_queue_len_before_clear = len(getattr(api, "_response_create_queue", deque()) or ())
        current_state_before_cleanup = getattr(api.state_manager, "state", InteractionState.IDLE)
        api._cancel_micro_ack(turn_id=turn_id, reason="response_done")
        api._response_done_serial += 1
        api.response_in_progress = False
        api._response_in_flight = False
        if done_canonical_key and bool(getattr(api, "_active_response_consumes_canonical_slot", True)):
            api._lifecycle_controller().on_response_done(done_canonical_key)
            api._log_lifecycle_event(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
                canonical_key=done_canonical_key,
                origin=active_response_origin_before_clear,
                response_id=active_response_id_before_clear,
                decision="transition_done:response_done",
            )
            api._debug_dump_canonical_key_timeline(
                canonical_key=done_canonical_key,
                trigger="response_done",
            )
            if delivery_state_before_done != "cancelled" and done_input_event_key:
                api._set_response_delivery_state(
                    turn_id=turn_id,
                    input_event_key=done_input_event_key,
                    state="done",
                )
        if active_canonical_key and api._tool_followup_state(canonical_key=active_canonical_key) in {"creating", "created"}:
            api._set_tool_followup_state(
                canonical_key=active_canonical_key,
                state="done",
                reason="response_done",
            )
        api._release_blocked_tool_followups_for_response_done(
            response_id=str(active_response_id_before_clear or ""),
        )
        if delivery_state_before_done == "cancelled":
            api._log_cancelled_deliverable_once(
                active_response_id,
                source_event="response.done.handler",
            )
        elif str(active_response_origin_before_clear or "").strip().lower() == "micro_ack":
            logger.info(
                "deliverable_selected response_id=%s selected=false reason=micro_ack_non_deliverable",
                active_response_id or "unknown",
            )
        else:
            logger.info(
                "deliverable_selected response_id=%s selected=true reason=normal",
                active_response_id or "unknown",
            )
        suppressed_turn_id = api._current_turn_id_or_unknown()
        suppressed_turn_present_before = suppressed_turn_id in suppressed_turns
        logger.debug(
            "[RESPTRACE] response_done_cleanup_before run_id=%s active_response_id=%s "
            "active_response_origin=%s active_response_input_event_key=%s active_response_canonical_key=%s "
            "suppressed_turns_count=%s suppressed_keys_count=%s suppressed_turn_present_before=%s "
            "obligation_count_before=%s pending_queue_len=%s response_done_serial=%s state=%s",
            api._current_run_id() or "",
            active_response_id_before_clear,
            active_response_origin_before_clear,
            active_response_input_event_key_before_clear,
            active_response_canonical_key_before_clear,
            len(suppressed_turns),
            len(suppressed_input_event_keys),
            suppressed_turn_present_before,
            obligation_count_before_clear,
            pending_queue_len_before_clear,
            getattr(api, "_response_done_serial", 0),
            current_state_before_cleanup,
        )
        api._preference_recall_suppressed_turns.discard(suppressed_turn_id)
        active_input_event_key = str(getattr(api, "_active_server_auto_input_event_key", "") or "").strip()
        active_input_event_key_present_before = bool(
            active_input_event_key and active_input_event_key in suppressed_input_event_keys
        )
        if active_input_event_key:
            api._preference_recall_suppressed_input_event_keys.discard(active_input_event_key)
        was_confirmation_guarded = api._active_response_confirmation_guarded
        api._active_response_id = None
        api._active_response_confirmation_guarded = False
        api._active_response_preference_guarded = False
        api._active_response_origin = "unknown"
        api._active_response_input_event_key = None
        api._active_response_canonical_key = None
        api._active_server_auto_input_event_key = None
        obligations_after_cleanup = getattr(api, "_response_obligations", {})
        obligation_count_after_clear = len(obligations_after_cleanup) if isinstance(obligations_after_cleanup, dict) else 0
        obligation_present_after = (
            isinstance(obligations_after_cleanup, dict) and obligation_key in obligations_after_cleanup
        )
        resolved_input_event_key = done_input_event_key or "unknown"
        resolved_canonical_key = api._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=done_input_event_key,
        )
        if active_response_id:
            api._record_response_trace_context(
                active_response_id,
                turn_id=turn_id,
                input_event_key=resolved_input_event_key,
                canonical_key=resolved_canonical_key,
                origin=active_response_origin,
            )
        api._emit_response_lifecycle_trace(
            event_type="response.done",
            response_id=active_response_id,
            turn_id=turn_id,
            input_event_key=resolved_input_event_key,
            canonical_key=resolved_canonical_key,
            origin=active_response_origin,
            active_input_event_key=active_input_event_key,
            active_canonical_key=active_canonical_key,
            payload_summary=f"delivery_state_before_done={delivery_state_before_done}",
        )
        logger.debug(
            "[RESPTRACE] response_done_cleanup_after run_id=%s removed_suppressed_turn=%s "
            "removed_suppressed_input_event_key=%s suppressed_turns_count=%s suppressed_keys_count=%s "
            "obligation_count_before=%s obligation_count_after=%s pending_queue_len=%s response_done_serial=%s state=%s "
            "turn_id=%s input_event_key=%s canonical_key=%s obligation_key=%s "
            "obligation_present_before=%s obligation_present_after=%s total_obligations=%s "
            "active_response_input_event_key=%s active_server_auto_input_event_key=%s",
            api._current_run_id() or "",
            suppressed_turn_present_before and suppressed_turn_id not in api._preference_recall_suppressed_turns,
            active_input_event_key_present_before
            and active_input_event_key not in api._preference_recall_suppressed_input_event_keys,
            len(suppressed_turns),
            len(suppressed_input_event_keys),
            obligation_count_before_clear,
            obligation_count_after_clear,
            len(getattr(api, "_response_create_queue", deque()) or ()),
            getattr(api, "_response_done_serial", 0),
            getattr(api.state_manager, "state", InteractionState.IDLE),
            turn_id,
            resolved_input_event_key,
            resolved_canonical_key,
            obligation_key,
            obligation_present_before,
            obligation_present_after,
            obligation_count_after_clear,
            str(getattr(api, "_active_response_input_event_key", "") or "").strip() or "none",
            str(getattr(api, "_active_server_auto_input_event_key", "") or "").strip() or "none",
        )
        current_state = getattr(api.state_manager, "state", InteractionState.IDLE)
        if current_state != InteractionState.LISTENING:
            api.state_manager.update_state(InteractionState.IDLE, "response done")
        else:
            logger.debug(
                "Skipping IDLE transition for response.done while still listening; deferring until speech stop."
            )
        logger.info("Received response.done event.")
        api._emit_utterance_info_summary(anchor="response.done")
        if api._is_empty_response_done(canonical_key=done_canonical_key):
            api._record_silent_turn_incident(
                turn_id=turn_id,
                canonical_key=done_canonical_key,
                origin=active_response_origin_before_clear,
                response_id=active_response_id_before_clear,
            )
        await api._maybe_schedule_empty_response_retry(
            websocket=api.websocket,
            turn_id=turn_id,
            canonical_key=done_canonical_key,
            input_event_key=done_input_event_key,
            origin=active_response_origin_before_clear,
            delivery_state_before_done=delivery_state_before_done,
        )
        if event:
            api._last_response_metadata = {
                "event_type": event.get("type"),
                "response": event.get("response"),
                "rate_limits": api.rate_limits,
            }
        api._emit_preference_recall_skip_trace_if_needed(turn_id=api._current_response_turn_id)
        api._log_turn_conversation_efficiency(
            turn_id=turn_id,
            canonical_key=resolved_canonical_key,
            close_reason="response_done",
        )
        transition = api._build_confirmation_transition_decision(
            reason="response_done",
            include_reminder_gate=True,
            was_confirmation_guarded=was_confirmation_guarded,
        )
        token_active, awaiting_phase, _, phase_is_awaiting_confirmation = api._confirmation_hold_components()
        if not transition.allow_response_transition:
            logger.info(
                "Confirmation state is holding phase progression; skipping REFLECT transition "
                "(phase=%s token_active=%s awaiting_phase=%s).",
                api.orchestration_state.phase,
                token_active,
                awaiting_phase,
            )
            if phase_is_awaiting_confirmation and token_active:
                logger.info("Staying in AWAITING_CONFIRMATION until user accepts/rejects.")
            elif transition.close_reason == "confirmation_follow_up_completed":
                api._awaiting_confirmation_completion = False
                api.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation follow-up completed",
                )
        else:
            api.orchestration_state.transition(
                OrchestrationPhase.REFLECT,
                reason="response done",
            )
            api._enqueue_response_done_reflection("response done")
            api.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="response done reflection",
            )
        if was_confirmation_guarded:
            if token_active:
                api._mark_confirmation_activity(reason="guarded_response_done")
            if (
                transition.emit_reminder
                and api.websocket is not None
                and api._should_send_response_done_fallback_reminder()
                and api._is_guarded_server_auto_reminder_allowed(reason="response_done_fallback")
            ):
                await api._maybe_emit_confirmation_reminder(api.websocket, reason="response_done_fallback")
            if transition.recover_mic:
                api._recover_confirmation_guard_microphone("response_done")
        api._response_create_queue_drain_source = "response_done"
        await api._drain_response_create_queue()
        if api._pending_image_stimulus and not api._pending_image_flush_after_playback:
            await api._flush_pending_image_stimulus("response done")

    async def handle_response_completed(self, event: dict[str, Any] | None = None) -> None:
        api = self._api
        response_id = api._response_id_from_event(event) or str(getattr(api, "_active_response_id", "") or "").strip()
        turn_id = api._current_turn_id_or_unknown()
        done_input_event_key = str(getattr(api, "_active_response_input_event_key", "") or "").strip()
        api._cancel_micro_ack(turn_id=turn_id, reason="response_completed")
        api.response_in_progress = False
        api._response_in_flight = False
        if done_input_event_key and bool(getattr(api, "_active_response_consumes_canonical_slot", True)):
            done_canonical_key = api._canonical_utterance_key(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
            )
            api._lifecycle_controller().on_response_done(done_canonical_key)
            api._log_lifecycle_event(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
                canonical_key=done_canonical_key,
                origin=getattr(api, "_active_response_origin", "unknown"),
                response_id=getattr(api, "_active_response_id", None),
                decision="transition_done:response_completed",
            )
            api._debug_dump_canonical_key_timeline(
                canonical_key=done_canonical_key,
                trigger="response_completed",
            )
            existing_delivery_state = api._response_delivery_state(
                turn_id=turn_id,
                input_event_key=done_input_event_key,
            )
            if existing_delivery_state != "cancelled":
                api._set_response_delivery_state(
                    turn_id=turn_id,
                    input_event_key=done_input_event_key,
                    state="done",
                )
        api._preference_recall_suppressed_turns.discard(api._current_turn_id_or_unknown())
        active_input_event_key = str(getattr(api, "_active_server_auto_input_event_key", "") or "").strip()
        if active_input_event_key:
            api._preference_recall_suppressed_input_event_keys.discard(active_input_event_key)
        api._active_response_id = None
        api._active_response_confirmation_guarded = False
        api._active_response_preference_guarded = False
        api._active_response_origin = "unknown"
        api._active_server_auto_input_event_key = None
        current_state = getattr(api.state_manager, "state", InteractionState.IDLE)
        if current_state != InteractionState.LISTENING:
            api.state_manager.update_state(InteractionState.IDLE, "response completed")
        else:
            logger.debug(
                "Skipping IDLE transition for response.completed while still listening; deferring until speech stop."
            )
        logger.info("Received response.completed event.")
        api._clear_cancelled_response_tracking(response_id)
        if event:
            api._last_response_metadata = {
                "event_type": event.get("type"),
                "response": event.get("response"),
                "rate_limits": api.rate_limits,
            }
        api._emit_preference_recall_skip_trace_if_needed(turn_id=api._current_response_turn_id)
        api._log_turn_conversation_efficiency(
            turn_id=turn_id,
            canonical_key=api._canonical_utterance_key(turn_id=turn_id, input_event_key=done_input_event_key),
            close_reason="response_completed",
        )
        transition = api._build_confirmation_transition_decision(reason="response_completed")
        token_active, awaiting_phase, _, phase_is_awaiting_confirmation = api._confirmation_hold_components()
        if not transition.allow_response_transition:
            logger.info(
                "Confirmation state is holding phase progression; skipping REFLECT transition "
                "(phase=%s token_active=%s awaiting_phase=%s).",
                api.orchestration_state.phase,
                token_active,
                awaiting_phase,
            )
            if phase_is_awaiting_confirmation and token_active:
                logger.info("Staying in AWAITING_CONFIRMATION until user accepts/rejects.")
            elif transition.close_reason == "confirmation_follow_up_completed":
                api._awaiting_confirmation_completion = False
                api.orchestration_state.transition(
                    OrchestrationPhase.IDLE,
                    reason="confirmation follow-up completed",
                )
        else:
            api.orchestration_state.transition(
                OrchestrationPhase.REFLECT,
                reason="response completed",
            )
            api._maybe_enqueue_reflection("response completed")
            api.orchestration_state.transition(
                OrchestrationPhase.IDLE,
                reason="reflection enqueued",
            )
        api._response_create_queue_drain_source = "active_cleared"
        await api._drain_response_create_queue(source_trigger="active_cleared")
        if api._pending_image_stimulus and not api._pending_image_flush_after_playback:
            await api._flush_pending_image_stimulus("response completed")
