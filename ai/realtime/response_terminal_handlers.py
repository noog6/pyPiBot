"""Handlers for terminal realtime response events."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import os
import time
from typing import TYPE_CHECKING, Any

from core.logging import log_info, logger
from ai.decision_arbitration_adapter import (
    build_semantic_owner_observation,
    build_terminal_selection_observation,
    merge_arbitration_observations_for_turn,
    summarize_turn_arbitration_diagnostics,
    summarize_turn_arbitration_for_review,
    summarize_turn_arbitration_trace,
)
from interaction import InteractionState
from ai.orchestration import OrchestrationPhase
from ai.realtime.asr_trust import topic_mismatch_detected
from ai.contract_breach import ContractBreachSnapshot, detect_contract_breach
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

    def _contract_breach_step_context(self, *, turn_id: str) -> tuple[str, str]:
        """Return continuity-derived step-position context for breach log enrichment only."""
        api = self._api
        run_id = str(api._current_run_id() or "").strip()
        normalized_turn_id = str(turn_id or "").strip()
        if not run_id or not normalized_turn_id:
            return ("none", "none")
        try:
            brief = api.get_continuity_brief(
                run_id=run_id,
                turn_id=normalized_turn_id,
                reason="contract_breach_detected",
            )
        except Exception:
            return ("none", "none")
        compound_state = getattr(brief, "compound_request", None)
        if compound_state is None:
            return ("none", "none")
        active_step_index = getattr(compound_state, "active_step_index", None)
        active_step_index_value = str(active_step_index) if isinstance(active_step_index, int) else "none"
        next_pending_step_id = str(getattr(compound_state, "next_pending_step_id", "") or "").strip() or "none"
        return (active_step_index_value, next_pending_step_id)

    async def handle_transcribe_response_done(self, event: dict[str, Any] | None = None) -> None:
        api = self._api
        if (
            api._active_response_confirmation_guarded
            and (api._has_active_confirmation_token() or api._pending_action is not None)
            and api._is_awaiting_confirmation_phase()
        ):
            api._clear_assistant_reply_buffers()
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
            api._set_active_response_state(confirmation_guarded=False)
            api.response_in_progress = False
            api._recover_confirmation_guard_microphone("transcribe_response_done")
            api._maybe_enqueue_reflection("response transcript done")
            if api._pending_image_stimulus and not api._pending_image_flush_after_playback:
                await api._flush_pending_image_stimulus("response transcript done")
            logger.info("Finished handle_transcribe_response_done()")
            return

        event_response_id = api._response_id_from_event(event)
        response_id = event_response_id or str(getattr(api, "_assistant_reply_response_id", "") or "").strip()
        active_response_id = str(getattr(api, "_active_response_id", "") or "").strip()
        if event_response_id and active_response_id and event_response_id != active_response_id:
            logger.debug(
                "transcript_done_ignored reason=inactive_response response_id=%s active_response_id=%s",
                event_response_id,
                active_response_id,
            )
            return

        reply_text = api._assistant_reply_text_for_response(response_id) if response_id else str(api.assistant_reply or "")

        if reply_text:
            active_input_event_key = str(getattr(api, "_active_response_input_event_key", "") or "").strip()
            snapshot = getattr(api, "_utterance_trust_snapshot_by_input_event_key", {}).get(active_input_event_key, {})
            transcript_text = str(snapshot.get("transcript_text") or "")
            if (
                transcript_text
                and topic_mismatch_detected(transcript_text, reply_text)
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
                    api._clear_assistant_reply_buffers(response_id=response_id)
                    api.response_in_progress = False
                    return
            normalized_reply = api._normalize_memory_recall_answer(reply_text)
            log_info(f"Assistant Response: {normalized_reply}", style="bold blue")
            if response_id:
                per_response = getattr(api, "_assistant_reply_by_response_id", None)
                if isinstance(per_response, dict):
                    per_response[response_id] = normalized_reply
            api.assistant_reply = normalized_reply
            api._record_terminal_response_text(
                response_id=response_id,
                text=normalized_reply,
            )
            api._clear_assistant_reply_buffers(response_id=response_id or None)

        api.response_in_progress = False
        api._maybe_enqueue_reflection("response transcript done")
        if api._pending_image_stimulus and not api._pending_image_flush_after_playback:
            await api._flush_pending_image_stimulus("response transcript done")
        logger.info("Finished handle_transcribe_response_done()")

    async def handle_audio_response_done(self) -> None:
        api = self._api
        response_id = str(getattr(api, "_active_response_id", "") or "").strip()
        trace_context = api._response_trace_by_id().get(response_id, {}) if response_id else {}
        if str(trace_context.get("tool_followup") or "").strip().lower() in {"true", "1", "yes"}:
            api._mark_tool_followup_timing(
                turn_id=str(trace_context.get("turn_id") or api._current_turn_id_or_unknown()).strip() or "turn-unknown",
                marker="followup_output_audio_done",
                call_id=str(trace_context.get("tool_call_id") or "").strip() or None,
                canonical_key=str(trace_context.get("canonical_key") or "").strip() or None,
                response_id=response_id or None,
                is_tool_followup=True,
                released=str(trace_context.get("tool_followup_release") or "").strip().lower() in {"true", "1", "yes"},
            )
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
        response_id = api._response_id_from_event(event)
        trace_context = api._response_trace_by_id().get(response_id, {}) if response_id else {}
        stale_context = api._stale_response_context(response_id) if response_id else {}
        if str(trace_context.get("tool_followup") or "").strip().lower() in {"true", "1", "yes"}:
            api._mark_tool_followup_timing(
                turn_id=str(trace_context.get("turn_id") or api._current_turn_id_or_unknown()).strip() or "turn-unknown",
                marker="followup_response_done",
                call_id=str(trace_context.get("tool_call_id") or "").strip() or None,
                canonical_key=str(trace_context.get("canonical_key") or "").strip() or None,
                response_id=response_id or None,
                is_tool_followup=True,
                released=str(trace_context.get("tool_followup_release") or "").strip().lower() in {"true", "1", "yes"},
            )
            api._maybe_emit_tool_followup_timing_summary(
                turn_id=str(trace_context.get("turn_id") or api._current_turn_id_or_unknown()).strip() or "turn-unknown",
            )
        active_turn_id = api._current_turn_id_or_unknown()
        turn_id = str(
            trace_context.get("turn_id")
            or stale_context.get("turn_id")
            or active_turn_id
        ).strip() or "turn-unknown"
        active_input_event_key_for_turn = str(api._active_input_event_key_for_turn(turn_id) or "").strip()
        done_input_event_key = str(
            trace_context.get("input_event_key")
            or stale_context.get("input_event_key")
            or getattr(api, "_active_response_input_event_key", "")
            or active_input_event_key_for_turn
            or getattr(api, "_current_input_event_key", "")
            or ""
        ).strip()
        active_done_canonical_key = str(
            stale_context.get("canonical_key")
            or trace_context.get("canonical_key")
            or getattr(api, "_active_response_canonical_key", "")
            or ""
        ).strip()
        done_canonical_key = active_done_canonical_key or api._canonical_utterance_key(
            turn_id=turn_id,
            input_event_key=done_input_event_key,
        )
        resolved_origin = str(
            stale_context.get("origin")
            or trace_context.get("origin")
            or getattr(api, "_active_response_origin", "unknown")
            or "unknown"
        ).strip() or "unknown"
        if not api._should_admit_response_terminal_event(
            event_type=str((event or {}).get("type") or "response.done"),
            response_id=response_id or None,
            turn_id=turn_id,
            input_event_key=done_input_event_key or None,
            canonical_key=done_canonical_key or None,
            origin=resolved_origin,
        ):
            return
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
        active_response_was_provisional = api._is_provisional_response(response_id=active_response_id_before_clear)
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
        transcript_linked_input_event_key = str(api._active_input_event_key_for_turn(turn_id) or "").strip()
        transcript_final_linked = bool(transcript_linked_input_event_key and transcript_linked_input_event_key.startswith("item_"))
        selection_decision = api._response_done_deliverable_arbitration(
            turn_id=turn_id,
            origin=str(active_response_origin_before_clear or ""),
            delivery_state_before_done=delivery_state_before_done,
            active_response_was_provisional=active_response_was_provisional,
            done_canonical_key=done_canonical_key,
            transcript_final_seen=transcript_final_linked,
            input_event_key=done_input_event_key,
            response_id=active_response_id_before_clear,
        )
        selected = selection_decision.selected
        selection_reason = selection_decision.reason_code
        required_deliverable_followthrough = api._response_done_marks_required_deliverable_followthrough(
            origin=str(active_response_origin_before_clear or ""),
            turn_id=turn_id,
            response_id=active_response_id_before_clear,
            trace_context=trace_context,
            stale_context=stale_context,
        )
        selected_response_has_terminal_text_evidence = api._selected_response_has_terminal_text_evidence(
            response_id=active_response_id_before_clear
        )
        required_deliverable_missing_substance = (
            required_deliverable_followthrough
            and selected
            and selection_reason == "normal"
            and not selected_response_has_terminal_text_evidence
        )
        required_deliverable_missing_tool_execution = (
            required_deliverable_followthrough
            and selected
            and selection_reason == "normal"
            and api._required_deliverable_tool_execution_missing(
                turn_id=turn_id,
                required_deliverable_followthrough=required_deliverable_followthrough,
                response_id=active_response_id_before_clear,
                trace_context=trace_context,
                stale_context=stale_context,
            )
        )
        if required_deliverable_missing_substance:
            selected = False
            selection_reason = "required_deliverable_missing_substantive_content"
            logger.info(
                "required_deliverable_completion_rejected run_id=%s turn_id=%s response_id=%s canonical_key=%s action=defer_settlement reason=missing_substantive_content",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
                done_canonical_key,
            )
        elif required_deliverable_missing_tool_execution:
            selected = False
            selection_reason = "required_deliverable_missing_tool_execution"
            logger.info(
                "required_deliverable_completion_rejected run_id=%s turn_id=%s response_id=%s canonical_key=%s action=defer_settlement reason=missing_tool_execution",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
                done_canonical_key,
            )
        followthrough_dispatch_source = str(
            trace_context.get("followthrough_dispatch_source")
            or stale_context.get("followthrough_dispatch_source")
            or ""
        ).strip().lower()
        local_runtime_followthrough = str(
            trace_context.get("local_runtime_followthrough")
            or stale_context.get("local_runtime_followthrough")
            or ""
        ).strip().lower() in {"true", "1", "yes"}
        required_deliverable_materialization_redrive_eligible = (
            required_deliverable_followthrough
            and str(active_response_origin_before_clear or "").strip().lower() == "tool_output"
            and (
                followthrough_dispatch_source == "deterministic_followthrough_motion_gate"
                or local_runtime_followthrough
            )
        )
        if (
            required_deliverable_materialization_redrive_eligible
            and (required_deliverable_missing_substance or required_deliverable_missing_tool_execution)
        ):
            retry_allowed, retry_attempt, retry_max_attempts = (
                api._consume_required_deliverable_materialization_retry_budget(turn_id=turn_id)
            )
            if retry_allowed:
                api._release_required_deliverable_followthrough_dispatch_lock(
                    turn_id=turn_id,
                    reason=selection_reason,
                )
                redispatched = await api._dispatch_required_deliverable_followthrough_response_create(
                    websocket=api.websocket,
                    turn_id=turn_id,
                )
                logger.info(
                    "required_deliverable_followthrough_materialization_redrive run_id=%s turn_id=%s response_id=%s reason=%s dispatched=%s retry_attempt=%s retry_max_attempts=%s",
                    api._current_run_id() or "",
                    turn_id,
                    str(active_response_id_before_clear or "none"),
                    selection_reason,
                    str(bool(redispatched)).lower(),
                    retry_attempt,
                    retry_max_attempts,
                )
            else:
                logger.info(
                    "required_deliverable_followthrough_materialization_redrive_skipped run_id=%s turn_id=%s response_id=%s reason=%s retry_attempt=%s retry_max_attempts=%s",
                    api._current_run_id() or "",
                    turn_id,
                    str(active_response_id_before_clear or "none"),
                    selection_reason,
                    retry_attempt,
                    retry_max_attempts,
                )
        terminal_selection_observation = build_terminal_selection_observation(
            run_id=api._current_run_id() or "",
            turn_id=turn_id,
            input_event_key=done_input_event_key,
            canonical_key=done_canonical_key,
            origin=str(active_response_origin_before_clear or ""),
            selected=selected,
            selection_reason=selection_reason,
            selected_candidate_id=selection_decision.selected_candidate_id,
            transcript_final_seen=transcript_final_linked,
            active_response_was_provisional=active_response_was_provisional,
        )
        logger.debug(
            "decision_adapter_terminal_selection_observation payload=%s",
            terminal_selection_observation.to_log_payload(),
        )
        exact_phrase_close_deferred = False
        if selection_reason == "exact_phrase_obligation_open":
            contract_before = api._active_turn_contract(turn_id=turn_id)
            repair_already_scheduled = bool(
                isinstance(contract_before, dict) and contract_before.get("exact_phrase_repair_scheduled")
            )
            repair_scheduled_now = False
            if not repair_already_scheduled:
                repair_scheduled_now = await api._schedule_turn_contract_exact_phrase_repair_response(
                    turn_id=turn_id,
                    input_event_key=done_input_event_key,
                    websocket=api.websocket,
                )
            contract_after = api._active_turn_contract(turn_id=turn_id)
            repair_pending = bool(
                repair_already_scheduled
                or repair_scheduled_now
                or (
                    isinstance(contract_after, dict)
                    and contract_after.get("exact_phrase_repair_scheduled")
                )
            )
            exact_phrase_close_deferred = repair_pending
            logger.info(
                "exact_phrase_close_guard run_id=%s turn_id=%s response_id=%s already_scheduled=%s scheduled_now=%s defer_close=%s",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
                str(repair_already_scheduled).lower(),
                str(repair_scheduled_now).lower(),
                str(exact_phrase_close_deferred).lower(),
            )
            if not exact_phrase_close_deferred:
                logger.warning(
                    "exact_phrase_close_guard_fallback run_id=%s turn_id=%s response_id=%s action=allow reason=repair_not_scheduled",
                    api._current_run_id() or "",
                    turn_id,
                    str(active_response_id_before_clear or "none"),
                )

        resolved_response_id = str(active_response_id_before_clear or "")
        semantic_owner_decision = api._semantic_owner_decision_for_response(
            turn_id=turn_id,
            input_event_key=done_input_event_key,
            origin=str(active_response_origin_before_clear or ""),
            response_id=resolved_response_id,
            done_canonical_key=done_canonical_key,
            selected=selected,
            selection_reason=selection_reason,
        )
        semantic_owner_canonical_key = semantic_owner_decision.semantic_owner_canonical_key
        semantic_owner_observation = build_semantic_owner_observation(
            run_id=api._current_run_id() or "",
            turn_id=turn_id,
            input_event_key=done_input_event_key,
            execution_canonical_key=done_canonical_key,
            semantic_owner_canonical_key=semantic_owner_canonical_key,
            origin=str(active_response_origin_before_clear or ""),
            selected=selected,
            selection_reason=selection_reason,
            parent_turn_id=semantic_owner_decision.parent_turn_id,
            parent_input_event_key=semantic_owner_decision.parent_input_event_key,
            selected_candidate_id=semantic_owner_decision.selected_candidate_id,
            native_reason_code=semantic_owner_decision.reason_code,
        )
        logger.debug(
            "decision_adapter_semantic_owner_observation payload=%s",
            semantic_owner_observation.to_log_payload(),
        )
        trace_store = getattr(api, "_turn_arbitration_trace_by_key", None)
        if not isinstance(trace_store, dict):
            trace_store = {}
            setattr(api, "_turn_arbitration_trace_by_key", trace_store)
        trace_key = (api._current_run_id() or "", turn_id)
        trace = merge_arbitration_observations_for_turn(
            existing_trace=trace_store.get(trace_key),
            terminal_selection_observation=terminal_selection_observation,
            semantic_owner_observation=semantic_owner_observation,
            semantic_owner_canonical_key=semantic_owner_canonical_key,
        )
        trace_store[trace_key] = trace
        while len(trace_store) > 128:
            trace_store.pop(next(iter(trace_store)))
        logger.debug(
            "decision_adapter_turn_trace payload=%s",
            summarize_turn_arbitration_trace(trace),
        )
        if trace.diagnostics is not None:
            logger.debug(
                "decision_adapter_turn_diagnostics payload=%s",
                summarize_turn_arbitration_diagnostics(trace.diagnostics),
            )
        logger.debug(
            "decision_adapter_turn_review_summary payload=%s",
            summarize_turn_arbitration_for_review(trace),
        )
        api._emit_turn_review_summary_info_if_material(trace)
        api._apply_terminal_deliverable_selection(
            canonical_key=done_canonical_key,
            semantic_owner_canonical_key=semantic_owner_canonical_key,
            response_id=resolved_response_id,
            turn_id=turn_id,
            input_event_key=done_input_event_key,
            selected=selected,
            selection_reason=selection_reason,
        )
        api._reconcile_semantic_substantive_owner(
            turn_id=turn_id,
            execution_canonical_key=done_canonical_key,
            semantic_owner_canonical_key=semantic_owner_canonical_key,
            response_id=resolved_response_id,
        )
        api._reconcile_terminal_substantive_response(
            turn_id=turn_id,
            canonical_key=semantic_owner_canonical_key,
            response_id=resolved_response_id,
            selected=selected,
            selection_reason=selection_reason,
        )
        api._release_blocked_tool_followups_for_response_done(
            response_id=resolved_response_id,
        )
        pending_tool_followup_after_release = api._turn_has_pending_tool_followup(turn_id=turn_id)
        pending_tool_followup_any_after_release = api._turn_has_pending_tool_followup(
            turn_id=turn_id,
            include_status_only=True,
        )
        transcript_linked_input_event_key = str(api._active_input_event_key_for_turn(turn_id) or "").strip()
        transcript_final_linked = bool(transcript_linked_input_event_key and transcript_linked_input_event_key.startswith("item_"))
        obligations_map = getattr(api, "_response_obligations", {})
        obligation_open = bool(
            isinstance(obligations_map, dict)
            and any(str(key).startswith(f"{turn_id}:") for key in obligations_map)
        )
        logger.info(
            "provisional_completion_eval run_id=%s turn_id=%s response_id=%s synthetic_key=%s canonical_key=%s transcript_linked=%s action=%s reason=%s",
            api._current_run_id() or "",
            turn_id,
            str(active_response_id_before_clear or "none"),
            str(done_input_event_key or "none"),
            done_canonical_key,
            str(transcript_final_linked).lower(),
            "allow",
            "response_done_received",
        )
        close_action = "allow"
        close_reason = "response_done_received"
        provisional_server_auto_close_deferred = False
        if selection_reason == "exact_phrase_obligation_open":
            if exact_phrase_close_deferred:
                close_action = "defer"
                close_reason = "exact_phrase_obligation_open"
            else:
                close_reason = "exact_phrase_repair_not_scheduled"
        if (
            close_action == "allow"
            and active_response_was_provisional
            and str(active_response_origin_before_clear or "").strip().lower() == "server_auto"
            and not transcript_final_linked
        ):
            close_action = "defer"
            close_reason = "provisional_server_auto_awaiting_transcript_final"
            provisional_server_auto_close_deferred = True
        logger.info(
            "turn_terminal_close_eval run_id=%s turn_id=%s response_id=%s origin=%s transcript_final_seen=%s obligation_open=%s action=%s reason=%s",
            api._current_run_id() or "",
            turn_id,
            str(active_response_id_before_clear or "none"),
            str(active_response_origin_before_clear or "unknown"),
            str(transcript_final_linked).lower(),
            str(obligation_open).lower(),
            close_action,
            close_reason,
        )
        terminal_assistant_text = str(api._terminal_response_text(active_response_id_before_clear) or "").strip()
        if not terminal_assistant_text:
            terminal_assistant_text = str(api._assistant_reply_text_for_response(active_response_id_before_clear) or "").strip()
        memory_usage_store = api._memory_usage_audit_store()
        turn_memory_usage = (
            memory_usage_store.get(api._memory_usage_audit_turn_key(turn_id=turn_id, input_event_key=done_input_event_key or None), {})
            if isinstance(memory_usage_store, dict)
            else {}
        )
        injection_types = turn_memory_usage.get("injection_types") if isinstance(turn_memory_usage, dict) else []
        if not isinstance(injection_types, list):
            injection_types = []
        logger.info(
            "terminal_answer_context_audit run_id=%s turn_id=%s input_event_key=%s response_id=%s canonical_key=%s origin=%s selected=%s selection_reason=%s close_action=%s close_reason=%s pref_context_injected=%s injection_types=%s terminal_text_present=%s",
            api._current_run_id() or "",
            turn_id,
            done_input_event_key or "unknown",
            str(resolved_response_id or active_response_id_before_clear or "").strip() or "none",
            semantic_owner_canonical_key or done_canonical_key,
            str(active_response_origin_before_clear or "unknown"),
            str(selected).lower(),
            selection_reason,
            close_action,
            close_reason,
            str("preference_context" in injection_types).lower(),
            ",".join(str(value) for value in injection_types) if injection_types else "none",
            str(bool(terminal_assistant_text)).lower(),
        )
        api._finalize_memory_usage_audit(
            turn_id=turn_id,
            input_event_key=done_input_event_key or None,
            selected_response_id=resolved_response_id or active_response_id_before_clear,
            selected_canonical_key=semantic_owner_canonical_key or done_canonical_key,
            close_reason=close_reason,
            final_assistant_text=terminal_assistant_text or None,
        )
        if selection_reason == "cancelled":
            api._log_cancelled_deliverable_once(
                active_response_id,
                source_event="response.done.handler",
            )
        else:
            logger.info(
                "deliverable_selected response_id=%s selected=%s reason=%s",
                active_response_id or "unknown",
                str(selected).lower(),
                selection_reason,
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
        api._clear_active_response_state()
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
        api._log_lifecycle_coherence(
            stage="response_done",
            turn_id=turn_id,
            response_id=resolved_response_id or active_response_id,
            canonical_key=resolved_canonical_key,
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
        if str(active_response_origin_before_clear or "").strip().lower() == "prompt":
            startup_terminal_state = "completed"
            startup_terminal_reason = "response_done"
            if str(delivery_state_before_done or "").strip().lower() == "cancelled":
                startup_terminal_state = "skipped"
                startup_terminal_reason = "cancelled_before_done"
            api._emit_startup_prompt_terminal(
                terminal_state=startup_terminal_state,
                reason=startup_terminal_reason,
                turn_id=turn_id,
                input_event_key=done_input_event_key,
                canonical_key=done_canonical_key,
            )
        api._emit_utterance_info_summary(anchor="response.done")
        api._prune_curiosity_surface_candidates(completed_turn_id=turn_id)
        response_state = api._canonical_response_state(done_canonical_key)
        audio_delta_seen = bool(getattr(response_state, "audio_started", False))
        if not audio_delta_seen:
            try:
                audio_delta_seen = bool(api._canonical_first_audio_started(done_canonical_key))
            except AttributeError:
                audio_delta_seen = False
        deliverable_observed = bool(getattr(response_state, "deliverable_observed", False))
        assistant_reply_present = bool(str(api._assistant_reply_text_for_response(active_response_id_before_clear) or "").strip())
        terminal_response_text_present = bool(str(api._terminal_response_text(active_response_id_before_clear) or "").strip())
        is_empty_done = api._is_empty_response_done(canonical_key=done_canonical_key)
        logger.debug(
            "response_done_empty_evidence run_id=%s turn_id=%s response_id=%s canonical_key=%s assistant_reply_present=%s terminal_response_text_present=%s audio_delta_seen=%s deliverable_observed=%s final_is_empty_done=%s",
            api._current_run_id() or "",
            turn_id,
            str(active_response_id_before_clear or "none"),
            done_canonical_key,
            str(assistant_reply_present).lower(),
            str(terminal_response_text_present).lower(),
            str(audio_delta_seen).lower(),
            str(deliverable_observed).lower(),
            str(is_empty_done).lower(),
        )
        suppress_empty_tool_followup_silent_incident = (
            is_empty_done
            and str(active_response_origin_before_clear or "").strip().lower() == "tool_output"
            and pending_tool_followup_after_release
        )
        followthrough_chain_bridge = (
            is_empty_done
            and str(active_response_origin_before_clear or "").strip().lower() == "tool_output"
            and selection_reason == "tool_followup_precedence"
        )
        interrupted_tool_output_candidate = (
            is_empty_done
            and api._is_interrupted_tool_output_candidate(
                response_id=active_response_id_before_clear,
                canonical_key=done_canonical_key,
            )
        )
        if suppress_empty_tool_followup_silent_incident:
            logger.info(
                "tool_followup_empty_bridge run_id=%s turn_id=%s response_id=%s canonical_key=%s action=suppress_silent_incident reason=pending_followup_after_release",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
                done_canonical_key,
            )
        elif followthrough_chain_bridge:
            logger.info(
                "tool_followup_empty_bridge run_id=%s turn_id=%s response_id=%s canonical_key=%s action=suppress_silent_incident reason=followthrough_chain_remaining",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
                done_canonical_key,
            )
        elif interrupted_tool_output_candidate:
            logger.info(
                "tool_output_interruption_recovery run_id=%s turn_id=%s response_id=%s canonical_key=%s action=defer_candidate reason=interrupted_pre_evidence",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
                done_canonical_key,
            )
        elif is_empty_done and not active_response_was_provisional:
            api._record_silent_turn_incident(
                turn_id=turn_id,
                canonical_key=done_canonical_key,
                origin=active_response_origin_before_clear,
                response_id=active_response_id_before_clear,
            )
        elif is_empty_done and active_response_was_provisional:
            api._mark_provisional_response_completed_empty(response_id=active_response_id_before_clear)
            logger.info(
                "provisional_response_completed_empty run_id=%s turn_id=%s canonical_key=%s origin=%s response_id=%s",
                api._current_run_id() or "",
                turn_id,
                done_canonical_key,
                str(active_response_origin_before_clear or "").strip() or "unknown",
                str(active_response_id_before_clear or "").strip() or "none",
            )
        continuity_close_allowed = not (
            provisional_server_auto_close_deferred or exact_phrase_close_deferred or is_empty_done
        )
        if interrupted_tool_output_candidate:
            continuity_close_allowed = False
        if required_deliverable_missing_substance:
            continuity_close_allowed = False
        if required_deliverable_missing_tool_execution:
            continuity_close_allowed = False
        continuity_origin = str(active_response_origin_before_clear or "").strip().lower()
        semantic_parent_turn_id = str(getattr(semantic_owner_decision, "parent_turn_id", "") or "").strip()
        continuity_candidate_turn_id = turn_id
        continuity_candidate_rebind = False
        if (
            continuity_origin == "tool_output"
            and selected
            and semantic_parent_turn_id
            and semantic_parent_turn_id != turn_id
            and semantic_owner_canonical_key
            and semantic_owner_canonical_key != done_canonical_key
        ):
            continuity_candidate_turn_id = semantic_parent_turn_id
            continuity_candidate_rebind = True
        followthrough_chain_remaining_for_close = api._response_done_followthrough_chain_remaining(
            turn_id=continuity_candidate_turn_id,
            origin=active_response_origin_before_clear,
            response_id=active_response_id_before_clear,
            include_report_followup=False,
        )
        followthrough_chain_remaining_for_retry = api._response_done_followthrough_chain_remaining(
            turn_id=continuity_candidate_turn_id,
            origin=active_response_origin_before_clear,
            response_id=active_response_id_before_clear,
            include_report_followup=True,
        )
        required_deliverable_path_pending = api._response_done_marks_required_deliverable_followthrough(
            origin=active_response_origin_before_clear,
            turn_id=continuity_candidate_turn_id,
            response_id=active_response_id_before_clear,
            trace_context=trace_context,
            stale_context=stale_context,
        )
        selected_response_has_terminal_text = api._selected_response_has_terminal_text_evidence(
            response_id=resolved_response_id or active_response_id_before_clear
        )
        continuity_completion_candidate = (
            selected
            and selection_reason == "normal"
            and selected_response_has_terminal_text
        )
        continuity_close_commitment = (
            continuity_close_allowed
            and continuity_origin == "tool_output"
            and not followthrough_chain_remaining_for_close
        )
        if required_deliverable_path_pending and not continuity_completion_candidate:
            continuity_close_commitment = False
        continuity_close_unresolved = (
            continuity_close_commitment
            and not obligation_open
            and not followthrough_chain_remaining_for_retry
        )
        # NOTE: This flag name is legacy compound-centric language. In practice this
        # marks that response.done delivered terminal substantive fulfillment
        # (selected + normal + terminal text evidence).
        continuity_complete_required_deliverable = (
            continuity_close_commitment and continuity_completion_candidate
        )
        continuity_turn_id = turn_id
        continuity_rebind_allowed = False
        continuity_rebind_reason = ""
        if (
            continuity_close_commitment
            and continuity_candidate_rebind
        ):
            continuity_turn_id = continuity_candidate_turn_id
            continuity_rebind_allowed = True
            continuity_rebind_reason = "semantic_owner_parent_promoted"
        # Use raw execution seam facts (done_canonical_key + selection_reason) for
        # observability to avoid conflating semantic-owner promotions.
        breach_artifact = detect_contract_breach(
            ContractBreachSnapshot(
                source_seam="response_terminal_handlers",
                turn_id=turn_id,
                response_id=str(active_response_id_before_clear or "none"),
                origin=str(active_response_origin_before_clear or "unknown").strip().lower(),
                canonical_key=done_canonical_key,
                reason_code=str(selection_reason or "unknown"),
                is_terminal_event=True,
                selected_deliverable=bool(selected),
                is_empty_done=bool(is_empty_done),
                pending_tool_followup=bool(pending_tool_followup_after_release),
                followthrough_chain_remaining=bool(followthrough_chain_remaining_for_close),
            )
        )
        if breach_artifact is not None:
            # Advisory observability context only; this must not influence breach detection semantics.
            active_step_index, next_pending_step_id = self._contract_breach_step_context(turn_id=turn_id)
            logger.info(
                "contract_breach_detected breach_type=%s source_seam=%s canonical_key=%s turn_id=%s response_id=%s origin=%s reason_code=%s recommended_action=%s fingerprint=%s active_step_index=%s next_pending_step_id=%s evidence=%s",
                breach_artifact.breach_type.value,
                breach_artifact.source_seam,
                breach_artifact.canonical_key,
                breach_artifact.turn_id,
                breach_artifact.response_id,
                breach_artifact.origin,
                breach_artifact.reason_code,
                breach_artifact.recommended_action.value,
                breach_artifact.fingerprint,
                active_step_index,
                next_pending_step_id,
                "|".join(breach_artifact.evidence),
            )

        logger.info(
            "continuity_response_done_handoff turn_id=%s continuity_turn_id=%s selected=%s selection_reason=%s close_commitment=%s close_unresolved=%s complete_required_deliverable=%s followthrough_chain_remaining=%s allow_cross_turn_rebind=%s rebind_reason=%s semantic_parent_turn_id=%s done_canonical_key=%s semantic_owner_canonical_key=%s",
            turn_id,
            continuity_turn_id,
            selected,
            str(selection_reason or "").strip() or "unknown",
            continuity_close_commitment,
            continuity_close_unresolved,
            continuity_complete_required_deliverable,
            followthrough_chain_remaining_for_close,
            continuity_rebind_allowed,
            continuity_rebind_reason or "none",
            semantic_parent_turn_id or "none",
            done_canonical_key,
            semantic_owner_canonical_key or "none",
        )
        api._apply_continuity_event(
            "response_done",
            run_id=api._current_run_id(),
            turn_id=continuity_turn_id,
            response_id=active_response_id_before_clear,
            origin=active_response_origin_before_clear,
            keep_ongoing="true" if provisional_server_auto_close_deferred or exact_phrase_close_deferred or interrupted_tool_output_candidate or followthrough_chain_bridge or required_deliverable_missing_substance or required_deliverable_missing_tool_execution else "",
            close_ongoing="true" if continuity_close_allowed else "",
            close_commitment="true" if continuity_close_commitment else "",
            close_unresolved="true" if continuity_close_unresolved else "",
            complete_required_deliverable="true" if continuity_complete_required_deliverable else "",
            complete_final_report="true" if continuity_complete_required_deliverable else "",
            allow_cross_turn_rebind="true" if continuity_rebind_allowed else "",
            cross_turn_rebind_reason=continuity_rebind_reason,
        )
        followthrough_bridge_retry_block_reason = ""
        if followthrough_chain_bridge and followthrough_chain_remaining_for_retry:
            followthrough_bridge_retry_block_reason = "followthrough_chain_remaining"
        elif followthrough_chain_bridge and pending_tool_followup_any_after_release:
            followthrough_bridge_retry_block_reason = "pending_tool_followup_after_release"
        if interrupted_tool_output_candidate or followthrough_bridge_retry_block_reason:
            logger.info(
                "empty_response_retry_skipped reason=%s run_id=%s turn_id=%s response_id=%s",
                (
                    "interrupted_tool_output_candidate"
                    if interrupted_tool_output_candidate
                    else followthrough_bridge_retry_block_reason
                ),
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
            )
        else:
            await api._maybe_schedule_empty_response_retry(
                websocket=api.websocket,
                turn_id=turn_id,
                canonical_key=done_canonical_key,
                input_event_key=done_input_event_key,
                origin=active_response_origin_before_clear,
                delivery_state_before_done=delivery_state_before_done,
            )
        api._clear_terminal_response_text(response_id=active_response_id)
        if event:
            api._last_response_metadata = {
                "event_type": event.get("type"),
                "response": event.get("response"),
                "rate_limits": api.rate_limits,
            }
        api._emit_preference_recall_skip_trace_if_needed(turn_id=api._current_response_turn_id)
        if not exact_phrase_close_deferred:
            api._log_turn_conversation_efficiency(
                turn_id=turn_id,
                canonical_key=semantic_owner_canonical_key or resolved_canonical_key,
                close_reason="response_done",
            )
        transition = api._build_confirmation_transition_decision(
            reason="response_done",
            include_reminder_gate=True,
            was_confirmation_guarded=was_confirmation_guarded,
        )
        token_active, awaiting_phase, _, phase_is_awaiting_confirmation = api._confirmation_hold_components()
        if exact_phrase_close_deferred:
            logger.info(
                "turn_terminal_close_deferred run_id=%s turn_id=%s response_id=%s reason=exact_phrase_obligation_open",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
            )
            return
        if provisional_server_auto_close_deferred:
            logger.info(
                "turn_terminal_close_deferred run_id=%s turn_id=%s response_id=%s reason=provisional_server_auto_awaiting_transcript_final",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
            )
            return
        # Intentional: response.done and response.completed both attempt release.
        # The release helper is idempotent, and dual seam coverage avoids misses
        # when providers emit one terminal event but not the other.
        api._release_preference_recall_reentry_for_canonical_key(
            canonical_key=resolved_canonical_key,
            reason="response_done_terminal_close",
        )
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
        if pending_tool_followup_after_release:
            logger.info(
                "response_done_mic_recovery_deferred run_id=%s turn_id=%s response_id=%s canonical_key=%s origin=%s trigger=%s reason=pending_tool_followup",
                api._current_run_id() or "",
                turn_id,
                str(active_response_id_before_clear or "none"),
                resolved_canonical_key,
                str(active_response_origin_before_clear or "unknown"),
                "response_done_terminal",
            )
        else:
            api._maybe_recover_mic_after_response_done(
                turn_id=turn_id,
                response_id=active_response_id_before_clear,
                canonical_key=resolved_canonical_key,
                origin=str(active_response_origin_before_clear or "unknown"),
                trigger="response_done_terminal",
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
        api._maybe_log_continuity_debug_summary_on_turn_close(
            run_id=api._current_run_id(),
            turn_id=turn_id,
            reason="turn_terminal_close",
        )
        if api._pending_image_stimulus and not api._pending_image_flush_after_playback:
            await api._flush_pending_image_stimulus("response done")

    async def handle_response_completed(self, event: dict[str, Any] | None = None) -> None:
        api = self._api
        response_id = api._response_id_from_event(event) or str(getattr(api, "_active_response_id", "") or "").strip()
        turn_id = api._current_turn_id_or_unknown()
        done_input_event_key = str(getattr(api, "_active_response_input_event_key", "") or "").strip()
        done_canonical_key = api._canonical_utterance_key(turn_id=turn_id, input_event_key=done_input_event_key)
        active_response_origin_before_clear = str(getattr(api, "_active_response_origin", "") or "").strip().lower()
        if not api._should_admit_response_terminal_event(
            event_type=str((event or {}).get("type") or "response.completed"),
            response_id=response_id or None,
            turn_id=turn_id,
            input_event_key=done_input_event_key or None,
            canonical_key=done_canonical_key or None,
            origin=active_response_origin_before_clear or "unknown",
        ):
            return
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
        api._clear_active_response_state()
        api._active_server_auto_input_event_key = None
        current_state = getattr(api.state_manager, "state", InteractionState.IDLE)
        if current_state != InteractionState.LISTENING:
            api.state_manager.update_state(InteractionState.IDLE, "response completed")
        else:
            logger.debug(
                "Skipping IDLE transition for response.completed while still listening; deferring until speech stop."
            )
        logger.info("Received response.completed event.")
        api._clear_terminal_response_text(response_id=response_id)
        if active_response_origin_before_clear == "prompt":
            startup_terminal_state = "completed"
            startup_terminal_reason = "response_done"
            if str(api._response_delivery_state(turn_id=turn_id, input_event_key=done_input_event_key) or "").strip().lower() == "cancelled":
                startup_terminal_state = "skipped"
                startup_terminal_reason = "cancelled_before_done"
            api._emit_startup_prompt_terminal(
                terminal_state=startup_terminal_state,
                reason=startup_terminal_reason,
                turn_id=turn_id,
                input_event_key=done_input_event_key,
                canonical_key=done_canonical_key,
            )
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
        # Intentional duplicate with response.done path (idempotent helper).
        # Keep this seam-local release to cover completed-only terminal traces.
        api._release_preference_recall_reentry_for_canonical_key(
            canonical_key=done_canonical_key,
            reason="response_completed_terminal_close",
        )
        api._response_create_queue_drain_source = "active_cleared"
        await api._drain_response_create_queue(source_trigger="active_cleared")
        if api._pending_image_stimulus and not api._pending_image_flush_after_playback:
            await api._flush_pending_image_stimulus("response completed")
