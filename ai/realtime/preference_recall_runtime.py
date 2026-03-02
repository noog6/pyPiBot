"""Extracted preference-recall runtime logic for RealtimeAPI."""
from __future__ import annotations

from collections import deque
import re
import time
from typing import Any

from core.logging import logger, log_ws_event
from ai.tools import function_map

async def _suppress_preference_recall_server_auto_response(controller, websocket: Any) -> None:
        turn_id = controller._current_turn_id_or_unknown()
        input_event_key = str(getattr(controller, "_current_input_event_key", "") or "").strip()
        pending_before = 1 if getattr(getattr(controller, "_pending_response_create", None), "__class__", type(None)).__name__ == "PendingResponseCreate" else 0
        queue_before = len(getattr(controller, "_response_create_queue", deque()) or ())
        suppressed_turns = getattr(controller, "_preference_recall_suppressed_turns", None)
        if not isinstance(suppressed_turns, set):
            suppressed_turns = set()
            controller._preference_recall_suppressed_turns = suppressed_turns
        suppressed_turns.add(turn_id)
        suppressed_input_event_keys = getattr(controller, "_preference_recall_suppressed_input_event_keys", None)
        if not isinstance(suppressed_input_event_keys, set):
            suppressed_input_event_keys = set()
            controller._preference_recall_suppressed_input_event_keys = suppressed_input_event_keys
        if input_event_key:
            suppressed_input_event_keys.add(input_event_key)
        controller._clear_pending_response_contenders(
            turn_id=turn_id,
            input_event_key=input_event_key,
            reason="preference_recall_suppressed",
        )
        controller._drop_suppressed_scheduled_response_creates(turn_id=turn_id, origin="server_auto")
        logger.info(
            "preference_recall_response_suppressed run_id=%s turn_id=%s input_event_key=%s reason=handled_preference_recall",
            controller._current_run_id() or "",
            turn_id,
            input_event_key or "unknown",
        )
        controller._log_response_site_debug(
            site="preference_recall_response_suppressed",
            turn_id=turn_id,
            input_event_key=input_event_key,
            canonical_key=controller._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key),
            origin="server_auto",
            trigger="handled_preference_recall",
        )
        pending_after = 1 if getattr(getattr(controller, "_pending_response_create", None), "__class__", type(None)).__name__ == "PendingResponseCreate" else 0
        queue_after = len(getattr(controller, "_response_create_queue", deque()) or ())
        active_server_auto_key = str(getattr(controller, "_active_server_auto_input_event_key", "") or "").strip() or "unknown"
        canonical_key = controller._canonical_utterance_key(turn_id=turn_id, input_event_key=input_event_key)
        logger.debug(
            "[RESPTRACE] suppression_applied run_id=%s turn_id=%s input_event_key=%s canonical_key=%s "
            "suppressed_turns_count=%s suppressed_keys_count=%s removed_pending_count=%s removed_queued_count=%s "
            "active_server_auto_key=%s",
            controller._current_run_id() or "",
            turn_id,
            input_event_key or "unknown",
            canonical_key,
            len(suppressed_turns),
            len(suppressed_input_event_keys),
            max(0, pending_before - pending_after),
            max(0, queue_before - queue_after),
            active_server_auto_key,
        )
        if str(getattr(controller, "_active_response_origin", "")).strip().lower() != "server_auto":
            return
        active_key = str(getattr(controller, "_active_server_auto_input_event_key", "") or "").strip()
        if input_event_key and active_key and active_key != input_event_key:
            return
        if not bool(getattr(controller, "_response_in_flight", False)):
            return
        cancel_event = {"type": "response.cancel"}
        log_ws_event("Outgoing", cancel_event)
        controller._track_outgoing_event(cancel_event, origin="preference_recall_guard")
        try:
            transport = controller._get_or_create_transport()
            await transport.send_json(websocket, cancel_event)
        except Exception as exc:
            message = str(exc)
            if "no active response found" in message.lower():
                logger.info(
                    "response_cancel_noop run_id=%s turn_id=%s reason=no_active_response",
                    controller._current_run_id() or "",
                    turn_id,
                )
                logger.debug(
                    "response_cancel_noop_detail run_id=%s turn_id=%s error=%s",
                    controller._current_run_id() or "",
                    turn_id,
                    message,
                )
                return
            logger.exception(
                "response_cancel_failed run_id=%s turn_id=%s origin=preference_recall_guard",
                controller._current_run_id() or "",
                turn_id,
            )

def _emit_preference_recall_skip_trace_if_needed(controller, *, turn_id: str | None) -> None:
        pending = controller._pending_preference_recall_trace if isinstance(controller._pending_preference_recall_trace, dict) else None
        if not pending:
            return
        resolved_turn_id = str(turn_id or "").strip() or "turn-unknown"
        if resolved_turn_id in controller._preference_recall_skip_logged_turn_ids:
            return
        if pending.get("intent") != "preference_recall":
            return
        recall_invoked = any(
            isinstance(record, dict)
            and record.get("name") == "recall_memories"
            and (
                str(record.get("turn_id") or "").strip() in {"", resolved_turn_id}
            )
            for record in (controller._tool_call_records or [])
        )
        if recall_invoked:
            controller._clear_preference_recall_candidate()
            return
        controller._preference_recall_skip_logged_turn_ids.add(resolved_turn_id)
        logger.info(
            "preference_recall_decision_trace intent=%s decision=%s reason=%s query_fingerprint=%s run_id=%s turn_id=%s source=%s",
            pending.get("intent", "preference_recall"),
            pending.get("decision", "skipped_tool"),
            pending.get("reason", "model_did_not_request_tool"),
            pending.get("query_fingerprint", ""),
            controller._current_run_id() or "",
            resolved_turn_id,
            pending.get("source", "unknown"),
        )
        controller._clear_preference_recall_candidate()

async def _maybe_handle_preference_recall_intent(controller, text: str, websocket: Any, *, source: str) -> bool:
        matched, keywords = controller._is_preference_recall_intent(text)
        if not matched:
            return False

        resolved_turn_id = controller._current_turn_id_or_unknown()
        replacement_input_event_key = str(getattr(controller, "_current_input_event_key", "") or "").strip()
        locked_input_event_keys = getattr(controller, "_preference_recall_locked_input_event_keys", None)
        if not isinstance(locked_input_event_keys, set):
            locked_input_event_keys = set()
            controller._preference_recall_locked_input_event_keys = locked_input_event_keys
        if replacement_input_event_key:
            locked_input_event_keys.add(replacement_input_event_key)
            logger.debug(
                "preference_recall_lock_set run_id=%s input_event_key=%s reason=intent_matched",
                controller._current_run_id() or "",
                replacement_input_event_key,
            )

        turn_timestamps_store = getattr(controller, "_turn_diagnostic_timestamps", None)
        if not isinstance(turn_timestamps_store, dict):
            turn_timestamps_store = {}
            controller._turn_diagnostic_timestamps = turn_timestamps_store
        turn_timestamps = turn_timestamps_store.setdefault(resolved_turn_id, {})
        turn_timestamps["preference_recall_start"] = time.monotonic()
        controller._mark_preference_recall_candidate(text, source=source)
        try:
            query = controller._build_preference_recall_query(text.lower(), keywords=keywords)
            now = time.monotonic()
            cooldown_s = max(0.0, float(getattr(controller, "_preference_recall_cooldown_s", 0.0)))
            cached = controller._preference_recall_cache.get(query)
            result_payload: dict[str, Any] | None = None
            recall_invoked = False
            if (
                cached
                and cooldown_s > 0.0
                and now - float(cached.get("timestamp", 0.0)) < cooldown_s
            ):
                result_payload = cached.get("payload") if isinstance(cached.get("payload"), dict) else None
                logger.info(
                    "Preference recall reused cached result source=%s query=%s cooldown_s=%.2f",
                    source,
                    query,
                    cooldown_s,
                )
            if result_payload is None:
                recall_fn = function_map.get("recall_memories")
                if recall_fn is None:
                    if isinstance(controller._pending_preference_recall_trace, dict):
                        controller._pending_preference_recall_trace["reason"] = "policy_skip"
                    logger.warning("Preference recall intent matched but recall_memories tool unavailable.")
                    return False
                recall_invoked = True
                tool_call_records = getattr(controller, "_tool_call_records", None)
                if isinstance(tool_call_records, list):
                    tool_call_records.append(
                        {
                            "name": "recall_memories",
                            "source": "preference_recall",
                            "turn_id": controller._current_turn_id_or_unknown(),
                            "query": query,
                        }
                    )
                if isinstance(controller._pending_preference_recall_trace, dict):
                    controller._pending_preference_recall_trace["decision"] = "invoked_tool"
                    controller._pending_preference_recall_trace["reason"] = "preference_intent_matched"
                result_payload, recall_invoked = await controller._run_preference_recall_with_fallbacks(
                    recall_fn=recall_fn,
                    source=source,
                    resolved_turn_id=resolved_turn_id,
                    query=query,
                )
                logger.info(
                    "Preference recall executed source=%s query=%s keywords=%s",
                    source,
                    query,
                    ",".join(keywords),
                )

            memories = controller._preference_recall_memories_from_payload(result_payload)
            cards = result_payload.get("memory_cards") if isinstance(result_payload, dict) else None
            cards_list = cards if isinstance(cards, list) else []
            memory_cards_text = ""
            if isinstance(result_payload, dict):
                memory_cards_text = controller._sanitize_memory_cards_text_for_user(
                    str(result_payload.get("memory_cards_text", "")).strip()
                )
            hit = (
                bool(memory_cards_text)
                or bool(cards_list)
                or (isinstance(memories, list) and len(memories) > 0)
            )
            await controller._suppress_preference_recall_server_auto_response(websocket)
            # preference_recall is an internal tool pathway: suppress server_auto output,
            # then guarantee a replacement assistant response for this input_event_key.
            if replacement_input_event_key and controller._is_input_event_key_already_scheduled(
                input_event_key=replacement_input_event_key
            ):
                logger.info(
                    "preference_recall_replacement_skip run_id=%s turn_id=%s input_event_key=%s reason=already_scheduled_for_input_event_key",
                    controller._current_run_id() or "",
                    resolved_turn_id,
                    replacement_input_event_key,
                )
                controller._clear_preference_recall_candidate()
                return True
            if not hit:
                await controller.send_assistant_message(
                    "I don’t have that saved yet. If you share it, I can remember it for next time.",
                    websocket,
                    response_metadata={
                        "turn_id": resolved_turn_id,
                        "input_event_key": replacement_input_event_key,
                        "trigger": "preference_recall",
                    },
                )
                controller._mark_input_event_key_scheduled(input_event_key=replacement_input_event_key)
                turn_timestamps["preference_recall_end"] = time.monotonic()
                handled_logged_turn_ids = getattr(controller, "_preference_recall_handled_logged_turn_ids", None)
                if not isinstance(handled_logged_turn_ids, set):
                    handled_logged_turn_ids = set()
                    controller._preference_recall_handled_logged_turn_ids = handled_logged_turn_ids
                if resolved_turn_id not in handled_logged_turn_ids:
                    handled_logged_turn_ids.add(resolved_turn_id)
                    logger.info(
                        "preference_recall_handled run_id=%s resolved_turn_id=%s source=%s query=%s",
                        controller._current_run_id() or "",
                        resolved_turn_id,
                        source,
                        query,
                    )
                    logger.debug(
                        "turn_diagnostic_timestamps run_id=%s turn_id=%s transcript_final_ts=%s preference_recall_start_ts=%s preference_recall_end_ts=%s response_schedule_ts=%s",
                        controller._current_run_id() or "",
                        resolved_turn_id,
                        turn_timestamps.get("transcript_final"),
                        turn_timestamps.get("preference_recall_start"),
                        turn_timestamps.get("preference_recall_end"),
                        turn_timestamps.get("response_schedule"),
                    )
                controller._clear_preference_recall_candidate()
                return True

            response_lines: list[str] = []
            if recall_invoked:
                response_lines.append("I’m checking what I remember.")
            if memory_cards_text:
                response_lines.append(memory_cards_text)
            else:
                first_memory = memories[0] if isinstance(memories[0], dict) else {}
                memory_text = str(first_memory.get("content", "")).strip()
                if not memory_text:
                    memory_text = "I found a saved preference, but it does not include enough detail to quote yet."
                response_lines.append(f'Relevant memory: "{memory_text}"')
                response_lines.append("Why it's relevant: " + '"It matches your preference question."')
                response_lines.append("Confidence: medium")

            if controller._memory_pin_followup_needed(cards=cards_list, memories=memories, query=query):
                response_lines.append("Want me to pin or rename this memory so it’s easier to recall later?")

            response_text = controller._normalize_memory_recall_answer("\n".join(response_lines))
            await controller.send_assistant_message(
                response_text,
                websocket,
                response_metadata={
                    "turn_id": resolved_turn_id,
                    "input_event_key": replacement_input_event_key,
                    "trigger": "preference_recall",
                },
            )
            controller._mark_input_event_key_scheduled(input_event_key=replacement_input_event_key)
            turn_timestamps["preference_recall_end"] = time.monotonic()
            handled_logged_turn_ids = getattr(controller, "_preference_recall_handled_logged_turn_ids", None)
            if not isinstance(handled_logged_turn_ids, set):
                handled_logged_turn_ids = set()
                controller._preference_recall_handled_logged_turn_ids = handled_logged_turn_ids
            if resolved_turn_id not in handled_logged_turn_ids:
                handled_logged_turn_ids.add(resolved_turn_id)
                logger.info(
                    "preference_recall_handled run_id=%s resolved_turn_id=%s source=%s query=%s",
                    controller._current_run_id() or "",
                    resolved_turn_id,
                    source,
                    query,
                )
                logger.debug(
                    "turn_diagnostic_timestamps run_id=%s turn_id=%s transcript_final_ts=%s preference_recall_start_ts=%s preference_recall_end_ts=%s response_schedule_ts=%s",
                    controller._current_run_id() or "",
                    resolved_turn_id,
                    turn_timestamps.get("transcript_final"),
                    turn_timestamps.get("preference_recall_start"),
                    turn_timestamps.get("preference_recall_end"),
                    turn_timestamps.get("response_schedule"),
                )
            controller._clear_preference_recall_candidate()
            return True
        finally:
            if replacement_input_event_key:
                locked_input_event_keys.discard(replacement_input_event_key)
                logger.debug(
                    "preference_recall_lock_cleared run_id=%s input_event_key=%s reason=completed",
                    controller._current_run_id() or "",
                    replacement_input_event_key,
                )

def _find_stop_word(controller, text: str) -> str | None:
        if not text or not controller._stop_words:
            return None
        lowered = text.lower()
        for word in controller._stop_words:
            if " " in word:
                if word in lowered:
                    return word
                continue
            if re.search(rf"\\b{re.escape(word)}\\b", lowered):
                return word
        return None
