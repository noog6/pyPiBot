"""Response lifecycle helpers (canonical keys + empty retry policy)."""

from __future__ import annotations

import hashlib
from typing import Any

from core.logging import logger


class ResponseLifecycleCoordinator:
    def __init__(self, api: Any) -> None:
        self._api = api

    def empty_response_retry_idempotency_key(self, *, canonical_key: str) -> str:
        normalized_canonical_key = str(canonical_key or "").strip() or "unknown"
        digest = hashlib.sha1(normalized_canonical_key.encode("utf-8")).hexdigest()[:16]
        return f"empty_response_done:{digest}"

    @staticmethod
    def strip_empty_retry_suffix_lineage(input_event_key: str | None) -> str:
        normalized_key = str(input_event_key or "").strip()
        if not normalized_key:
            return ""
        suffix = "__empty_retry"
        while normalized_key.endswith(suffix):
            normalized_key = normalized_key[: -len(suffix)]
        return normalized_key

    def canonical_key_for_empty_retry_origin(self, *, turn_id: str, input_event_key: str | None) -> str:
        origin_input_event_key = self.strip_empty_retry_suffix_lineage(input_event_key)
        return self._api._canonical_utterance_key(turn_id=turn_id, input_event_key=origin_input_event_key)

    def is_empty_response_done(self, *, canonical_key: str) -> bool:
        response_state = self._api._canonical_response_state(canonical_key)
        audio_delta_seen = bool(getattr(response_state, "audio_started", False)) or self._api._canonical_first_audio_started(canonical_key)
        assistant_reply_present = bool(str(getattr(self._api, "assistant_reply", "") or "").strip())
        assistant_buffer_present = bool(str(getattr(self._api, "_assistant_reply_accum", "") or "").strip())
        return not assistant_reply_present and not assistant_buffer_present and not audio_delta_seen

    async def maybe_schedule_empty_response_retry(
        self,
        *,
        websocket: Any,
        turn_id: str,
        canonical_key: str,
        input_event_key: str,
        origin: str,
        delivery_state_before_done: str | None,
    ) -> None:
        normalized_origin = str(origin or "unknown").strip().lower()
        run_id = self._api._current_run_id() or ""
        origin_input_event_key = self.strip_empty_retry_suffix_lineage(input_event_key)
        origin_canonical_key = self.canonical_key_for_empty_retry_origin(turn_id=turn_id, input_event_key=input_event_key)
        if not self.is_empty_response_done(canonical_key=canonical_key):
            return
        logger.info(
            "empty_response_detected run_id=%s turn_id=%s origin=%s canonical_key=%s origin_canonical_key=%s",
            run_id,
            turn_id,
            normalized_origin or "unknown",
            canonical_key,
            origin_canonical_key,
        )
        retry_registry = getattr(self._api, "_empty_response_retry_canonical_keys", None)
        if not isinstance(retry_registry, set):
            retry_registry = set()
            self._api._empty_response_retry_canonical_keys = retry_registry
        retry_counts = getattr(self._api, "_empty_response_retry_counts", None)
        if not isinstance(retry_counts, dict):
            retry_counts = {}
            self._api._empty_response_retry_counts = retry_counts
        max_attempts = max(1, int(getattr(self._api, "_empty_response_retry_max_attempts", 2) or 1))
        prior_attempts = int(retry_counts.get(origin_canonical_key, 0) or 0)
        if prior_attempts >= max_attempts:
            logger.info(
                "empty_response_retry_skipped reason=max_retries_exhausted run_id=%s turn_id=%s canonical_key=%s attempts=%s max_attempts=%s",
                run_id,
                turn_id,
                origin_canonical_key,
                prior_attempts,
                max_attempts,
            )
            await self._api._emit_empty_response_retry_exhausted_fallback(
                websocket=websocket,
                turn_id=turn_id,
                input_event_key=origin_input_event_key,
                canonical_key=origin_canonical_key,
                origin=normalized_origin,
            )
            return
        if canonical_key in retry_registry:
            logger.info("empty_response_retry_skipped reason=already_retried run_id=%s turn_id=%s", run_id, turn_id)
            return
        state_not_retryable = websocket is None or normalized_origin not in {"prompt", "server_auto", "user_transcript"} or delivery_state_before_done == "cancelled"
        if state_not_retryable:
            logger.info("empty_response_retry_skipped reason=state_not_retryable run_id=%s turn_id=%s", run_id, turn_id)
            return
        retry_registry.add(canonical_key)
        retry_counts[origin_canonical_key] = prior_attempts + 1
        retry_input_event_key = f"{origin_input_event_key or 'unknown'}__empty_retry"
        response_create_event = {
            "type": "response.create",
            "response": {
                "metadata": {
                    "turn_id": turn_id,
                    "input_event_key": retry_input_event_key,
                    "retry_reason": "empty_response_done",
                    "idempotency_key": self.empty_response_retry_idempotency_key(canonical_key=f"{origin_canonical_key}:attempt:{prior_attempts + 1}"),
                }
            },
        }
        sent = await self._api._send_response_create(websocket, response_create_event, origin=normalized_origin, record_ai_call=True)
        if sent:
            logger.info(
                "empty_response_retry_scheduled run_id=%s turn_id=%s canonical_key=%s attempt=%s max_attempts=%s",
                run_id,
                turn_id,
                origin_canonical_key,
                prior_attempts + 1,
                max_attempts,
            )
        else:
            logger.info("empty_response_retry_skipped reason=state_not_retryable run_id=%s turn_id=%s", run_id, turn_id)
