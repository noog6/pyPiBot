"""Response lifecycle helpers (canonical keys + deterministic empty retry policy)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.logging import logger


class EmptyResponseDecisionAction(str, Enum):
    NOOP = "NOOP"
    SCHEDULE_RETRY = "SCHEDULE_RETRY"
    EMIT_EXHAUSTED_FALLBACK = "EMIT_EXHAUSTED_FALLBACK"


@dataclass(frozen=True)
class EmptyResponseDecision:
    action: EmptyResponseDecisionAction
    reason_code: str


TERMINAL_DELIVERY_STATES = frozenset({"cancelled", "failed", "errored"})
RETRYABLE_ORIGINS = frozenset({"prompt", "server_auto", "user_transcript"})


def decide_empty_response_done_action(
    *,
    origin: str,
    delivery_state_before_done: str | None,
    assistant_text_present: bool,
    audio_started: bool,
    attempt_count: int,
    max_attempts: int,
    websocket_available: bool,
) -> EmptyResponseDecision:
    normalized_origin = str(origin or "unknown").strip().lower() or "unknown"
    normalized_delivery_state = str(delivery_state_before_done or "").strip().lower()
    normalized_attempt_count = max(0, int(attempt_count or 0))
    normalized_max_attempts = max(1, int(max_attempts or 1))

    if assistant_text_present or audio_started:
        return EmptyResponseDecision(EmptyResponseDecisionAction.NOOP, "non_empty_response")
    if not websocket_available:
        return EmptyResponseDecision(EmptyResponseDecisionAction.NOOP, "websocket_unavailable")
    if normalized_origin not in RETRYABLE_ORIGINS:
        return EmptyResponseDecision(EmptyResponseDecisionAction.NOOP, "origin_not_retryable")
    if normalized_delivery_state in TERMINAL_DELIVERY_STATES:
        return EmptyResponseDecision(EmptyResponseDecisionAction.NOOP, "delivery_state_terminal")
    if normalized_attempt_count >= normalized_max_attempts:
        return EmptyResponseDecision(
            EmptyResponseDecisionAction.EMIT_EXHAUSTED_FALLBACK,
            "max_retries_exhausted",
        )
    return EmptyResponseDecision(EmptyResponseDecisionAction.SCHEDULE_RETRY, "empty_response_done")


class ResponseLifecycleTracker:
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

    @staticmethod
    def is_terminal_delivery_state(delivery_state: str | None) -> bool:
        return str(delivery_state or "").strip().lower() in TERMINAL_DELIVERY_STATES

    def mark_created(self, *, canonical_key: str) -> None:
        if not str(canonical_key or "").strip():
            return
        created_registry = getattr(self._api, "_response_created_canonical_keys", None)
        if not isinstance(created_registry, set):
            created_registry = set()
            self._api._response_created_canonical_keys = created_registry
        created_registry.add(canonical_key)

    def mark_done(self, *, canonical_key: str) -> None:
        if not str(canonical_key or "").strip():
            return
        done_registry = getattr(self._api, "_response_done_canonical_keys", None)
        if not isinstance(done_registry, set):
            done_registry = set()
            self._api._response_done_canonical_keys = done_registry
        done_registry.add(canonical_key)

    def retry_registry(self) -> set[str]:
        retry_registry = getattr(self._api, "_empty_response_retry_canonical_keys", None)
        if not isinstance(retry_registry, set):
            retry_registry = set()
            self._api._empty_response_retry_canonical_keys = retry_registry
        return retry_registry

    def retry_counts(self) -> dict[str, int]:
        retry_counts = getattr(self._api, "_empty_response_retry_counts", None)
        if not isinstance(retry_counts, dict):
            retry_counts = {}
            self._api._empty_response_retry_counts = retry_counts
        return retry_counts

    def attempt_count(self, *, origin_canonical_key: str) -> int:
        return int(self.retry_counts().get(origin_canonical_key, 0) or 0)

    def max_attempts(self) -> int:
        return max(1, int(getattr(self._api, "_empty_response_retry_max_attempts", 2) or 1))

    def is_empty_response_done(self, *, canonical_key: str) -> bool:
        response_state = self._api._canonical_response_state(canonical_key)
        audio_delta_seen = bool(getattr(response_state, "audio_started", False)) or self._api._canonical_first_audio_started(canonical_key)
        deliverable_observed = bool(getattr(response_state, "deliverable_observed", False))
        assistant_reply_present = bool(str(getattr(self._api, "assistant_reply", "") or "").strip())
        assistant_buffer_present = bool(str(getattr(self._api, "_assistant_reply_accum", "") or "").strip())
        return (
            not assistant_reply_present
            and not assistant_buffer_present
            and not audio_delta_seen
            and not deliverable_observed
        )

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
        retry_registry = self.retry_registry()
        retry_counts = self.retry_counts()
        max_attempts = self.max_attempts()
        prior_attempts = self.attempt_count(origin_canonical_key=origin_canonical_key)
        if canonical_key in retry_registry:
            logger.info("empty_response_retry_skipped reason=already_retried run_id=%s turn_id=%s", run_id, turn_id)
            return

        decision = decide_empty_response_done_action(
            origin=normalized_origin,
            delivery_state_before_done=delivery_state_before_done,
            assistant_text_present=False,
            audio_started=False,
            attempt_count=prior_attempts,
            max_attempts=max_attempts,
            websocket_available=websocket is not None,
        )
        if decision.action == EmptyResponseDecisionAction.NOOP:
            logger.info(
                "empty_response_retry_skipped reason=%s run_id=%s turn_id=%s",
                decision.reason_code,
                run_id,
                turn_id,
            )
            return
        if hasattr(self._api, "_turn_has_final_deliverable") and self._api._turn_has_final_deliverable(turn_id=turn_id):
            logger.info(
                "empty_response_retry_skipped reason=turn_final_deliverable run_id=%s turn_id=%s",
                run_id,
                turn_id,
            )
            return
        if decision.action == EmptyResponseDecisionAction.EMIT_EXHAUSTED_FALLBACK:
            logger.info(
                "empty_response_retry_skipped reason=%s run_id=%s turn_id=%s canonical_key=%s attempts=%s max_attempts=%s",
                decision.reason_code,
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
