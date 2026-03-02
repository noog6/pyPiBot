"""Runtime service for per-turn memory retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Protocol

from core.logging import logger
from services.memory_manager import MemoryScope


class MemoryRuntimeAPI(Protocol):
    _pending_turn_memory_brief: object | None
    _memory_retrieval_last_error_log_at: float
    _memory_retrieval_suppressed_errors: int

    def _should_skip_turn_memory_retrieval(self, user_text: str) -> bool: ...


@dataclass
class MemoryRuntime:
    """Handles retrieval of turn-scoped memory briefs."""

    api: MemoryRuntimeAPI

    def prepare_turn_memory_brief(self, user_text: str, *, source: str, memory_intent: bool = False) -> None:
        api = self.api
        if api._should_skip_turn_memory_retrieval(user_text):
            api._pending_turn_memory_brief = None
            return
        if not getattr(api, "_memory_retrieval_enabled", False):
            api._pending_turn_memory_brief = None
            return
        manager = getattr(api, "_memory_manager", None)
        if manager is None:
            api._pending_turn_memory_brief = None
            return
        try:
            api._pending_turn_memory_brief = manager.retrieve_for_turn(
                latest_user_utterance=user_text,
                user_id=manager.get_active_user_id(),
                max_memories=int(getattr(api, "_memory_retrieval_max_memories", 3)),
                max_chars=int(getattr(api, "_memory_retrieval_max_chars", 450)),
                cooldown_s=float(getattr(api, "_memory_retrieval_cooldown_s", 10.0)),
                bypass_cooldown=memory_intent,
                scope=str(getattr(api, "_memory_retrieval_scope", MemoryScope.USER_GLOBAL.value)),
                session_id=manager.get_active_session_id(),
            )
            retrieval_debug = manager.get_last_turn_retrieval_debug_metadata()
            if retrieval_debug:
                if memory_intent:
                    logger.info(
                        "turn_memory_retrieval_cooldown_bypassed source=%s cooldown_bypassed_reason=memory_intent",
                        source,
                    )
                semantic_runtime_health = manager.get_semantic_runtime_health()
                semantic_streak = int(semantic_runtime_health.get("query_embedding_not_ready_streak", 0))
                semantic_health_suffix = (
                    " semantic_runtime_ready=%s semantic_runtime_streak=%s semantic_runtime_last_error=%s "
                    "semantic_runtime_readiness_last_transition_at=%s semantic_runtime_readiness_age_ms=%s "
                    "semantic_runtime_readiness_transition_count=%s"
                    % (
                        semantic_runtime_health.get("query_embedding_runtime_ready"),
                        semantic_streak,
                        semantic_runtime_health.get("query_embedding_last_error"),
                        semantic_runtime_health.get("query_embedding_readiness_last_transition_at"),
                        semantic_runtime_health.get("query_embedding_readiness_age_ms"),
                        semantic_runtime_health.get("query_embedding_readiness_transition_count"),
                    )
                    if semantic_runtime_health
                    else ""
                )
                logger.debug(
                    "turn_memory_retrieval_result source=%s mode=%s lexical_candidates=%s semantic_candidates=%s semantic_scored=%s "
                    "candidates_without_ready_embedding=%s candidates_below_threshold=%s candidates_semantic_applied=%s selected=%s "
                    "fallback_reason=%s latency_ms=%s truncated=%s truncation_count=%s dedupe_count=%s semantic_provider=%s semantic_model=%s semantic_query_timeout_ms=%s semantic_query_timeout_ms_used=%s semantic_query_duration_ms=%s semantic_query_embed_elapsed_ms=%s semantic_timeout_source=%s semantic_result_status=%s semantic_error_code=%s semantic_error_class=%s semantic_failure_class=%s canary_last_error_code=%s canary_last_latency_ms=%s canary_last_checked_age_ms=%s semantic_provider_last_error_code=%s timeout_backoff_until_remaining_ms=%s semantic_scoring_skipped_reason=%s query_fingerprint_hash=%s query_fingerprint_length=%s%s",
                    source,
                    retrieval_debug.get("mode"),
                    retrieval_debug.get("lexical_candidate_count"),
                    retrieval_debug.get("semantic_candidate_count"),
                    retrieval_debug.get("semantic_scored_count"),
                    retrieval_debug.get("candidates_without_ready_embedding"),
                    retrieval_debug.get("candidates_below_influence_threshold"),
                    retrieval_debug.get("candidates_semantic_applied"),
                    retrieval_debug.get("selected_count"),
                    retrieval_debug.get("fallback_reason"),
                    retrieval_debug.get("latency_ms"),
                    retrieval_debug.get("truncated"),
                    retrieval_debug.get("truncation_count"),
                    retrieval_debug.get("dedupe_count"),
                    retrieval_debug.get("semantic_provider"),
                    retrieval_debug.get("semantic_model"),
                    retrieval_debug.get("semantic_query_timeout_ms"),
                    retrieval_debug.get("semantic_query_timeout_ms_used"),
                    retrieval_debug.get("semantic_query_duration_ms"),
                    retrieval_debug.get("semantic_query_embed_elapsed_ms"),
                    retrieval_debug.get("semantic_timeout_source"),
                    retrieval_debug.get("semantic_result_status"),
                    retrieval_debug.get("semantic_error_code"),
                    retrieval_debug.get("semantic_error_class"),
                    retrieval_debug.get("semantic_failure_class"),
                    retrieval_debug.get("canary_last_error_code"),
                    retrieval_debug.get("canary_last_latency_ms"),
                    retrieval_debug.get("canary_last_checked_age_ms"),
                    retrieval_debug.get("semantic_provider_last_error_code"),
                    retrieval_debug.get("timeout_backoff_until_remaining_ms"),
                    retrieval_debug.get("semantic_scoring_skipped_reason"),
                    retrieval_debug.get("query_fingerprint_hash"),
                    retrieval_debug.get("query_fingerprint_length"),
                    semantic_health_suffix,
                )
        except Exception as exc:  # pragma: no cover - defensive fail-open
            api._pending_turn_memory_brief = None
            now = time.monotonic()
            throttle_s = max(1.0, float(getattr(api, "_memory_retrieval_error_throttle_s", 60.0)))
            should_log = (
                api._memory_retrieval_last_error_log_at <= 0.0
                or (now - api._memory_retrieval_last_error_log_at) >= throttle_s
            )
            if should_log:
                suppressed = int(getattr(api, "_memory_retrieval_suppressed_errors", 0))
                logger.warning(
                    "Turn memory retrieval failed for source=%s suppressed_since_last=%s: %s",
                    source,
                    suppressed,
                    exc,
                )
                api._memory_retrieval_last_error_log_at = now
                api._memory_retrieval_suppressed_errors = 0
            else:
                api._memory_retrieval_suppressed_errors += 1
