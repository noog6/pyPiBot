"""Runtime service for per-turn memory retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import time
import re
from typing import Any, Protocol

from core.logging import logger
from services.memory_manager import render_realtime_memory_brief_item, render_startup_memory_digest_item
from services.memory_manager import MemoryScope


MEMORY_INTENT_MARKERS = (
    "remember",
    "memory",
    "memories",
    "recall",
    "do you know",
    "what's my name",
    "what is my name",
    "my name",
    "prefer",
    "preferred",
    "preference",
    "favorite",
    "favourite",
)

PREFERENCE_MEMORY_MARKERS = (
    "prefer",
    "preferred",
    "preference",
    "favorite",
    "favourite",
    "what do i use",
    "remind me",
    "which",
)

PREFERENCE_MEMORY_DOMAINS = {
    "editor",
    "ide",
    "tool",
    "workflow",
    "language",
    "theme",
    "music",
    "drink",
    "food",
}


class MemoryRuntimeAPI(Protocol):
    _pending_turn_memory_brief: object | None
    _memory_retrieval_last_error_log_at: float
    _memory_retrieval_suppressed_errors: int


@dataclass(frozen=True)
class PerceptionMemoryVerdict:
    """Structured perception/memory result with runtime effects applied later."""

    memory_intent: bool = False
    memory_intent_subtype: str = "none"
    preference_recall_requested: bool = False
    preference_recall_context: dict[str, Any] | None = None
    companion_gesture_tool_request: dict[str, Any] | None = None
    runtime_request: dict[str, Any] | None = None

    @classmethod
    def empty(cls, *, memory_intent_subtype: str = "none") -> "PerceptionMemoryVerdict":
        normalized_subtype = str(memory_intent_subtype or "none").strip() or "none"
        return cls(
            memory_intent=normalized_subtype != "none",
            memory_intent_subtype=normalized_subtype,
        )


@dataclass
class MemoryRuntime:
    """Handles retrieval of turn-scoped memory briefs."""

    api: MemoryRuntimeAPI

    def classify_memory_intent(self, text: str) -> str:
        normalized = " ".join((text or "").lower().split())
        if not normalized:
            return "none"
        if not any(marker in normalized for marker in MEMORY_INTENT_MARKERS):
            return "none"

        has_preference_marker = any(marker in normalized for marker in PREFERENCE_MEMORY_MARKERS)
        has_preference_domain = any(domain in normalized for domain in PREFERENCE_MEMORY_DOMAINS)
        if has_preference_marker and has_preference_domain:
            return "preference_recall"

        if "remember about" in normalized or "what do you remember" in normalized:
            return "topic_recall"

        return "general_memory"

    def is_memory_intent(self, text: str) -> bool:
        return self.classify_memory_intent(text) != "none"

    def build_memory_retrieval_query(self, user_text: str, *, memory_intent_subtype: str) -> str:
        normalized = " ".join((user_text or "").split())
        if memory_intent_subtype == "topic_recall":
            lowered = normalized.lower()
            for marker in ("remember about", "remember of"):
                if marker not in lowered:
                    continue
                topic_fragment = lowered.split(marker, 1)[1]
                topic_fragment = re.split(r"[?.!,;:]", topic_fragment, maxsplit=1)[0]
                topic_fragment = re.sub(r"\b(my|the|a|an)\b", " ", topic_fragment)
                topic = " ".join(topic_fragment.split())
                if topic:
                    return topic
        return normalized

    def should_skip_turn_memory_retrieval(self, user_text: str) -> bool:
        text = user_text.strip()
        if len(text) < int(getattr(self.api, "_memory_retrieval_min_user_chars", 12)):
            return True
        if len(text.split()) < int(getattr(self.api, "_memory_retrieval_min_user_tokens", 3)):
            return True
        noisy = {"ok", "okay", "thanks", "thank you", "yes", "no", "cool", "nice", "got it"}
        return text.lower() in noisy

    def prepare_turn_memory_brief(
        self,
        user_text: str,
        *,
        source: str,
        memory_intent: bool = False,
        memory_intent_subtype: str = "none",
    ) -> None:
        api = self.api
        if self.should_skip_turn_memory_retrieval(user_text):
            api._pending_turn_memory_brief = None
            return
        if not getattr(api, "_memory_retrieval_enabled", False):
            api._pending_turn_memory_brief = None
            return
        manager = getattr(api, "_memory_manager", None)
        if manager is None:
            api._pending_turn_memory_brief = None
            return
        retrieval_query = self.build_memory_retrieval_query(
            user_text,
            memory_intent_subtype=memory_intent_subtype,
        )
        try:
            api._pending_turn_memory_brief = manager.retrieve_for_turn(
                latest_user_utterance=retrieval_query,
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
                        "turn_memory_retrieval_cooldown_bypassed source=%s cooldown_bypassed_reason=memory_intent memory_intent_subtype=%s retrieval_query=%s",
                        source,
                        memory_intent_subtype,
                        retrieval_query,
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

    def consume_pending_memory_brief_note(self) -> str | None:
        brief = getattr(self.api, "_pending_turn_memory_brief", None)
        self.api._pending_turn_memory_brief = None
        if brief is None or not brief.items:
            return None
        lines = [
            "Turn memory brief (retrieved long-term context; do not quote verbatim unless asked):"
        ]
        for index, item in enumerate(brief.items, start=1):
            lines.append(render_realtime_memory_brief_item(index=index, item=item))
        if brief.truncated:
            lines.append("Additional relevant memories were omitted due to retrieval limits.")
        return "\n".join(lines)

    def build_startup_memory_digest_note(self) -> str | None:
        api = self.api
        if not getattr(api, "_startup_memory_digest_enabled", True):
            return None
        manager = getattr(api, "_memory_manager", None)
        if manager is None:
            return None
        try:
            digest = manager.retrieve_startup_digest(
                max_items=int(getattr(api, "_startup_memory_digest_max_items", 2)),
                max_chars=int(getattr(api, "_startup_memory_digest_max_chars", 280)),
                user_id=manager.get_active_user_id(),
            )
        except Exception as exc:  # pragma: no cover - defensive fail-open
            logger.warning("Startup memory digest retrieval failed: %s", exc)
            return None
        if digest is None or not digest.items:
            return None
        lines = ["Startup memory digest (stable user context):"]
        for index, item in enumerate(digest.items, start=1):
            lines.append(render_startup_memory_digest_item(index=index, item=item))
        if digest.truncated:
            lines.append("Additional pinned memories were omitted due to startup digest limits.")
        return "\n".join(lines)
