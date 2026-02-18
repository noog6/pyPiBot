"""Manage memory entries for long-term recall."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
import math
import re
import struct
import time

from config import ConfigController
from services.embedding_provider import EmbeddingProvider, build_embedding_provider
from services.memory_embedding_worker import MemoryEmbeddingWorker
from storage.memories import MemoryEntry, MemoryStore

MAX_CONTENT_LENGTH = 400
MAX_TAGS = 6
MAX_TAG_LENGTH = 24
MAX_RECALL_LIMIT = 10
MIN_IMPORTANCE = 1
MAX_IMPORTANCE = 5
RANK_CANDIDATE_MULTIPLIER = 8
STALE_MEMORY_MAX_AGE_S = 60.0 * 60.0 * 24.0 * 365.0
NEAR_DUPLICATE_THRESHOLD = 0.75
NEAR_DUPLICATE_CHAR_RATIO = 0.9
WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")


class MemoryScope(str, Enum):
    """Supported memory scope modes."""

    USER_GLOBAL = "user_global"
    SESSION_LOCAL = "session_local"


MemoryScopeInput = MemoryScope | str


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _normalize_scope(scope: MemoryScopeInput | None, *, fallback: MemoryScope) -> MemoryScope:
    if isinstance(scope, MemoryScope):
        return scope
    if isinstance(scope, str):
        lowered = scope.strip().lower()
        if lowered == MemoryScope.SESSION_LOCAL.value:
            return MemoryScope.SESSION_LOCAL
        if lowered == MemoryScope.USER_GLOBAL.value:
            return MemoryScope.USER_GLOBAL
    return fallback


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text)}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _recency_weight(age_s: float) -> float:
    if age_s <= 0.0:
        return 1.0
    # 30-day half-life keeps recent facts favored without erasing durable memories.
    return math.exp(-age_s / (60.0 * 60.0 * 24.0 * 30.0))


def _score_entry(entry: MemoryEntry, *, utterance_tokens: set[str], now_s: float) -> float:
    content_tokens = _tokenize(entry.content)
    tag_tokens = {tag.lower() for tag in entry.tags}
    lexical_overlap = _jaccard_similarity(utterance_tokens, content_tokens)
    tag_overlap = _jaccard_similarity(utterance_tokens, tag_tokens)

    importance_weight = max(0.0, (entry.importance - MIN_IMPORTANCE) / (MAX_IMPORTANCE - MIN_IMPORTANCE))
    entry_ts_s = entry.timestamp / 1000.0
    age_s = max(0.0, now_s - entry_ts_s)
    recency = _recency_weight(age_s)

    # Weighted blend: lexical relevance first, then tags/importance/recency.
    return (lexical_overlap * 0.45) + (tag_overlap * 0.20) + (importance_weight * 0.20) + (recency * 0.15)


def _cosine_similarity_bytes(
    *,
    query_vector: bytes,
    query_norm: float,
    entry_vector: bytes,
    entry_norm: float,
) -> float | None:
    if query_norm <= 0.0 or entry_norm <= 0.0:
        return None
    if not query_vector or not entry_vector or len(query_vector) != len(entry_vector):
        return None
    if len(query_vector) % 4 != 0:
        return None

    dot = 0.0
    for (qv,), (ev,) in zip(struct.iter_unpack("<f", query_vector), struct.iter_unpack("<f", entry_vector)):
        dot += float(qv) * float(ev)
    return dot / (query_norm * entry_norm)


def _is_near_duplicate(
    candidate_text: str,
    existing_text: str,
    *,
    token_threshold: float,
    char_threshold: float,
) -> bool:
    candidate_tokens = _tokenize(candidate_text)
    existing_tokens = _tokenize(existing_text)
    token_sim = _jaccard_similarity(candidate_tokens, existing_tokens)
    if token_sim >= token_threshold:
        return True
    char_ratio = SequenceMatcher(a=candidate_text.lower(), b=existing_text.lower()).ratio()
    return char_ratio >= char_threshold


def _is_stale(entry: MemoryEntry, *, now_s: float, max_age_s: float) -> bool:
    if max_age_s <= 0:
        return False
    entry_ts_s = entry.timestamp / 1000.0
    return (now_s - entry_ts_s) > max_age_s


def _normalize_content(content: str) -> str:
    trimmed = " ".join(content.strip().split())
    if len(trimmed) <= MAX_CONTENT_LENGTH:
        return trimmed
    return f"{trimmed[: MAX_CONTENT_LENGTH - 1]}…"


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if not tags:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tags:
        tag = raw.strip().lower()
        if not tag:
            continue
        tag = tag[:MAX_TAG_LENGTH]
        if tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
        if len(normalized) >= MAX_TAGS:
            break
    return normalized


@dataclass(frozen=True)
class MemorySummary:
    """Summarized memory entry suitable for prompts."""

    memory_id: int
    content: str
    tags: list[str]
    importance: int
    source: str
    pinned: bool
    needs_review: bool

    @classmethod
    def from_entry(cls, entry: MemoryEntry) -> "MemorySummary":
        return cls(
            memory_id=entry.memory_id,
            content=entry.content,
            tags=entry.tags,
            importance=entry.importance,
            source=entry.source,
            pinned=entry.pinned,
            needs_review=entry.needs_review,
        )


@dataclass(frozen=True)
class MemoryBrief:
    """Bounded memory context block for a single user turn."""

    items: list[MemorySummary]
    total_chars: int
    max_chars: int
    truncated: bool
    scope: MemoryScope


@dataclass(frozen=True)
class MemorySemanticConfig:
    """Semantic memory retrieval and embedding options."""

    enabled: bool
    provider: str
    rerank_enabled: bool
    max_candidates_for_semantic: int
    min_similarity: float
    rerank_influence_min_cosine: float
    dedupe_strong_match_cosine: float | None
    background_embedding_enabled: bool
    write_timeout_ms: int
    query_timeout_ms: int
    max_writes_per_minute: int
    max_queries_per_minute: int


class MemoryManager:
    """Singleton manager for memory storage access."""

    _instance: "MemoryManager | None" = None

    def __init__(self) -> None:
        if MemoryManager._instance is not None:
            raise RuntimeError("You cannot create another MemoryManager class")

        config = ConfigController.get_instance().get_config()
        self._active_user_id = config.get("active_user_id", "default")
        self._active_session_id = config.get("active_session_id")
        memory_cfg = config.get("memory") or {}
        self._default_scope = _normalize_scope(
            memory_cfg.get("default_scope"),
            fallback=MemoryScope.USER_GLOBAL,
        )
        self._auto_pin_min_importance = int(memory_cfg.get("auto_pin_min_importance", 5))
        self._auto_pin_requires_review = bool(memory_cfg.get("auto_pin_requires_review", True))
        auto_dedupe_cfg = memory_cfg.get("auto_reflection_semantic_dedupe") or {}
        self._auto_reflection_semantic_dedupe_enabled = bool(auto_dedupe_cfg.get("enabled", False))
        self._auto_reflection_dedupe_recent_limit = int(auto_dedupe_cfg.get("recent_approved_limit", 24))
        self._auto_reflection_dedupe_high_risk_cosine = float(auto_dedupe_cfg.get("high_risk_cosine", 0.90))
        self._auto_reflection_dedupe_policy = str(auto_dedupe_cfg.get("on_high_risk", "skip_write")).strip().lower()
        self._auto_reflection_dedupe_importance = int(auto_dedupe_cfg.get("downgrade_importance_to", 2))
        self._auto_reflection_dedupe_clear_pin = bool(auto_dedupe_cfg.get("downgrade_clear_pin", True))
        self._auto_reflection_dedupe_needs_review = bool(auto_dedupe_cfg.get("downgrade_mark_needs_review", True))
        self._auto_reflection_dedupe_apply_to_manual_tool = bool(auto_dedupe_cfg.get("apply_to_manual_tool", False))
        semantic_cfg = config.get("memory_semantic") or {}
        self._semantic_config = MemorySemanticConfig(
            enabled=bool(semantic_cfg.get("enabled", False)),
            provider=str(semantic_cfg.get("provider", "none")),
            rerank_enabled=bool(semantic_cfg.get("rerank_enabled", False)),
            max_candidates_for_semantic=int(semantic_cfg.get("max_candidates_for_semantic", 64)),
            min_similarity=float(semantic_cfg.get("min_similarity", 0.25)),
            rerank_influence_min_cosine=float(semantic_cfg.get("rerank_influence_min_cosine", 0.25)),
            dedupe_strong_match_cosine=(
                float(semantic_cfg.get("dedupe_strong_match_cosine"))
                if semantic_cfg.get("dedupe_strong_match_cosine") is not None
                else None
            ),
            background_embedding_enabled=bool(semantic_cfg.get("background_embedding_enabled", True)),
            write_timeout_ms=int(semantic_cfg.get("write_timeout_ms", 75)),
            query_timeout_ms=int(semantic_cfg.get("query_timeout_ms", 40)),
            max_writes_per_minute=int(semantic_cfg.get("max_writes_per_minute", 120)),
            max_queries_per_minute=int(semantic_cfg.get("max_queries_per_minute", 240)),
        )
        self._store = MemoryStore()
        self._embedding_worker: MemoryEmbeddingWorker | None = None
        self._embedding_provider: EmbeddingProvider = build_embedding_provider(config)
        if self._semantic_config.enabled and self._semantic_config.background_embedding_enabled:
            self._embedding_worker = MemoryEmbeddingWorker(store=self._store)
        self._last_turn_retrieval_at: dict[tuple[str, MemoryScope], float] = {}
        self._last_turn_retrieval_debug: dict[str, object] = {}
        self._retrieval_total_count = 0
        self._retrieval_total_latency_ms = 0.0
        self._retrieval_semantic_attempt_count = 0
        self._retrieval_semantic_error_count = 0
        self._embedding_coverage_cache_ttl_s = 60.0
        self._embedding_coverage_cache_at = 0.0
        self._embedding_coverage_cache: dict[str, float | int] = {
            "total_memories": 0,
            "ready_embeddings": 0,
            "coverage_pct": 0.0,
        }
        MemoryManager._instance = self

    @classmethod
    def get_instance(cls) -> "MemoryManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_active_user_id(self, user_id: str) -> None:
        self._active_user_id = user_id

    def set_active_session_id(self, session_id: str | None) -> None:
        self._active_session_id = session_id

    def get_active_user_id(self) -> str:
        return self._active_user_id

    def get_active_session_id(self) -> str | None:
        return self._active_session_id

    def get_semantic_config(self) -> MemorySemanticConfig:
        return self._semantic_config

    def get_embedding_worker(self) -> MemoryEmbeddingWorker | None:
        return self._embedding_worker

    def get_last_turn_retrieval_debug_metadata(self) -> dict[str, object]:
        """Return internal retrieval metadata for debugging and audit logs."""

        return dict(self._last_turn_retrieval_debug)

    def get_retrieval_health_metrics(self) -> dict[str, float | int]:
        """Return retrieval counters for diagnostics and health summaries."""

        total = max(0, int(self._retrieval_total_count))
        avg_latency_ms = (
            float(self._retrieval_total_latency_ms) / total
            if total > 0
            else 0.0
        )
        semantic_attempts = max(0, int(self._retrieval_semantic_attempt_count))
        semantic_errors = max(0, int(self._retrieval_semantic_error_count))
        semantic_error_rate_pct = (
            (semantic_errors / semantic_attempts) * 100.0
            if semantic_attempts > 0
            else 0.0
        )

        now = time.monotonic()
        if now - self._embedding_coverage_cache_at >= self._embedding_coverage_cache_ttl_s:
            total_memories, ready_embeddings = self._store.get_embedding_coverage_counts(
                user_id=self._active_user_id,
                scope=self._default_scope.value,
                session_id=self._active_session_id,
            )
            coverage_pct = (float(ready_embeddings) / float(total_memories) * 100.0) if total_memories else 0.0
            self._embedding_coverage_cache = {
                "total_memories": total_memories,
                "ready_embeddings": ready_embeddings,
                "coverage_pct": round(coverage_pct, 2),
            }
            self._embedding_coverage_cache_at = now

        return {
            "retrieval_count": total,
            "average_retrieval_latency_ms": round(avg_latency_ms, 2),
            "semantic_provider_attempts": semantic_attempts,
            "semantic_provider_errors": semantic_errors,
            "semantic_provider_error_rate_pct": round(semantic_error_rate_pct, 2),
            "embedding_total_memories": int(self._embedding_coverage_cache["total_memories"]),
            "embedding_ready_memories": int(self._embedding_coverage_cache["ready_embeddings"]),
            "embedding_coverage_pct": float(self._embedding_coverage_cache["coverage_pct"]),
        }

    def remember_memory(
        self,
        *,
        content: str,
        tags: list[str] | None = None,
        importance: int = 3,
        user_id: str | None = None,
        source: str = "manual_tool",
        session_id: str | None = None,
        scope: MemoryScopeInput | None = None,
        pinned: bool = False,
        needs_review: bool = False,
    ) -> MemoryEntry:
        normalized_content = _normalize_content(content)
        normalized_tags = _normalize_tags(tags)
        bounded_importance = _clamp(importance, MIN_IMPORTANCE, MAX_IMPORTANCE)
        resolved_scope = _normalize_scope(scope, fallback=self._default_scope)
        resolved_session_id = self._resolve_session_id_for_scope(
            scope=resolved_scope,
            session_id=session_id,
        )

        auto_pin = source == "auto_reflection" and bounded_importance >= self._auto_pin_min_importance
        effective_pinned = bool(pinned or auto_pin)
        effective_review = bool(needs_review or (auto_pin and self._auto_pin_requires_review))

        duplicate_match = self._find_semantic_duplicate(
            content=normalized_content,
            user_id=user_id if user_id is not None else self._active_user_id,
            scope=resolved_scope,
            session_id=resolved_session_id,
            source=source,
        )
        if duplicate_match is not None:
            if self._auto_reflection_dedupe_policy == "downgrade":
                bounded_importance = _clamp(
                    self._auto_reflection_dedupe_importance,
                    MIN_IMPORTANCE,
                    MAX_IMPORTANCE,
                )
                if self._auto_reflection_dedupe_clear_pin:
                    effective_pinned = False
                if self._auto_reflection_dedupe_needs_review:
                    effective_review = True
            else:
                return duplicate_match

        entry = self._store.append_memory(
            content=normalized_content,
            tags=normalized_tags,
            importance=bounded_importance,
            user_id=user_id if user_id is not None else self._active_user_id,
            session_id=resolved_session_id,
            source=source,
            pinned=effective_pinned,
            needs_review=effective_review,
        )
        if self._embedding_worker is not None:
            try:
                self._embedding_worker.enqueue_memory(memory_id=entry.memory_id)
            except Exception:  # noqa: BLE001
                # Embedding scheduling is best-effort and must never block writes.
                pass
        return entry

    def _find_semantic_duplicate(
        self,
        *,
        content: str,
        user_id: str,
        scope: MemoryScope,
        session_id: str | None,
        source: str,
    ) -> MemoryEntry | None:
        if not self._semantic_config.enabled:
            return None
        if source == "manual_tool" and not self._auto_reflection_dedupe_apply_to_manual_tool:
            return None
        if source != "auto_reflection" and source != "manual_tool":
            return None
        if not self._auto_reflection_semantic_dedupe_enabled:
            return None

        threshold = self._semantic_config.dedupe_strong_match_cosine
        if threshold is None:
            threshold = self._auto_reflection_dedupe_high_risk_cosine
        if threshold <= 0.0:
            return None

        recent_limit = _clamp(self._auto_reflection_dedupe_recent_limit, 1, 128)
        recent_entries = self._store.search_memories(
            query=None,
            limit=recent_limit,
            user_id=user_id,
            scope=scope,
            session_id=session_id,
            review_state="approved",
        )
        if not recent_entries:
            return None

        provider = getattr(self, "_embedding_provider", None)
        if provider is None:
            provider = build_embedding_provider(ConfigController.get_instance().get_config())
        candidate_embedding = provider.embed_text(content)
        if (
            candidate_embedding.status != "ready"
            or candidate_embedding.dimension <= 0
            or not candidate_embedding.vector
            or float(candidate_embedding.vector_norm or 0.0) <= 0.0
        ):
            return None

        embeddings = self._store.fetch_embeddings_for_memories(
            memory_ids=[entry.memory_id for entry in recent_entries]
        )
        strongest: tuple[float, MemoryEntry] | None = None
        for entry in recent_entries:
            embedding = embeddings.get(entry.memory_id)
            if embedding is None or embedding.status != "ready":
                continue
            cosine = _cosine_similarity_bytes(
                query_vector=candidate_embedding.vector,
                query_norm=float(candidate_embedding.vector_norm or 0.0),
                entry_vector=embedding.vector,
                entry_norm=float(embedding.vector_norm or 0.0),
            )
            if cosine is None or cosine < threshold:
                continue
            if strongest is None or cosine > strongest[0]:
                strongest = (cosine, entry)

        if strongest is None:
            return None
        return strongest[1]

    def recall_memories(
        self,
        *,
        query: str | None = None,
        limit: int = 5,
        scope: MemoryScopeInput | None = None,
        session_id: str | None = None,
    ) -> list[MemorySummary]:
        bounded_limit = _clamp(limit, 1, MAX_RECALL_LIMIT)
        resolved_scope = _normalize_scope(scope, fallback=self._default_scope)
        resolved_session_id = self._resolve_session_id_for_scope(
            scope=resolved_scope,
            session_id=session_id,
            strict=False,
        )
        entries = self._store.search_memories(
            query=query,
            limit=bounded_limit,
            user_id=self._active_user_id,
            scope=resolved_scope,
            session_id=resolved_session_id,
        )
        return [MemorySummary.from_entry(entry) for entry in entries]

    def retrieve_startup_digest(
        self,
        *,
        max_items: int = 2,
        max_chars: int = 280,
        user_id: str | None = None,
    ) -> MemoryBrief | None:
        """Return a tiny pinned-memory digest for session initialization."""

        effective_user_id = user_id if user_id is not None else self._active_user_id
        bounded_max_items = _clamp(max_items, 1, 4)
        bounded_max_chars = _clamp(max_chars, 80, 800)
        entries = self._store.search_memories(
            query=None,
            limit=max(bounded_max_items * 5, 5),
            user_id=effective_user_id,
            scope=MemoryScope.USER_GLOBAL.value,
            pinned_only=True,
            review_state="approved",
        )
        if not entries:
            return None

        selected: list[MemorySummary] = []
        used_chars = 0
        truncated = False
        for entry in entries:
            if len(selected) >= bounded_max_items:
                truncated = True
                break
            summary = MemorySummary.from_entry(entry)
            chars = len(summary.content)
            if used_chars + chars > bounded_max_chars:
                truncated = True
                continue
            selected.append(summary)
            used_chars += chars

        if not selected:
            return None
        return MemoryBrief(
            items=selected,
            total_chars=used_chars,
            max_chars=bounded_max_chars,
            truncated=truncated,
            scope=MemoryScope.USER_GLOBAL,
        )

    def forget_memory(self, *, memory_id: int) -> bool:
        return self._store.delete_memory(memory_id=memory_id)

    def retrieve_for_turn(
        self,
        *,
        latest_user_utterance: str,
        user_id: str | None = None,
        max_memories: int = 4,
        max_chars: int = 450,
        cooldown_s: float = 0.0,
        now_monotonic: float | None = None,
        scope: MemoryScopeInput | None = None,
        session_id: str | None = None,
    ) -> MemoryBrief | None:
        """Retrieve a compact, deterministic memory brief for turn-time context."""

        started_at = time.monotonic()

        self._last_turn_retrieval_debug = {
            "mode": "none",
            "semantic_enabled": False,
            "semantic_attempted": False,
            "semantic_applied": False,
            "semantic_error": None,
            "candidate_count": 0,
            "lexical_candidate_count": 0,
            "semantic_candidate_count": 0,
            "semantic_scored_count": 0,
            "selected_count": 0,
            "scope": None,
            "cooldown_skipped": False,
            "fallback_reason": None,
            "latency_ms": 0,
            "dedupe_count": 0,
            "truncation_count": 0,
        }

        clean_utterance = " ".join(latest_user_utterance.strip().split())
        if not clean_utterance:
            return None

        effective_user_id = user_id if user_id is not None else self._active_user_id
        resolved_scope = _normalize_scope(scope, fallback=self._default_scope)
        resolved_session_id = self._resolve_session_id_for_scope(
            scope=resolved_scope,
            session_id=session_id,
            strict=False,
        )

        timestamp = now_monotonic if now_monotonic is not None else time.monotonic()
        cooldown_key = (effective_user_id, resolved_scope)
        self._last_turn_retrieval_debug["scope"] = resolved_scope.value
        if cooldown_s > 0.0:
            last_retrieval = self._last_turn_retrieval_at.get(cooldown_key)
            if last_retrieval is not None and timestamp - last_retrieval < cooldown_s:
                self._last_turn_retrieval_debug.update({"mode": "cooldown", "cooldown_skipped": True})
                return None

        bounded_max_memories = _clamp(max_memories, 1, MAX_RECALL_LIMIT)
        bounded_max_chars = _clamp(max_chars, 80, 4000)
        candidate_limit = max(bounded_max_memories * RANK_CANDIDATE_MULTIPLIER, bounded_max_memories)
        entries = self._store.search_memories(
            query=None,
            limit=candidate_limit,
            user_id=effective_user_id,
            scope=resolved_scope,
            session_id=resolved_session_id,
            review_state="approved",
        )
        if not entries:
            self._last_turn_retrieval_at[cooldown_key] = timestamp
            self._last_turn_retrieval_debug["mode"] = "empty"
            return None

        now_s = time.time()
        utterance_tokens = _tokenize(clean_utterance)
        scored_entries: list[tuple[float, MemoryEntry]] = []
        for entry in entries:
            if _is_stale(entry, now_s=now_s, max_age_s=STALE_MEMORY_MAX_AGE_S):
                continue
            score = _score_entry(entry, utterance_tokens=utterance_tokens, now_s=now_s)
            if score < 0.05:
                continue
            scored_entries.append((score, entry))

        if not scored_entries:
            self._last_turn_retrieval_at[cooldown_key] = timestamp
            self._last_turn_retrieval_debug["mode"] = "filtered_empty"
            return None

        self._last_turn_retrieval_debug["candidate_count"] = len(scored_entries)

        lexical_ordered = sorted(
            scored_entries,
            key=lambda pair: (-pair[0], -pair[1].importance, -pair[1].timestamp, pair[1].memory_id),
        )
        ordered = [item[1] for item in lexical_ordered]

        semantic_enabled = self._semantic_config.enabled and self._semantic_config.rerank_enabled
        self._last_turn_retrieval_debug["semantic_enabled"] = semantic_enabled
        if semantic_enabled and lexical_ordered:
            self._retrieval_semantic_attempt_count += 1
            self._last_turn_retrieval_debug["semantic_attempted"] = True
            semantic_limit = _clamp(
                self._semantic_config.max_candidates_for_semantic,
                1,
                len(lexical_ordered),
            )
            semantic_pool = lexical_ordered[:semantic_limit]
            self._last_turn_retrieval_debug["semantic_candidate_count"] = semantic_limit
            semantic_memory_ids = [entry.memory_id for _, entry in semantic_pool]
            try:
                provider = getattr(self, "_embedding_provider", None)
                if provider is None:
                    provider = build_embedding_provider(ConfigController.get_instance().get_config())
                query_embedding = provider.embed_text(clean_utterance)
                if query_embedding.status == "ready" and query_embedding.dimension > 0 and query_embedding.vector:
                    query_norm = float(query_embedding.vector_norm or 0.0)
                    if query_norm > 0.0:
                        embeddings = self._store.fetch_embeddings_for_memories(memory_ids=semantic_memory_ids)
                        hybrid_pool: list[tuple[float, MemoryEntry]] = []
                        semantic_scored_count = 0
                        for lexical_score, entry in semantic_pool:
                            embedding = embeddings.get(entry.memory_id)
                            semantic_score = 0.0
                            if embedding is not None and embedding.status == "ready":
                                cosine = _cosine_similarity_bytes(
                                    query_vector=query_embedding.vector,
                                    query_norm=query_norm,
                                    entry_vector=embedding.vector,
                                    entry_norm=float(embedding.vector_norm or 0.0),
                                )
                                influence_threshold = max(
                                    self._semantic_config.min_similarity,
                                    self._semantic_config.rerank_influence_min_cosine,
                                )
                                if cosine is not None and cosine >= influence_threshold:
                                    semantic_score = cosine
                                    semantic_scored_count += 1
                            hybrid_pool.append((lexical_score + semantic_score, entry))

                        reranked_pool = sorted(
                            hybrid_pool,
                            key=lambda pair: (-pair[0], -pair[1].importance, -pair[1].timestamp, pair[1].memory_id),
                        )
                        ordered = [entry for _, entry in reranked_pool] + [entry for _, entry in lexical_ordered[semantic_limit:]]
                        self._last_turn_retrieval_debug.update(
                            {
                                "mode": "hybrid",
                                "semantic_applied": True,
                                "semantic_pool_size": semantic_limit,
                                "semantic_scored_count": semantic_scored_count,
                            }
                        )
                else:
                    self._last_turn_retrieval_debug["fallback_reason"] = "query_embedding_not_ready"
            except Exception:  # noqa: BLE001
                # Semantic reranking is best-effort and must never degrade lexical retrieval.
                ordered = [item[1] for item in lexical_ordered]
                self._retrieval_semantic_error_count += 1
                self._last_turn_retrieval_debug.update(
                    {
                        "mode": "hybrid_fallback_lexical",
                        "semantic_error": "exception",
                        "fallback_reason": "semantic_provider_error",
                    }
                )
        elif semantic_enabled:
            self._last_turn_retrieval_debug["fallback_reason"] = "no_lexical_candidates"
        if self._last_turn_retrieval_debug.get("mode") == "none":
            self._last_turn_retrieval_debug["mode"] = "lexical"
            if semantic_enabled and not self._last_turn_retrieval_debug.get("fallback_reason"):
                self._last_turn_retrieval_debug["fallback_reason"] = "semantic_not_applied"
        selected: list[MemorySummary] = []
        seen_contents: list[str] = []
        used_chars = 0
        truncated = False
        dedupe_count = 0
        truncation_count = 0
        for entry in ordered:
            if len(selected) >= bounded_max_memories:
                truncated = True
                truncation_count += 1
                break

            if any(
                _is_near_duplicate(
                    entry.content,
                    existing_content,
                    token_threshold=NEAR_DUPLICATE_THRESHOLD,
                    char_threshold=NEAR_DUPLICATE_CHAR_RATIO,
                )
                for existing_content in seen_contents
            ):
                truncated = True
                dedupe_count += 1
                continue

            candidate_summary = MemorySummary.from_entry(entry)
            candidate_chars = len(candidate_summary.content)
            if used_chars + candidate_chars > bounded_max_chars:
                remaining = bounded_max_chars - used_chars
                if selected or remaining < 24:
                    truncated = True
                    truncation_count += 1
                    continue
                clipped = candidate_summary.content[: max(remaining - 1, 1)].rstrip()
                if clipped != candidate_summary.content:
                    clipped = f"{clipped}…"
                candidate_summary = MemorySummary(
                    memory_id=candidate_summary.memory_id,
                    content=clipped,
                    tags=candidate_summary.tags,
                    importance=candidate_summary.importance,
                    source=candidate_summary.source,
                    pinned=candidate_summary.pinned,
                    needs_review=candidate_summary.needs_review,
                )
                candidate_chars = len(candidate_summary.content)
                truncated = True
                truncation_count += 1

            selected.append(candidate_summary)
            seen_contents.append(candidate_summary.content)
            used_chars += candidate_chars

        self._last_turn_retrieval_at[cooldown_key] = timestamp
        latency_ms = int((time.monotonic() - started_at) * 1000)
        self._retrieval_total_count += 1
        self._retrieval_total_latency_ms += latency_ms
        self._last_turn_retrieval_debug["lexical_candidate_count"] = len(lexical_ordered)
        self._last_turn_retrieval_debug["selected_count"] = len(selected)
        self._last_turn_retrieval_debug["truncated"] = truncated
        self._last_turn_retrieval_debug["dedupe_count"] = dedupe_count
        self._last_turn_retrieval_debug["truncation_count"] = truncation_count
        self._last_turn_retrieval_debug["latency_ms"] = latency_ms
        if not selected:
            return None
        return MemoryBrief(
            items=selected,
            total_chars=used_chars,
            max_chars=bounded_max_chars,
            truncated=truncated,
            scope=resolved_scope,
        )

    def _resolve_session_id_for_scope(
        self,
        *,
        scope: MemoryScope,
        session_id: str | None,
        strict: bool = True,
    ) -> str | None:
        if scope is MemoryScope.USER_GLOBAL:
            return None
        resolved = session_id if session_id is not None else self._active_session_id
        if strict and not resolved:
            raise ValueError("session_local memory scope requires an active session id")
        return resolved
