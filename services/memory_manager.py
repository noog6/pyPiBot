"""Manage memory entries for long-term recall."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
import hashlib
import inspect
import logging
import math
import os
import argparse
import re
import struct
import sys
import threading
import time
from types import SimpleNamespace
from typing import Callable

from config import ConfigController
from services.embedding_provider import EmbeddingProvider, NoopEmbeddingProvider, build_embedding_provider
from services.memory_embedding_worker import MemoryEmbeddingWorker
from storage.factories import create_memory_store
from storage.memories import MemoryEntry, MemoryStore

MAX_CONTENT_LENGTH = 400
MAX_TAGS = 6
MAX_TAG_LENGTH = 24
MAX_RECALL_LIMIT = 10
MIN_IMPORTANCE = 1
MAX_IMPORTANCE = 5
RANK_CANDIDATE_MULTIPLIER = 8
MAX_RETRIEVAL_CANDIDATE_CAP = MAX_RECALL_LIMIT * RANK_CANDIDATE_MULTIPLIER
STALE_MEMORY_MAX_AGE_S = 60.0 * 60.0 * 24.0 * 365.0
NEAR_DUPLICATE_THRESHOLD = 0.75
NEAR_DUPLICATE_CHAR_RATIO = 0.9
WORD_RE = re.compile(r"[a-zA-Z0-9]{2,}")
logger = logging.getLogger(__name__)

SEMANTIC_QUERY_TIMEOUT_FLOOR_MS = 100
SEMANTIC_CANARY_REFRESH_INTERVAL_S = 60.0 * 5.0
LATENCY_SAMPLE_WINDOW_SIZE = 64
LATENCY_BUCKET_UPPER_BOUNDS_MS: tuple[int, ...] = (25, 50, 100, 250, 500, 1000)


class _LatencyWindowSampler:
    """Track bounded latency samples and summarize percentiles and buckets."""

    def __init__(self, *, max_samples: int = LATENCY_SAMPLE_WINDOW_SIZE) -> None:
        self._samples_ms: deque[int] = deque(maxlen=max(1, int(max_samples)))

    def add_sample(self, latency_ms: int) -> None:
        self._samples_ms.append(max(0, int(latency_ms)))

    def summary(self) -> dict[str, int]:
        if not self._samples_ms:
            return {
                "samples": 0,
                "p50_ms": 0,
                "p90_ms": 0,
                "p99_ms": 0,
                "bucket_le_25ms": 0,
                "bucket_le_50ms": 0,
                "bucket_le_100ms": 0,
                "bucket_le_250ms": 0,
                "bucket_le_500ms": 0,
                "bucket_le_1000ms": 0,
                "bucket_gt_1000ms": 0,
            }

        values = sorted(self._samples_ms)
        sample_count = len(values)

        def _nearest_rank_percentile(percentile: float) -> int:
            rank = max(1, math.ceil(percentile * sample_count))
            return values[min(sample_count - 1, rank - 1)]

        buckets: dict[str, int] = {
            "bucket_le_25ms": 0,
            "bucket_le_50ms": 0,
            "bucket_le_100ms": 0,
            "bucket_le_250ms": 0,
            "bucket_le_500ms": 0,
            "bucket_le_1000ms": 0,
            "bucket_gt_1000ms": 0,
        }
        for value in values:
            if value <= LATENCY_BUCKET_UPPER_BOUNDS_MS[0]:
                buckets["bucket_le_25ms"] += 1
            elif value <= LATENCY_BUCKET_UPPER_BOUNDS_MS[1]:
                buckets["bucket_le_50ms"] += 1
            elif value <= LATENCY_BUCKET_UPPER_BOUNDS_MS[2]:
                buckets["bucket_le_100ms"] += 1
            elif value <= LATENCY_BUCKET_UPPER_BOUNDS_MS[3]:
                buckets["bucket_le_250ms"] += 1
            elif value <= LATENCY_BUCKET_UPPER_BOUNDS_MS[4]:
                buckets["bucket_le_500ms"] += 1
            elif value <= LATENCY_BUCKET_UPPER_BOUNDS_MS[5]:
                buckets["bucket_le_1000ms"] += 1
            else:
                buckets["bucket_gt_1000ms"] += 1

        return {
            "samples": sample_count,
            "p50_ms": _nearest_rank_percentile(0.50),
            "p90_ms": _nearest_rank_percentile(0.90),
            "p99_ms": _nearest_rank_percentile(0.99),
            **buckets,
        }


def _safe_text_fingerprint(text: str) -> dict[str, object]:
    normalized = " ".join(text.strip().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return {
        "hash": digest,
        "length": len(normalized),
    }


def _sanitize_error_class_name(value: object) -> str | None:
    if value is None:
        return None
    class_name = re.sub(r"[^a-zA-Z0-9_]", "", str(value).strip())
    if not class_name:
        return None
    return class_name[:64]


def _normalize_semantic_failure_class(*, error_code: object, error_class: object) -> str | None:
    code = str(error_code or "").strip().lower()
    klass = str(error_class or "").strip().lower()
    haystack = f"{code} {klass}"

    if "rate" in haystack and "limit" in haystack:
        return "rate_limited"
    if code == "rate_limited":
        return "rate_limited"
    if "timeout" in haystack:
        return "timeout"
    if "auth" in haystack or "forbidden" in haystack or "unauthorized" in haystack:
        return "auth"
    if "model" in haystack:
        return "model"
    if (
        "connect" in haystack
        or "network" in haystack
        or "dns" in haystack
        or "socket" in haystack
        or "tls" in haystack
    ):
        return "connection"
    return None


def _normalize_canary_error_code(raw_error_code: str | None, *, exc: Exception | None = None) -> str:
    code = str(raw_error_code or "").strip().lower()
    if code in {"timeout", "timeout_backoff", "request_timeout"}:
        return "timeout"
    if code in {"missing_api_key", "auth_forbidden", "unauthorized", "forbidden", "invalid_api_key", "auth"}:
        return "auth"
    if code in {"model_not_found", "unknown_model", "invalid_model"}:
        return "model"
    if code in {"provider_http_error", "http_error"}:
        return "http"
    if code in {"not_found", "provider_not_found"}:
        return "not_found"
    if code in {
        "connection_refused",
        "connection_error",
        "network_error",
        "dns_error",
        "connection",
        "provider_request_failed",
    }:
        return "connection"

    exc_name = type(exc).__name__ if exc is not None else ""
    exc_message = str(exc) if exc is not None else ""
    raw = f"{code} {exc_name} {exc_message}".lower()
    if "timeout" in raw:
        return "timeout"
    if any(token in raw for token in ["auth", "unauthor", "forbidden", "api key"]):
        return "auth"
    if "model" in raw:
        return "model"
    if "not found" in raw:
        return "not_found"
    if any(token in raw for token in ["connection", "network", "dns", "refused"]):
        return "connection"
    return code if code else "unknown"


def _classify_canary_failure_code(code: str | None) -> str:
    normalized_code = str(code or "").strip().lower()
    if normalized_code in {"timeout", "auth", "model", "http", "connection", "not_found"}:
        return f"canary_failed:{normalized_code}"
    if normalized_code in {"", "unknown"}:
        return "canary_failed:unknown"
    return "canary_failed:other"


def _invoke_embed_text(
    provider: object,
    *,
    text: str,
    timeout_s: float,
    timeout_ms: int,
    operation: str,
):
    embed_text = getattr(provider, "embed_text")
    try:
        signature = inspect.signature(embed_text)
    except (TypeError, ValueError):
        signature = None

    kwargs: dict[str, object] = {}
    if signature is not None:
        params = signature.parameters
        accepts_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())
        if accepts_kwargs or "timeout_s" in params:
            kwargs["timeout_s"] = timeout_s
        if accepts_kwargs or "timeout_ms" in params:
            kwargs["timeout_ms"] = timeout_ms
        if accepts_kwargs or "operation" in params:
            kwargs["operation"] = operation
    if kwargs:
        return embed_text(text, **kwargs)
    return embed_text(text)


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


def render_realtime_memory_brief_item(*, index: int, item: MemorySummary) -> str:
    """Render a single turn-memory line exactly as RealtimeAPI injects it."""

    tags = f" tags=[{', '.join(item.tags)}]" if item.tags else ""
    return f"{index}. (importance={item.importance}{tags}) {item.content}"


def estimate_realtime_memory_brief_item_chars(*, index: int, item: MemorySummary) -> int:
    """Estimate injected chars for one memory item in the turn-memory brief."""

    return len(render_realtime_memory_brief_item(index=index, item=item))


def render_startup_memory_digest_item(*, index: int, item: MemorySummary) -> str:
    """Render a single startup-digest line exactly as RealtimeAPI injects it."""

    tags = f" tags=[{', '.join(item.tags)}]" if item.tags else ""
    pin_state = " pinned" if item.pinned else ""
    return f"{index}. (importance={item.importance}{pin_state}{tags}) {item.content}"


def estimate_startup_memory_digest_item_chars(*, index: int, item: MemorySummary) -> int:
    """Estimate injected chars for one memory item in the startup digest."""

    return len(render_startup_memory_digest_item(index=index, item=item))


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
    provider_model: str
    provider_timeout_s: float
    rerank_enabled: bool
    max_candidates_for_semantic: int
    min_similarity: float
    rerank_influence_min_cosine: float
    dedupe_strong_match_cosine: float | None
    background_embedding_enabled: bool
    inline_embedding_on_write_when_background_disabled: bool
    rolling_backfill_batch_size: int
    rolling_backfill_interval_idle_cycles: int
    write_timeout_ms: int
    query_timeout_ms: int
    startup_canary_timeout_ms: int
    startup_canary_bypass: bool
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
        openai_cfg = semantic_cfg.get("openai") or {}
        self._semantic_config = MemorySemanticConfig(
            enabled=bool(semantic_cfg.get("enabled", False)),
            provider=str(semantic_cfg.get("provider", "none")),
            provider_model=str(openai_cfg.get("model", "text-embedding-3-small")),
            provider_timeout_s=float(openai_cfg.get("timeout_s", 10.0)),
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
            inline_embedding_on_write_when_background_disabled=bool(
                semantic_cfg.get("inline_embedding_on_write_when_background_disabled", False)
            ),
            rolling_backfill_batch_size=max(1, int(semantic_cfg.get("rolling_backfill_batch_size", 4))),
            rolling_backfill_interval_idle_cycles=max(1, int(semantic_cfg.get("rolling_backfill_interval_idle_cycles", 15))),
            write_timeout_ms=int(semantic_cfg.get("write_timeout_ms", 75)),
            query_timeout_ms=int(semantic_cfg.get("query_timeout_ms", 2000)),
            startup_canary_timeout_ms=int(semantic_cfg.get("startup_canary_timeout_ms", 120)),
            startup_canary_bypass=bool(semantic_cfg.get("startup_canary_bypass", False)),
            max_writes_per_minute=int(semantic_cfg.get("max_writes_per_minute", 120)),
            max_queries_per_minute=int(semantic_cfg.get("max_queries_per_minute", 240)),
        )
        self._semantic_write_call_timestamps: list[float] = []
        self._semantic_query_call_timestamps: list[float] = []
        self._store = create_memory_store()
        self._embedding_worker: MemoryEmbeddingWorker | None = None
        self._embedding_executor: ThreadPoolExecutor | None = None
        self._embedding_executor_lock = threading.Lock()
        self._embedding_provider: EmbeddingProvider = build_embedding_provider(config)
        self._semantic_provider_enabled = {
            "openai": bool(openai_cfg.get("enabled", False)),
        }
        if self._semantic_config.enabled and self._semantic_config.background_embedding_enabled:
            self._embedding_worker = MemoryEmbeddingWorker(
                store=self._store,
                rolling_backfill_batch_size=self._semantic_config.rolling_backfill_batch_size,
                rolling_backfill_interval_idle_cycles=self._semantic_config.rolling_backfill_interval_idle_cycles,
            )
        self._last_turn_retrieval_at: dict[tuple[str, MemoryScope, str | None], float] = {}
        self._last_turn_retrieval_debug: dict[str, object] = {}
        self._last_semantic_dedupe_debug: dict[str, object] = {}
        self._retrieval_total_count = 0
        self._retrieval_total_latency_ms = 0.0
        self._retrieval_semantic_attempt_count = 0
        self._retrieval_semantic_error_count = 0
        self._semantic_timeout_count = 0
        self._semantic_timeout_consecutive_count = 0
        self._semantic_timeout_backoff_until_monotonic = 0.0
        self._semantic_timeout_backoff_activation_count = 0
        self._semantic_timeout_backoff_threshold = 3
        self._semantic_timeout_backoff_window_s = 5.0
        self._semantic_provider_last_error_code = "none"
        self._semantic_provider_ready_last = False
        self._semantic_canary_bypass = bool(
            self._semantic_config.startup_canary_bypass
            or semantic_cfg.get("test_mode_bypass_canary", False)
            or semantic_cfg.get("offline_mode", False)
            or os.getenv("PYPIBOT_SEMANTIC_CANARY_BYPASS", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        self._semantic_canary_last: dict[str, bool | int | float | str | None] = {
            "canary_success": False,
            "latency_ms": None,
            "dimension": None,
            "error_code": "not_run",
        }
        self._semantic_canary_last_checked_monotonic = 0.0
        self._semantic_readiness_reason_last = "not_run"
        self._semantic_readiness_last_transition_monotonic = time.monotonic()
        self._semantic_readiness_transition_count = 0
        self._semantic_canary_refresh_interval_s = max(
            30.0,
            float(semantic_cfg.get("canary_refresh_interval_s", SEMANTIC_CANARY_REFRESH_INTERVAL_S)),
        )
        self._query_embedding_latency_samples = _LatencyWindowSampler()
        self._canary_refresh_latency_samples = _LatencyWindowSampler()
        self._run_embedding_canary()
        _, self._semantic_readiness_reason_last = self._is_semantic_provider_ready()
        self._semantic_query_timeout_floor_warned = False
        self._semantic_query_embedding_not_ready_streak = 0
        self._embedding_coverage_cache_ttl_s = 60.0
        self._embedding_coverage_cache: dict[tuple[str, MemoryScope, str | None], dict[str, float | int]] = {}
        self._embedding_backlog_last: dict[tuple[str, MemoryScope, str | None], int] = {}
        semantic_active, semantic_reason = self._semantic_rerank_readiness()
        logger.info("semantic_rerank_active=%s reason=%s", semantic_active, semantic_reason)
        MemoryManager._instance = self

    def _get_embedding_executor(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_embedding_executor", None)
        if executor is not None:
            return executor

        lock = getattr(self, "_embedding_executor_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._embedding_executor_lock = lock

        with lock:
            executor = getattr(self, "_embedding_executor", None)
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=1)
                self._embedding_executor = executor
        return executor

    def _shutdown_embedding_executor(self) -> None:
        executor = getattr(self, "_embedding_executor", None)
        if executor is None:
            return
        self._embedding_executor = None
        executor.shutdown(wait=False, cancel_futures=True)

    def close(self) -> None:
        """Release background resources owned by the manager."""

        self._shutdown_embedding_executor()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _is_semantic_provider_ready(self) -> tuple[bool, str]:
        raw_provider = getattr(self._semantic_config, "provider", None)
        provider_name = str(raw_provider).strip().lower() if raw_provider is not None else ""
        if provider_name in {"none", "noop"}:
            return False, "provider_not_configured"

        provider_enabled = None
        if provider_name:
            provider_enabled = getattr(self, "_semantic_provider_enabled", {}).get(provider_name)
            if provider_enabled is False:
                return False, "provider_disabled"

        provider = getattr(self, "_embedding_provider", None)
        if provider is None:
            return False, "provider_missing"
        if isinstance(provider, NoopEmbeddingProvider):
            return False, "provider_unavailable"
        if not hasattr(provider, "embed_text"):
            return False, "provider_invalid"
        if bool(getattr(self, "_semantic_canary_bypass", False)):
            return True, "canary_bypassed"

        canary_state = getattr(self, "_semantic_canary_last", {}) or {}
        if not canary_state:
            return False, "canary_not_run"

        if bool(canary_state.get("canary_success", False)):
            return True, "canary_ready"

        raw_error_code = canary_state.get("error_code")
        canary_error = str(raw_error_code or "").strip().lower()
        if canary_error == "not_run" or (not canary_error and not bool(canary_state.get("canary_success", False))):
            return False, "canary_not_run"
        return False, _classify_canary_failure_code(canary_error)

    def _map_canary_error_code(self, raw_error_code: str | None, *, exc: Exception | None = None) -> str:
        return _normalize_canary_error_code(raw_error_code, exc=exc)

    def _run_embedding_canary(self) -> dict[str, bool | int | float | str | None]:
        canary_state: dict[str, bool | int | float | str | None] = {
            "canary_success": False,
            "latency_ms": 0,
            "dimension": None,
            "error_code": "skipped",
        }
        if not bool(getattr(self._semantic_config, "enabled", False)):
            canary_state["error_code"] = "disabled"
        elif bool(getattr(self, "_semantic_canary_bypass", False)):
            canary_state["canary_success"] = True
            canary_state["error_code"] = "bypassed"
        else:
            started = time.perf_counter()
            try:
                result = self._embed_text_with_semantic_policy(
                    text="ping",
                    operation="query",
                    enforce_budget=False,
                    timeout_override_ms=max(1, int(getattr(self._semantic_config, "startup_canary_timeout_ms", 120))),
                    suppress_canary_refresh=True,
                )
                latency_ms = int((time.perf_counter() - started) * 1000.0)
                canary_state["latency_ms"] = latency_ms
                dimension = int(getattr(result, "dimension", 0) or 0)
                canary_state["dimension"] = dimension if dimension > 0 else None

                status = str(getattr(result, "status", "") or "")
                vector = getattr(result, "vector", b"")
                if status == "ready" and bool(vector):
                    canary_state["canary_success"] = True
                    canary_state["error_code"] = "none"
                else:
                    raw_error = str(getattr(result, "error_code", "") or status or "unknown")
                    normalized_error = self._map_canary_error_code(raw_error)
                    canary_state["error_code"] = normalized_error
                    if normalized_error != raw_error or normalized_error == "unknown":
                        canary_state["raw_error_code"] = raw_error
            except Exception as exc:  # noqa: BLE001
                latency_ms = int((time.perf_counter() - started) * 1000.0)
                canary_state["latency_ms"] = latency_ms
                normalized_error = self._map_canary_error_code(None, exc=exc)
                canary_state["error_code"] = normalized_error
                if normalized_error == "unknown":
                    canary_state["raw_error_code"] = ""

        self._semantic_canary_last = canary_state
        self._semantic_canary_last_checked_monotonic = time.monotonic()
        self._canary_refresh_latency_samples.add_sample(int(canary_state.get("latency_ms", 0) or 0))
        return dict(canary_state)

    def _maybe_refresh_canary(self, *, reason: str) -> bool:
        now_monotonic = time.monotonic()
        last_checked = float(getattr(self, "_semantic_canary_last_checked_monotonic", 0.0) or 0.0)
        refresh_interval_s = max(30.0, float(getattr(self, "_semantic_canary_refresh_interval_s", 300.0)))
        is_stale = (now_monotonic - last_checked) >= refresh_interval_s
        timeout_streak = max(0, int(getattr(self, "_semantic_timeout_consecutive_count", 0)))
        timeout_threshold = max(1, int(getattr(self, "_semantic_timeout_backoff_threshold", 3)))
        timeout_triggered = reason == "runtime_timeout_streak" and timeout_streak >= timeout_threshold

        if not is_stale and not timeout_triggered:
            return False

        before_ready, _ = self._is_semantic_provider_ready()
        self._run_embedding_canary()
        after_ready, after_reason = self._is_semantic_provider_ready()
        previous_reason = str(getattr(self, "_semantic_readiness_reason_last", "not_run") or "not_run")
        reason_changed = after_reason != previous_reason
        readiness_changed = before_ready != after_ready
        if readiness_changed or reason_changed:
            self._semantic_readiness_last_transition_monotonic = time.monotonic()
            self._semantic_readiness_transition_count = (
                max(0, int(getattr(self, "_semantic_readiness_transition_count", 0))) + 1
            )
            logger.info(
                "semantic_readiness_transition event=%s previous_ready=%s ready=%s previous_reason=%s reason=%s",
                reason,
                before_ready,
                after_ready,
                previous_reason,
                after_reason,
            )
        self._semantic_readiness_reason_last = after_reason
        return True

    def _semantic_rerank_readiness(self) -> tuple[bool, str]:
        if not getattr(self._semantic_config, "enabled", False):
            return False, "top_level_disabled"
        if not getattr(self._semantic_config, "rerank_enabled", False):
            return False, "rerank_disabled"
        return self._is_semantic_provider_ready()

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

    def run_periodic_maintenance(self, *, optimize_allowed: bool = True) -> dict[str, int | bool]:
        """Run best-effort memory retention maintenance and return telemetry counters."""

        pruned = self._store.prune_memories_by_retention_policy()
        purged = self._store.purge_orphan_embeddings()
        optimize_triggered = False
        if optimize_allowed:
            optimize_triggered = self._store.maybe_optimize_storage(
                deleted_rows=pruned + purged,
                force=False,
            )
        return {
            "pruned_rows": int(pruned),
            "purged_rows": int(purged),
            "optimize_triggered": bool(optimize_triggered),
        }

    def get_last_turn_retrieval_debug_metadata(self) -> dict[str, object]:
        """Return internal retrieval metadata for debugging and audit logs."""

        return dict(self._last_turn_retrieval_debug)

    def get_last_semantic_dedupe_debug_metadata(self) -> dict[str, object]:
        """Return semantic dedupe metadata for diagnostics."""

        return dict(self._last_semantic_dedupe_debug)

    def get_retrieval_health_metrics(
        self,
        *,
        scope: MemoryScopeInput | None = None,
        session_id: str | None = None,
    ) -> dict[str, float | int]:
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

        normalized_scope = _normalize_scope(scope, fallback=self._default_scope)
        resolved_session_id = self._active_session_id if session_id is None else session_id
        cache_key = (self._active_user_id, normalized_scope, resolved_session_id)

        now = time.monotonic()
        cached_coverage = self._embedding_coverage_cache.get(cache_key)
        cache_stale = (
            cached_coverage is None
            or (now - float(cached_coverage.get("cached_at", 0.0))) >= self._embedding_coverage_cache_ttl_s
        )
        if cache_stale:
            total_memories, ready_embeddings = self._store.get_embedding_coverage_counts(
                user_id=self._active_user_id,
                scope=normalized_scope.value,
                session_id=resolved_session_id,
            )
            coverage_pct = (float(ready_embeddings) / float(total_memories) * 100.0) if total_memories else 0.0
            cached_coverage = {
                "total_memories": total_memories,
                "ready_embeddings": ready_embeddings,
                "coverage_pct": round(coverage_pct, 2),
                "cached_at": now,
            }
            self._embedding_coverage_cache[cache_key] = cached_coverage

        assert cached_coverage is not None

        pending_embeddings, missing_embeddings, error_embeddings = self._store.get_embedding_backlog_counts(
            user_id=self._active_user_id,
            scope=normalized_scope.value,
            session_id=resolved_session_id,
        )
        backlog_total_legacy = int(pending_embeddings + missing_embeddings)
        backlog_total_with_errors = int(backlog_total_legacy + error_embeddings)
        prior_backlog_total = self._embedding_backlog_last.get(cache_key)
        backlog_delta = (
            0
            if prior_backlog_total is None
            else int(backlog_total_with_errors - prior_backlog_total)
        )
        self._embedding_backlog_last[cache_key] = backlog_total_with_errors

        semantic_backoff_remaining_ms = max(
            0,
            int(
                (
                    float(getattr(self, "_semantic_timeout_backoff_until_monotonic", 0.0))
                    - time.monotonic()
                )
                * 1000.0
            ),
        )

        query_latency = self._query_embedding_latency_samples.summary()
        canary_latency = self._canary_refresh_latency_samples.summary()

        metrics: dict[str, float | int] = {
            "retrieval_count": total,
            "average_retrieval_latency_ms": round(avg_latency_ms, 2),
            "semantic_provider_attempts": semantic_attempts,
            "semantic_provider_errors": semantic_errors,
            "semantic_provider_error_rate_pct": round(semantic_error_rate_pct, 2),
            "semantic_timeout_count": int(getattr(self, "_semantic_timeout_count", 0)),
            "semantic_timeout_consecutive_count": int(getattr(self, "_semantic_timeout_consecutive_count", 0)),
            "semantic_timeout_backoff_activation_count": int(
                getattr(self, "_semantic_timeout_backoff_activation_count", 0)
            ),
            "semantic_timeout_backoff_active": int(semantic_backoff_remaining_ms > 0),
            "semantic_timeout_backoff_remaining_ms": semantic_backoff_remaining_ms,
            "embedding_total_memories": int(cached_coverage["total_memories"]),
            "embedding_ready_memories": int(cached_coverage["ready_embeddings"]),
            "embedding_coverage_pct": float(cached_coverage["coverage_pct"]),
            "embedding_pending_memories": int(pending_embeddings),
            "embedding_missing_memories": int(missing_embeddings),
            "embedding_error_memories": int(error_embeddings),
            "embedding_backlog_memories": backlog_total_legacy,
            "embedding_backlog_memories_with_errors": backlog_total_with_errors,
            "embedding_backlog_delta_since_last": backlog_delta,
            "query_embedding_latency_samples": int(query_latency["samples"]),
            "query_embedding_latency_p50_ms": int(query_latency["p50_ms"]),
            "query_embedding_latency_p90_ms": int(query_latency["p90_ms"]),
            "query_embedding_latency_p99_ms": int(query_latency["p99_ms"]),
            "query_embedding_latency_bucket_le_25ms": int(query_latency["bucket_le_25ms"]),
            "query_embedding_latency_bucket_le_50ms": int(query_latency["bucket_le_50ms"]),
            "query_embedding_latency_bucket_le_100ms": int(query_latency["bucket_le_100ms"]),
            "query_embedding_latency_bucket_le_250ms": int(query_latency["bucket_le_250ms"]),
            "query_embedding_latency_bucket_le_500ms": int(query_latency["bucket_le_500ms"]),
            "query_embedding_latency_bucket_le_1000ms": int(query_latency["bucket_le_1000ms"]),
            "query_embedding_latency_bucket_gt_1000ms": int(query_latency["bucket_gt_1000ms"]),
            "canary_refresh_latency_samples": int(canary_latency["samples"]),
            "canary_refresh_latency_p50_ms": int(canary_latency["p50_ms"]),
            "canary_refresh_latency_p90_ms": int(canary_latency["p90_ms"]),
            "canary_refresh_latency_p99_ms": int(canary_latency["p99_ms"]),
            "canary_refresh_latency_bucket_le_25ms": int(canary_latency["bucket_le_25ms"]),
            "canary_refresh_latency_bucket_le_50ms": int(canary_latency["bucket_le_50ms"]),
            "canary_refresh_latency_bucket_le_100ms": int(canary_latency["bucket_le_100ms"]),
            "canary_refresh_latency_bucket_le_250ms": int(canary_latency["bucket_le_250ms"]),
            "canary_refresh_latency_bucket_le_500ms": int(canary_latency["bucket_le_500ms"]),
            "canary_refresh_latency_bucket_le_1000ms": int(canary_latency["bucket_le_1000ms"]),
            "canary_refresh_latency_bucket_gt_1000ms": int(canary_latency["bucket_gt_1000ms"]),
        }
        if self._embedding_worker is not None:
            try:
                metrics.update(self._embedding_worker.get_metrics())
            except Exception as exc:  # noqa: BLE001
                pass
        return metrics

    def get_semantic_startup_summary(self) -> dict[str, str | bool | int | float]:
        """Return semantic memory startup diagnostics for one-line logging."""

        provider_ready, readiness_reason = self._is_semantic_provider_ready()
        canary_state = getattr(self, "_semantic_canary_last", {}) or {}
        provider_timeout_ms = int(max(1.0, float(getattr(self._semantic_config, "provider_timeout_s", 0.0)) * 1000.0))
        effective_timeout_budget_ms = min(
            int(getattr(self._semantic_config, "startup_canary_timeout_ms", 0)),
            max(1, provider_timeout_ms - 1),
        )
        return {
            "enabled": bool(self._semantic_config.enabled),
            "provider": str(self._semantic_config.provider),
            "provider_model": str(getattr(self._semantic_config, "provider_model", "")),
            "provider_timeout_s": float(getattr(self._semantic_config, "provider_timeout_s", 0.0)),
            "query_timeout_ms": int(self._semantic_config.query_timeout_ms),
            "write_timeout_ms": int(self._semantic_config.write_timeout_ms),
            "rerank_enabled": bool(self._semantic_config.rerank_enabled),
            "background_embedding_enabled": bool(self._semantic_config.background_embedding_enabled),
            "provider_ready": bool(provider_ready),
            "provider_readiness_reason": str(readiness_reason),
            "canary_success": bool(canary_state.get("canary_success", False)),
            "canary_latency_ms": int(canary_state.get("latency_ms", 0) or 0),
            "canary_dimension": int(canary_state.get("dimension", 0) or 0),
            "canary_error_code": str(canary_state.get("error_code", "not_run") or "not_run"),
            "startup_canary_timeout_ms": int(getattr(self._semantic_config, "startup_canary_timeout_ms", 0)),
            "effective_timeout_budget_ms": int(effective_timeout_budget_ms),
            "startup_canary_bypass": bool(getattr(self, "_semantic_canary_bypass", False)),
            "max_queries_per_minute": int(self._semantic_config.max_queries_per_minute),
            "max_writes_per_minute": int(self._semantic_config.max_writes_per_minute),
        }

    def get_semantic_runtime_health(self) -> dict[str, bool | int | str]:
        """Return runtime semantic-readiness telemetry for operations and auditing."""

        self._maybe_refresh_canary(reason="periodic")
        provider_ready, readiness_reason = self._is_semantic_provider_ready()
        last_checked = float(getattr(self, "_semantic_canary_last_checked_monotonic", 0.0) or 0.0)
        canary_age_ms = int(max(0.0, (time.monotonic() - last_checked) * 1000.0)) if last_checked > 0.0 else -1
        readiness_last_transition = float(
            getattr(self, "_semantic_readiness_last_transition_monotonic", 0.0) or 0.0
        )
        readiness_age_ms = (
            int(max(0.0, (time.monotonic() - readiness_last_transition) * 1000.0))
            if readiness_last_transition > 0.0
            else -1
        )
        return {
            "ready": bool(provider_ready and self._semantic_provider_ready_last),
            "query_embedding_not_ready_streak": int(self._semantic_query_embedding_not_ready_streak),
            "last_error_code": str(getattr(self, "_semantic_provider_last_error_code", "none") or "none"),
            "readiness_reason": str(readiness_reason),
            "last_canary_age_ms": canary_age_ms,
            "readiness_last_transition_at": readiness_last_transition,
            "readiness_age_ms": readiness_age_ms,
            "readiness_transition_count": int(getattr(self, "_semantic_readiness_transition_count", 0)),
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
                scope_label = resolved_scope.value
                session_label = resolved_session_id if resolved_session_id is not None else "none"
                logger.info(
                    "memory_embedding_audit event=enqueued memory_id=%s source=%s scope=%s session_id=%s mode=background",
                    entry.memory_id,
                    source,
                    scope_label,
                    session_label,
                )
            except Exception as exc:  # noqa: BLE001
                # Embedding scheduling is best-effort and must never block writes.
                pass
            return entry

        should_try_inline_embedding = (
            self._semantic_config.enabled
            and not self._semantic_config.background_embedding_enabled
            and self._semantic_config.inline_embedding_on_write_when_background_disabled
        )
        if not should_try_inline_embedding:
            return entry

        inline_started_monotonic = time.monotonic()
        try:
            embedding_result = self._embed_text_with_semantic_policy(text=normalized_content, operation="write")
        except Exception as exc:  # noqa: BLE001
            self._store.upsert_memory_embedding(
                memory_id=entry.memory_id,
                model_id="",
                dim=0,
                vector=b"",
                vector_norm=None,
                status="error",
                error=f"inline_embedding_exception:{exc.__class__.__name__}",
            )
            latency_ms = int((time.monotonic() - inline_started_monotonic) * 1000.0)
            logger.info(
                "memory_embedding_audit event=inline-attempted memory_id=%s source=%s scope=%s session_id=%s "
                "mode=inline outcome=failure error_code=exception:%s latency_ms=%s",
                entry.memory_id,
                source,
                resolved_scope.value,
                resolved_session_id if resolved_session_id is not None else "none",
                exc.__class__.__name__,
                latency_ms,
            )
            return entry

        latency_ms = int((time.monotonic() - inline_started_monotonic) * 1000.0)
        if (
            embedding_result.status == "ready"
            and embedding_result.dimension > 0
            and bool(embedding_result.vector)
        ):
            self._store.upsert_memory_embedding(
                memory_id=entry.memory_id,
                model_id=str(getattr(embedding_result, "model", "")),
                dim=int(embedding_result.dimension),
                vector=bytes(embedding_result.vector),
                vector_norm=float(embedding_result.vector_norm) if embedding_result.vector_norm is not None else None,
                status="ready",
                error=None,
            )
            logger.info(
                "memory_embedding_audit event=inline-attempted memory_id=%s source=%s scope=%s session_id=%s "
                "mode=inline outcome=success error_code=none latency_ms=%s",
                entry.memory_id,
                source,
                resolved_scope.value,
                resolved_session_id if resolved_session_id is not None else "none",
                latency_ms,
            )
            return entry

        error_code = str(getattr(embedding_result, "error_code", "") or "unknown")
        pending_codes = {"timeout", "rate_limited"}
        failure_status = "pending" if error_code in pending_codes else "error"
        self._store.upsert_memory_embedding(
            memory_id=entry.memory_id,
            model_id=str(getattr(embedding_result, "model", "")),
            dim=0,
            vector=b"",
            vector_norm=None,
            status=failure_status,
            error=f"inline_embedding_{error_code}",
        )
        logger.info(
            "memory_embedding_audit event=inline-attempted memory_id=%s source=%s scope=%s session_id=%s "
            "mode=inline outcome=failure error_code=%s latency_ms=%s",
            entry.memory_id,
            source,
            resolved_scope.value,
            resolved_session_id if resolved_session_id is not None else "none",
            error_code,
            latency_ms,
        )
        return entry

    def _reserve_semantic_budget(self, *, operation: str, now_monotonic: float | None = None) -> bool:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if operation == "write":
            limit = int(getattr(self._semantic_config, "max_writes_per_minute", 120))
            timestamps = getattr(self, "_semantic_write_call_timestamps", None)
            if timestamps is None:
                timestamps = []
                self._semantic_write_call_timestamps = timestamps
        else:
            limit = int(getattr(self._semantic_config, "max_queries_per_minute", 240))
            timestamps = getattr(self, "_semantic_query_call_timestamps", None)
            if timestamps is None:
                timestamps = []
                self._semantic_query_call_timestamps = timestamps

        if limit <= 0:
            return False

        cutoff = now - 60.0
        timestamps[:] = [ts for ts in timestamps if ts >= cutoff]
        if len(timestamps) >= limit:
            return False
        timestamps.append(now)
        return True

    def _embed_text_with_semantic_policy(
        self,
        *,
        text: str,
        operation: str,
        enforce_budget: bool = True,
        timeout_override_ms: int | None = None,
        suppress_canary_refresh: bool = False,
    ):
        provider = getattr(self, "_embedding_provider", None)
        if provider is None:
            provider = build_embedding_provider(ConfigController.get_instance().get_config())

        now_monotonic = time.monotonic()
        backoff_until = float(getattr(self, "_semantic_timeout_backoff_until_monotonic", 0.0))
        if backoff_until > now_monotonic and not suppress_canary_refresh:
            result = SimpleNamespace(
                status="unavailable",
                dimension=0,
                vector=b"",
                vector_norm=None,
                error_code="timeout_backoff",
            )
            self._semantic_provider_ready_last = False
            self._semantic_provider_last_error_code = "timeout_backoff"
            return result

        if enforce_budget and not self._reserve_semantic_budget(operation=operation):
            result = SimpleNamespace(
                status="unavailable",
                dimension=0,
                vector=b"",
                vector_norm=None,
                error_code="rate_limited",
            )
            self._semantic_provider_ready_last = False
            self._semantic_provider_last_error_code = "rate_limited"
            return result

        timeout_ms = int(
            timeout_override_ms
            if timeout_override_ms is not None
            else getattr(
                self._semantic_config,
                "write_timeout_ms" if operation == "write" else "query_timeout_ms",
                75 if operation == "write" else 2000,
            )
        )
        if operation == "query" and timeout_ms < SEMANTIC_QUERY_TIMEOUT_FLOOR_MS:
            if not bool(getattr(self, "_semantic_query_timeout_floor_warned", False)):
                logger.warning(
                    "Semantic query timeout below supported floor; clamping to floor.",
                    extra={
                        "configured_query_timeout_ms": timeout_ms,
                        "query_timeout_floor_ms": SEMANTIC_QUERY_TIMEOUT_FLOOR_MS,
                    },
                )
                setattr(self, "_semantic_query_timeout_floor_warned", True)
            timeout_ms = SEMANTIC_QUERY_TIMEOUT_FLOOR_MS
        timeout_s = max(0.001, timeout_ms / 1000.0)

        executor = self._get_embedding_executor()
        started_monotonic = time.monotonic()
        future = executor.submit(
            _invoke_embed_text,
            provider,
            text=text,
            timeout_s=timeout_s,
            timeout_ms=timeout_ms,
            operation=operation,
        )

        def _attach_timing_metadata(response: object) -> None:
            elapsed_ms = int((time.monotonic() - started_monotonic) * 1000)
            timeout_override_ms_used = int(timeout_override_ms) if timeout_override_ms is not None else None
            setattr(response, "timeout_ms_used", timeout_ms)
            setattr(response, "timeout_override_ms", timeout_override_ms_used)
            setattr(response, "operation", operation)
            setattr(response, "elapsed_ms", elapsed_ms)
            if operation == "query" and not suppress_canary_refresh:
                self._query_embedding_latency_samples.add_sample(elapsed_ms)
            logger.debug(
                "semantic_embedding_attempt operation=%s provider=%s timeout_ms_used=%s timeout_override_ms=%s elapsed_ms=%s error_code=%s",
                operation,
                type(provider).__name__,
                timeout_ms,
                timeout_override_ms_used,
                elapsed_ms,
                getattr(response, "error_code", None),
            )

        try:
            result = future.result(timeout=timeout_s)
            _attach_timing_metadata(result)
            self._semantic_timeout_consecutive_count = 0
            status = str(getattr(result, "status", ""))
            ready = status == "ready" and bool(getattr(result, "vector", b""))
            error_code = str(getattr(result, "error_code", "") or ("none" if ready else (status or "unknown")))
            self._semantic_provider_ready_last = ready
            self._semantic_provider_last_error_code = error_code
            return result
        except FutureTimeoutError:
            future.cancel()
            self._semantic_timeout_count = max(0, int(getattr(self, "_semantic_timeout_count", 0))) + 1
            previous_timeout_streak = max(0, int(getattr(self, "_semantic_timeout_consecutive_count", 0)))
            timeout_streak = previous_timeout_streak + 1
            self._semantic_timeout_consecutive_count = timeout_streak
            threshold = max(1, int(getattr(self, "_semantic_timeout_backoff_threshold", 3)))
            crossed_timeout_threshold = previous_timeout_streak < threshold <= timeout_streak
            if operation == "write":
                self._shutdown_embedding_executor()
            if timeout_streak >= threshold:
                window_s = max(0.1, float(getattr(self, "_semantic_timeout_backoff_window_s", 5.0)))
                self._semantic_timeout_backoff_until_monotonic = time.monotonic() + window_s
                self._semantic_timeout_backoff_activation_count = (
                    max(0, int(getattr(self, "_semantic_timeout_backoff_activation_count", 0))) + 1
                )
                self._shutdown_embedding_executor()
            if crossed_timeout_threshold and not suppress_canary_refresh:
                self._maybe_refresh_canary(reason="runtime_timeout_streak")
            result = SimpleNamespace(
                status="error",
                dimension=0,
                vector=b"",
                vector_norm=None,
                error_code="timeout",
            )
            _attach_timing_metadata(result)
            self._semantic_provider_ready_last = False
            self._semantic_provider_last_error_code = "timeout"
            return result

    def _find_semantic_duplicate(
        self,
        *,
        content: str,
        user_id: str,
        scope: MemoryScope,
        session_id: str | None,
        source: str,
    ) -> MemoryEntry | None:
        self._last_semantic_dedupe_debug = {
            "enabled": False,
            "status": "skipped",
            "error_code": None,
            "error_class": None,
        }
        if not self._semantic_config.enabled:
            return None
        if source == "manual_tool" and not self._auto_reflection_dedupe_apply_to_manual_tool:
            return None
        if source != "auto_reflection" and source != "manual_tool":
            return None
        if not self._auto_reflection_semantic_dedupe_enabled:
            return None
        self._last_semantic_dedupe_debug["enabled"] = True

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

        try:
            candidate_embedding = self._embed_text_with_semantic_policy(text=content, operation="write")
        except Exception as exc:  # noqa: BLE001
            self._last_semantic_dedupe_debug.update(
                {
                    "status": "provider_error",
                    "error_code": "embedding_provider_exception",
                    "error_class": exc.__class__.__name__,
                }
            )
            logger.warning(
                "Semantic dedupe embedding failed; proceeding without duplicate match.",
                extra={
                    "error_code": "embedding_provider_exception",
                    "error_class": exc.__class__.__name__,
                    "source": source,
                },
            )
            return None

        self._last_semantic_dedupe_debug["status"] = "embedded"
        if (
            candidate_embedding.status != "ready"
            or candidate_embedding.dimension <= 0
            or not candidate_embedding.vector
            or float(candidate_embedding.vector_norm or 0.0) <= 0.0
        ):
            error_code = getattr(candidate_embedding, "error_code", None)
            if error_code:
                self._last_semantic_dedupe_debug["error_code"] = str(error_code)
            self._last_semantic_dedupe_debug["status"] = "embedding_not_ready"
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
            self._last_semantic_dedupe_debug["status"] = "no_match"
            return None
        self._last_semantic_dedupe_debug["status"] = "match"
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
            strict=True,
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
            chars = estimate_startup_memory_digest_item_chars(index=len(selected) + 1, item=summary)
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

    def forget_memory(self, *, memory_id: int, allow_admin_override: bool = False) -> bool:
        del allow_admin_override  # Backward-compatible argument retained for callers.
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
            "semantic_provider_ready": False,
            "semantic_attempted": False,
            "semantic_applied": False,
            "semantic_error": None,
            "candidate_count": 0,
            "lexical_candidate_count": 0,
            "semantic_candidate_count": 0,
            "semantic_scored_count": 0,
            "candidates_without_ready_embedding": 0,
            "candidates_below_influence_threshold": 0,
            "candidates_semantic_applied": 0,
            "selected_count": 0,
            "scope": None,
            "cooldown_skipped": False,
            "cooldown_consumed": False,
            "fallback_reason": None,
            "latency_ms": 0,
            "dedupe_count": 0,
            "truncation_count": 0,
            "semantic_timeout_count": int(getattr(self, "_semantic_timeout_count", 0)),
            "semantic_timeout_consecutive_count": int(
                getattr(self, "_semantic_timeout_consecutive_count", 0)
            ),
            "semantic_timeout_backoff_active": False,
            "semantic_timeout_backoff_remaining_ms": 0,
            "semantic_timeout_backoff_activation_count": int(
                getattr(self, "_semantic_timeout_backoff_activation_count", 0)
            ),
            "semantic_provider": str(getattr(self._semantic_config, "provider", "none") or "none"),
            "semantic_model": None,
            "semantic_query_timeout_ms": int(getattr(self._semantic_config, "query_timeout_ms", 2000)),
            "semantic_query_timeout_ms_used": None,
            "semantic_query_duration_ms": 0,
            "semantic_query_embed_elapsed_ms": 0,
            "semantic_result_status": None,
            "semantic_error_code": None,
            "semantic_error_class": None,
            "semantic_failure_class": None,
            "semantic_scoring_skipped_reason": None,
            "query_fingerprint_hash": None,
            "query_fingerprint_length": 0,
        }

        now_for_backoff = time.monotonic()
        backoff_until = float(getattr(self, "_semantic_timeout_backoff_until_monotonic", 0.0))
        if backoff_until > now_for_backoff:
            self._last_turn_retrieval_debug["semantic_timeout_backoff_active"] = True
            self._last_turn_retrieval_debug["semantic_timeout_backoff_remaining_ms"] = int(
                (backoff_until - now_for_backoff) * 1000.0
            )

        clean_utterance = " ".join(latest_user_utterance.strip().split())
        if not clean_utterance:
            return None
        query_fingerprint = _safe_text_fingerprint(clean_utterance)
        self._last_turn_retrieval_debug["query_fingerprint_hash"] = query_fingerprint["hash"]
        self._last_turn_retrieval_debug["query_fingerprint_length"] = query_fingerprint["length"]

        effective_user_id = user_id if user_id is not None else self._active_user_id
        resolved_scope = _normalize_scope(scope, fallback=self._default_scope)
        resolved_session_id = self._resolve_session_id_for_scope(
            scope=resolved_scope,
            session_id=session_id,
            strict=False,
        )

        timestamp = now_monotonic if now_monotonic is not None else time.monotonic()
        cooldown_key = (effective_user_id, resolved_scope, resolved_session_id)
        self._last_turn_retrieval_debug["scope"] = resolved_scope.value
        if cooldown_s > 0.0:
            last_retrieval = self._last_turn_retrieval_at.get(cooldown_key)
            if last_retrieval is not None and timestamp - last_retrieval < cooldown_s:
                self._last_turn_retrieval_debug.update({"mode": "cooldown", "cooldown_skipped": True})
                return None

        bounded_max_memories = _clamp(max_memories, 1, MAX_RECALL_LIMIT)
        bounded_max_chars = _clamp(max_chars, 80, 4000)
        candidate_limit = _clamp(
            max(bounded_max_memories * RANK_CANDIDATE_MULTIPLIER, bounded_max_memories),
            bounded_max_memories,
            MAX_RETRIEVAL_CANDIDATE_CAP,
        )
        recency_importance_entries = self._store.search_memories(
            query=None,
            limit=candidate_limit,
            user_id=effective_user_id,
            scope=resolved_scope,
            session_id=resolved_session_id,
            review_state="approved",
        )
        query_entries = self._store.search_memories(
            query=clean_utterance,
            limit=candidate_limit,
            user_id=effective_user_id,
            scope=resolved_scope,
            session_id=resolved_session_id,
            review_state="approved",
        )

        entries: list[MemoryEntry] = []
        seen_memory_ids: set[int] = set()
        for entry in [*query_entries, *recency_importance_entries]:
            if entry.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(entry.memory_id)
            entries.append(entry)
            if len(entries) >= candidate_limit:
                break

        if not entries:
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
            self._last_turn_retrieval_debug["mode"] = "filtered_empty"
            return None

        self._last_turn_retrieval_debug["candidate_count"] = len(scored_entries)

        lexical_ordered = sorted(
            scored_entries,
            key=lambda pair: (-pair[0], -pair[1].importance, -pair[1].timestamp, pair[1].memory_id),
        )
        ordered = [item[1] for item in lexical_ordered]

        self._maybe_refresh_canary(reason="periodic")
        semantic_provider_ready, semantic_readiness_reason = self._is_semantic_provider_ready()
        semantic_enabled = (
            self._semantic_config.enabled
            and self._semantic_config.rerank_enabled
            and semantic_provider_ready
        )
        self._last_turn_retrieval_debug["semantic_enabled"] = semantic_enabled
        self._last_turn_retrieval_debug["semantic_provider_ready"] = semantic_provider_ready
        self._last_turn_retrieval_debug["semantic_readiness_reason"] = semantic_readiness_reason
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
                semantic_started_at = time.monotonic()
                query_embedding = self._embed_text_with_semantic_policy(
                    text=clean_utterance,
                    operation="query",
                )
                self._last_turn_retrieval_debug["semantic_query_duration_ms"] = int(
                    (time.monotonic() - semantic_started_at) * 1000
                )
                self._last_turn_retrieval_debug["semantic_query_timeout_ms_used"] = int(
                    getattr(query_embedding, "timeout_ms_used", self._semantic_config.query_timeout_ms)
                )
                self._last_turn_retrieval_debug["semantic_query_embed_elapsed_ms"] = int(
                    getattr(query_embedding, "elapsed_ms", self._last_turn_retrieval_debug["semantic_query_duration_ms"])
                )
                self._last_turn_retrieval_debug["semantic_result_status"] = str(
                    getattr(query_embedding, "status", "unknown") or "unknown"
                )
                self._last_turn_retrieval_debug["semantic_model"] = str(
                    getattr(query_embedding, "model", None)
                    or getattr(self._embedding_provider, "_model", None)
                    or "unknown"
                )
                if query_embedding.status == "ready" and query_embedding.dimension > 0 and query_embedding.vector:
                    query_norm = float(query_embedding.vector_norm or 0.0)
                    if query_norm > 0.0:
                        embeddings = self._store.fetch_embeddings_for_memories(memory_ids=semantic_memory_ids)
                        hybrid_pool: list[tuple[float, MemoryEntry]] = []
                        semantic_scored_count = 0
                        candidates_without_ready_embedding = 0
                        candidates_below_influence_threshold = 0
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
                                else:
                                    candidates_below_influence_threshold += 1
                            else:
                                candidates_without_ready_embedding += 1
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
                                "candidates_without_ready_embedding": candidates_without_ready_embedding,
                                "candidates_below_influence_threshold": candidates_below_influence_threshold,
                                "candidates_semantic_applied": semantic_scored_count,
                            }
                        )
                    else:
                        self._last_turn_retrieval_debug["semantic_scoring_skipped_reason"] = "query_embedding_zero_norm"
                else:
                    self._last_turn_retrieval_debug["fallback_reason"] = "query_embedding_not_ready"
                    error_code = str(getattr(query_embedding, "error_code", None) or getattr(query_embedding, "status", None) or "unknown")
                    error_class = _sanitize_error_class_name(
                        getattr(query_embedding, "error_class", None)
                    )
                    failure_class = _normalize_semantic_failure_class(
                        error_code=error_code,
                        error_class=error_class,
                    )
                    self._last_turn_retrieval_debug["semantic_error_code"] = error_code
                    self._last_turn_retrieval_debug["semantic_error_class"] = error_class
                    self._last_turn_retrieval_debug["semantic_failure_class"] = failure_class
                    self._last_turn_retrieval_debug["semantic_scoring_skipped_reason"] = (
                        "query_embedding_backoff"
                        if error_code == "timeout_backoff"
                        else (
                            "query_embedding_timeout"
                            if failure_class == "timeout"
                            else (
                                "query_embedding_rate_limited"
                                if failure_class == "rate_limited"
                                else "query_embedding_not_ready"
                            )
                        )
                    )
                    if self._last_turn_retrieval_debug["semantic_scoring_skipped_reason"] == "query_embedding_backoff":
                        self._last_turn_retrieval_debug["fallback_reason"] = "query_embedding_backoff"
                    elif self._last_turn_retrieval_debug["semantic_scoring_skipped_reason"] == "query_embedding_rate_limited":
                        self._last_turn_retrieval_debug["fallback_reason"] = "query_embedding_rate_limited"
                    elif self._last_turn_retrieval_debug["semantic_scoring_skipped_reason"] == "query_embedding_timeout":
                        self._last_turn_retrieval_debug["fallback_reason"] = "query_embedding_timeout"
            except Exception as exc:  # noqa: BLE001
                # Semantic reranking is best-effort and must never degrade lexical retrieval.
                ordered = [item[1] for item in lexical_ordered]
                self._retrieval_semantic_error_count += 1
                exception_error_class = _sanitize_error_class_name(exc.__class__.__name__)
                self._last_turn_retrieval_debug.update(
                    {
                        "mode": "hybrid_fallback_lexical",
                        "semantic_error": "exception",
                        "semantic_result_status": "exception",
                        "semantic_error_code": "semantic_provider_exception",
                        "semantic_error_class": exception_error_class,
                        "semantic_failure_class": _normalize_semantic_failure_class(
                            error_code="semantic_provider_exception",
                            error_class=exception_error_class,
                        ),
                        "semantic_scoring_skipped_reason": "semantic_provider_error",
                        "fallback_reason": "semantic_provider_error",
                    }
                )
        elif semantic_enabled:
            self._last_turn_retrieval_debug["fallback_reason"] = "no_lexical_candidates"
            self._last_turn_retrieval_debug["semantic_scoring_skipped_reason"] = "no_lexical_candidates"
        elif self._semantic_config.enabled and self._semantic_config.rerank_enabled:
            self._last_turn_retrieval_debug["fallback_reason"] = semantic_readiness_reason
            self._last_turn_retrieval_debug["semantic_scoring_skipped_reason"] = "semantic_not_ready"

        if self._last_turn_retrieval_debug.get("semantic_scoring_skipped_reason") in {
            "query_embedding_not_ready",
            "query_embedding_timeout",
            "query_embedding_rate_limited",
            "query_embedding_backoff",
        }:
            self._semantic_query_embedding_not_ready_streak = (
                max(0, int(getattr(self, "_semantic_query_embedding_not_ready_streak", 0))) + 1
            )
        else:
            self._semantic_query_embedding_not_ready_streak = 0
        self._last_turn_retrieval_debug["semantic_query_embedding_not_ready_streak"] = int(
            self._semantic_query_embedding_not_ready_streak
        )

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
            candidate_index = len(selected) + 1
            candidate_chars = estimate_realtime_memory_brief_item_chars(index=candidate_index, item=candidate_summary)
            if used_chars + candidate_chars > bounded_max_chars:
                remaining = bounded_max_chars - used_chars
                overhead_chars = estimate_realtime_memory_brief_item_chars(
                    index=candidate_index,
                    item=MemorySummary(
                        memory_id=candidate_summary.memory_id,
                        content="",
                        tags=candidate_summary.tags,
                        importance=candidate_summary.importance,
                        source=candidate_summary.source,
                        pinned=candidate_summary.pinned,
                        needs_review=candidate_summary.needs_review,
                    ),
                )
                remaining_content_budget = remaining - overhead_chars
                if selected or remaining_content_budget < 24:
                    truncated = True
                    truncation_count += 1
                    continue
                clipped = candidate_summary.content[: max(remaining_content_budget - 1, 1)].rstrip()
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
                candidate_chars = estimate_realtime_memory_brief_item_chars(index=candidate_index, item=candidate_summary)
                truncated = True
                truncation_count += 1

            selected.append(candidate_summary)
            seen_contents.append(candidate_summary.content)
            used_chars += candidate_chars

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
        self._last_turn_retrieval_at[cooldown_key] = timestamp
        self._last_turn_retrieval_debug["cooldown_consumed"] = True
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


def _semantic_api_key_present(config: dict[str, object]) -> bool:
    semantic_cfg = dict(config.get("memory_semantic") or {})
    provider = str(semantic_cfg.get("provider", "none")).strip().lower()
    if provider != "openai":
        return False
    openai_cfg = dict(semantic_cfg.get("openai") or {})
    configured_key = str(openai_cfg.get("api_key") or "").strip()
    env_key = str(os.getenv("OPENAI_API_KEY", "")).strip()
    return bool(configured_key or env_key)


def _semantic_canary_bypass_enabled(config: dict[str, object]) -> bool:
    semantic_cfg = dict(config.get("memory_semantic") or {})
    return bool(
        semantic_cfg.get("startup_canary_bypass", False)
        or semantic_cfg.get("test_mode_bypass_canary", False)
        or semantic_cfg.get("offline_mode", False)
        or os.getenv("PYPIBOT_SEMANTIC_CANARY_BYPASS", "").strip().lower() in {"1", "true", "yes", "on"}
    )


def run_embedding_probe_once(
    *,
    provider: EmbeddingProvider,
    enabled: bool,
    bypass: bool,
    timeout_ms: int,
) -> dict[str, bool | int | None | str]:
    result: dict[str, bool | int | None | str] = {
        "canary_success": False,
        "latency_ms": 0,
        "dimension": None,
        "error_code": "skipped",
        "error_class": "none",
        "timeout_triggered": "none",
    }
    if not enabled:
        result["error_code"] = "disabled"
        return result
    if bypass:
        result["canary_success"] = True
        result["error_code"] = "bypassed"
        return result

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(provider.embed_text, "ping")
        try:
            response = future.result(timeout=max(0.001, timeout_ms / 1000.0))
        except FutureTimeoutError:
            future.cancel()
            result["latency_ms"] = int((time.perf_counter() - started) * 1000.0)
            result["error_code"] = "timeout"
            result["error_class"] = "FutureTimeoutError"
            result["timeout_triggered"] = "canary_timeout"
            return result
        except Exception as exc:  # noqa: BLE001
            result["latency_ms"] = int((time.perf_counter() - started) * 1000.0)
            result["error_code"] = _normalize_canary_error_code(None, exc=exc)
            result["error_class"] = _sanitize_error_class_name(exc.__class__.__name__)
            return result

    result["latency_ms"] = int((time.perf_counter() - started) * 1000.0)
    dimension = int(getattr(response, "dimension", 0) or 0)
    result["dimension"] = dimension if dimension > 0 else 0
    status = str(getattr(response, "status", "") or "")
    vector = getattr(response, "vector", b"")
    if status == "ready" and bool(vector):
        result["canary_success"] = True
        result["error_code"] = "none"
        return result

    raw_error = str(getattr(response, "error_code", "") or status or "unknown")
    result["error_code"] = _normalize_canary_error_code(raw_error)
    response_error_class = _sanitize_error_class_name(getattr(response, "error_class", None))
    result["error_class"] = response_error_class or "none"
    if result["error_code"] == "timeout":
        result["timeout_triggered"] = "provider_timeout"
    return result


def _run_embed_canary_once(
    *,
    provider: EmbeddingProvider,
    enabled: bool,
    bypass: bool,
    startup_canary_timeout_ms: int,
) -> dict[str, bool | int | None | str]:
    return run_embedding_probe_once(
        provider=provider,
        enabled=enabled,
        bypass=bypass,
        timeout_ms=startup_canary_timeout_ms,
    )


def run_embed_canary_cli(
    argv: list[str] | None = None,
    *,
    provider_factory: Callable[[dict[str, object]], EmbeddingProvider] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run memory semantic startup canary diagnostics.")
    parser.add_argument("--embed-canary", action="store_true", help="Run semantic embedding startup canary checks.")
    parser.add_argument("--embed-probe", action="store_true", help="Run one semantic embedding diagnostic probe.")
    args = parser.parse_args(argv)
    if not args.embed_canary and not args.embed_probe:
        parser.print_help()
        return 2

    ConfigController._instance = None
    config = ConfigController.get_instance().get_config()
    semantic_cfg = dict(config.get("memory_semantic") or {})
    openai_cfg = dict(semantic_cfg.get("openai") or {})

    provider_name = str(semantic_cfg.get("provider", "none"))
    provider_model = str(openai_cfg.get("model", "text-embedding-3-small"))
    provider_timeout_s = float(openai_cfg.get("timeout_s", 10.0))
    startup_canary_timeout_ms = int(semantic_cfg.get("startup_canary_timeout_ms", 0))
    enabled = bool(semantic_cfg.get("enabled", False))
    bypass = _semantic_canary_bypass_enabled(config)

    provider = provider_factory(config) if provider_factory is not None else build_embedding_provider(config)
    canary = _run_embed_canary_once(
        provider=provider,
        enabled=enabled,
        bypass=bypass,
        startup_canary_timeout_ms=startup_canary_timeout_ms,
    )

    print(
        f"provider={provider_name} model={provider_model} timeout_s={provider_timeout_s} "
        f"startup_canary_timeout_ms={startup_canary_timeout_ms}"
    )
    print(f"api_key_present={_semantic_api_key_present(config)}")
    print(
        f"canary_success={canary['canary_success']} latency_ms={canary['latency_ms']} "
        f"dimension={canary['dimension']} error_code={canary['error_code']} "
        f"error_class={canary['error_class']} timeout_triggered={canary['timeout_triggered']}"
    )
    return 0 if bool(canary["canary_success"]) else 1


def main(argv: list[str] | None = None) -> int:
    return run_embed_canary_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
