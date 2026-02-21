"""Background worker for asynchronous memory embedding generation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time

from config import ConfigController
from services.embedding_provider import EmbeddingProvider, build_embedding_provider
from storage.memories import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class _FailureState:
    failures: int
    next_retry_monotonic: float


class MemoryEmbeddingWorker:
    """Best-effort, fail-open embedding worker for pending memories."""

    _TERMINAL_ERROR_MARKER = "terminal_retry_exhausted"

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        provider: EmbeddingProvider | None = None,
        batch_size: int = 4,
        idle_sleep_s: float = 2.0,
        base_backoff_s: float = 2.0,
        max_backoff_s: float = 60.0,
        rolling_backfill_batch_size: int | None = None,
        rolling_backfill_interval_idle_cycles: int | None = None,
    ) -> None:
        config = ConfigController.get_instance().get_config()
        self._store = store if store is not None else MemoryStore()
        self._provider = provider if provider is not None else build_embedding_provider(config)
        self._batch_size = max(1, int(batch_size))
        self._idle_sleep_s = max(0.1, float(idle_sleep_s))
        self._base_backoff_s = max(0.1, float(base_backoff_s))
        self._max_backoff_s = max(self._base_backoff_s, float(max_backoff_s))
        semantic_cfg = config.get("memory_semantic", {})
        configured_max_embedding_retries = int(semantic_cfg.get("max_embedding_retries", 8))
        self._max_embedding_retries = max(1, configured_max_embedding_retries)
        configured_backfill_batch_size = int(
            semantic_cfg.get("rolling_backfill_batch_size", 4)
        )
        configured_backfill_interval = int(
            semantic_cfg.get("rolling_backfill_interval_idle_cycles", 15)
        )
        self._rolling_backfill_batch_size = max(
            1,
            int(
                rolling_backfill_batch_size
                if rolling_backfill_batch_size is not None
                else configured_backfill_batch_size
            ),
        )
        self._rolling_backfill_interval_idle_cycles = max(
            1,
            int(
                rolling_backfill_interval_idle_cycles
                if rolling_backfill_interval_idle_cycles is not None
                else configured_backfill_interval
            ),
        )
        self._idle_cycles_since_backfill = 0

        self._failures: dict[int, _FailureState] = {}
        self._consecutive_failures = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        queue_metrics = self.get_metrics()
        logger.info(
            "memory embedding worker start pending=%s retry_blocked=%s oldest_pending_age_ms=%s",
            queue_metrics.get("pending_count", 0),
            queue_metrics.get("retry_blocked_count", 0),
            queue_metrics.get("oldest_pending_age_ms", 0),
        )
        self._thread = threading.Thread(target=self._run_loop, name="memory-embedding-worker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, float(timeout_s)))
        queue_metrics = self.get_metrics()
        logger.info(
            "memory embedding worker stop pending=%s retry_blocked=%s oldest_pending_age_ms=%s",
            queue_metrics.get("pending_count", 0),
            queue_metrics.get("retry_blocked_count", 0),
            queue_metrics.get("oldest_pending_age_ms", 0),
        )
        self._thread = None

    def enqueue_memory(self, *, memory_id: int) -> None:
        self._store.enqueue_memory_embedding(memory_id=memory_id)
        logger.debug("memory embedding enqueued memory_id=%s", memory_id)

    def backfill_recent_missing_embeddings(self, *, limit: int = 8) -> int:
        """Queue a small startup backfill batch for recent rows missing embeddings."""

        queued = 0
        for memory_id in self._store.list_recent_memories_missing_embeddings(limit=limit):
            self.enqueue_memory(memory_id=memory_id)
            queued += 1
        return queued

    def run_once(self, *, max_items: int | None = None) -> int:
        """Process at most one bounded batch and return successful upsert count."""

        limit = self._batch_size if max_items is None else max(1, min(self._batch_size, int(max_items)))
        entries = self._store.fetch_pending_memories_for_embedding(limit=limit)
        if not entries:
            self._consecutive_failures = 0
            return 0

        now = time.monotonic()
        ready_entries = [entry for entry in entries if self._is_retry_ready(entry.memory_id, now_monotonic=now)]
        if not ready_entries:
            logger.debug(
                "memory embedding batch summary processed=%s succeeded=0 failed=0 skipped_retry_blocked=%s",
                len(entries),
                len(entries),
            )
            return 0

        try:
            results = self._provider.embed_batch([entry.content for entry in ready_entries])
        except Exception as exc:  # noqa: BLE001
            error_code = f"provider_exception:{type(exc).__name__}"
            for entry in ready_entries:
                self._record_failure(entry=entry, error_message=error_code, error_code=error_code)
            self._consecutive_failures += 1
            logger.warning(
                "memory embedding batch summary processed=%s succeeded=0 failed=%s error_code=%s",
                len(ready_entries),
                len(ready_entries),
                error_code,
            )
            return 0

        success_count = 0
        failure_count = 0
        for entry, result in zip(ready_entries, results):
            if result.status == "ready" and result.dimension > 0 and result.vector:
                self._store.upsert_memory_embedding(
                    memory_id=entry.memory_id,
                    model_id=result.model,
                    dim=result.dimension,
                    vector=result.vector,
                    vector_norm=result.vector_norm,
                    status="ready",
                    error=None,
                )
                self._failures.pop(entry.memory_id, None)
                success_count += 1
                continue
            error_code = result.error_code or "embedding_failed"
            self._record_failure(
                entry=entry,
                error_message=result.error_message or error_code,
                error_code=error_code,
            )
            failure_count += 1
        if len(results) < len(ready_entries):
            for entry in ready_entries[len(results) :]:
                self._record_failure(
                    entry=entry,
                    error_message="provider_returned_fewer_results",
                    error_code="provider_returned_fewer_results",
                )
                failure_count += 1
        if success_count > 0:
            self._consecutive_failures = 0
        elif ready_entries:
            self._consecutive_failures += 1
        logger.info(
            "memory embedding batch summary processed=%s succeeded=%s failed=%s",
            len(ready_entries),
            success_count,
            failure_count,
        )
        return success_count

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                processed = self.run_once()
                if processed > 0:
                    self._idle_cycles_since_backfill = 0
                else:
                    self._idle_cycles_since_backfill += 1
                    if self._idle_cycles_since_backfill >= self._rolling_backfill_interval_idle_cycles:
                        self._idle_cycles_since_backfill = 0
                        queued = self.backfill_recent_missing_embeddings(
                            limit=self._rolling_backfill_batch_size
                        )
                        if queued > 0:
                            logger.debug("memory embedding rolling backfill queued=%s", queued)
            except Exception as exc:  # noqa: BLE001
                self._consecutive_failures += 1
                logger.debug("memory embedding worker pass failed open: %s", exc)
            self._stop_event.wait(self._idle_sleep_s)

    def get_metrics(self) -> dict[str, int]:
        """Return low-overhead worker queue and retry telemetry."""

        pending_count, oldest_pending_updated_at = self._store.get_pending_embedding_queue_stats()
        now_monotonic = time.monotonic()
        retry_blocked_count = sum(
            1
            for failure in self._failures.values()
            if float(failure.next_retry_monotonic) > now_monotonic
        )
        oldest_pending_age_ms = 0
        if oldest_pending_updated_at is not None:
            oldest_pending_age_ms = max(0, int(time.time() * 1000) - int(oldest_pending_updated_at))
        return {
            "pending_count": int(pending_count),
            "retry_blocked_count": int(retry_blocked_count),
            "consecutive_failures": int(self._consecutive_failures),
            "oldest_pending_age_ms": int(oldest_pending_age_ms),
        }

    def _is_retry_ready(self, memory_id: int, *, now_monotonic: float) -> bool:
        failure_state = self._failures.get(memory_id)
        if failure_state is None:
            return True
        return now_monotonic >= failure_state.next_retry_monotonic

    def _record_failure(self, *, entry: MemoryEntry, error_message: str, error_code: str) -> None:
        prior = self._failures.get(entry.memory_id)
        failures = 1 if prior is None else prior.failures + 1
        if failures > self._max_embedding_retries:
            terminal_error = f"{self._TERMINAL_ERROR_MARKER}:{error_code}"
            self._store.upsert_memory_embedding(
                memory_id=entry.memory_id,
                model_id="",
                dim=0,
                vector=b"",
                vector_norm=None,
                status="error",
                error=terminal_error,
            )
            self._failures.pop(entry.memory_id, None)
            logger.info(
                "memory embedding failure memory_id=%s classification=retry_exhausted failures=%s error_code=%s",
                entry.memory_id,
                failures,
                error_code,
            )
            return
        backoff_s = min(self._max_backoff_s, self._base_backoff_s * (2 ** (failures - 1)))
        self._failures[entry.memory_id] = _FailureState(
            failures=failures,
            next_retry_monotonic=time.monotonic() + backoff_s,
        )
        self._store.upsert_memory_embedding(
            memory_id=entry.memory_id,
            model_id="",
            dim=0,
            vector=b"",
            vector_norm=None,
            status="pending",
            error=error_message,
        )
        logger.debug(
            "memory embedding failure memory_id=%s classification=retry_backoff failures=%s error_code=%s next_retry_in_s=%.2f",
            entry.memory_id,
            failures,
            error_code,
            backoff_s,
        )
