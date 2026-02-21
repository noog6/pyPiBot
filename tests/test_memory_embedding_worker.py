"""Tests for asynchronous memory embedding worker behavior."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path

from config.controller import ConfigController
from services.embedding_provider import EmbeddingResult
from services.memory_embedding_worker import MemoryEmbeddingWorker
from services.memory_manager import MemoryManager, MemoryScope
from storage.memories import MemoryStore


@dataclass
class _SequenceProvider:
    results: list[EmbeddingResult]

    def embed_text(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.results.pop(0) for _ in texts]


def _reset_singletons() -> None:
    ConfigController._instance = None
    MemoryManager._instance = None


def test_worker_backfills_and_generates_embeddings(tmp_path: Path, monkeypatch) -> None:
    _reset_singletons()
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="remember this", tags=[], importance=3, user_id="default")

    provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=(1.0).hex().encode("utf-8"),
                dimension=1,
                model="test-model",
                model_version="v1",
                vector_norm=1.0,
                provider="test",
                status="ready",
            )
        ]
    )
    worker = MemoryEmbeddingWorker(store=store, provider=provider, batch_size=2)

    assert worker.backfill_recent_missing_embeddings(limit=3) == 1
    assert worker.run_once() == 1

    embedding = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert embedding.status == "ready"
    assert embedding.model_id == "test-model"


def test_worker_keeps_pending_with_backoff_on_failures(tmp_path: Path) -> None:
    _reset_singletons()
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="fail me", tags=[], importance=3, user_id="default")
    store.enqueue_memory_embedding(memory_id=entry.memory_id)

    provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=b"",
                dimension=0,
                model="test-model",
                model_version=None,
                vector_norm=None,
                provider="test",
                status="error",
                error_code="boom",
                error_message="boom",
            )
        ]
    )
    worker = MemoryEmbeddingWorker(store=store, provider=provider, batch_size=2, base_backoff_s=30.0)

    assert worker.run_once() == 0
    pending = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert pending.status == "pending"
    assert pending.error == "boom"

    metrics = worker.get_metrics()
    assert metrics["pending_count"] == 1
    assert metrics["retry_blocked_count"] == 1
    assert metrics["consecutive_failures"] == 1
    assert metrics["oldest_pending_age_ms"] >= 0


def test_worker_marks_embedding_error_after_retry_budget_exhausted(tmp_path: Path, caplog) -> None:
    _reset_singletons()
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="fail forever", tags=[], importance=3, user_id="default")
    store.enqueue_memory_embedding(memory_id=entry.memory_id)

    provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=b"",
                dimension=0,
                model="test-model",
                model_version=None,
                vector_norm=None,
                provider="test",
                status="error",
                error_code="boom",
                error_message="boom",
            )
            for _ in range(2)
        ]
    )
    worker = MemoryEmbeddingWorker(
        store=store,
        provider=provider,
        batch_size=1,
        base_backoff_s=0.01,
        max_backoff_s=0.01,
    )
    worker._max_embedding_retries = 1

    caplog.set_level("INFO")
    assert worker.run_once() == 0
    worker._failures[entry.memory_id].next_retry_monotonic = 0
    assert worker.run_once() == 0

    failed = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert failed.status == "error"
    assert failed.error == "terminal_retry_exhausted:boom"
    assert entry.memory_id not in worker._failures
    assert (
        f"memory embedding failure memory_id={entry.memory_id} classification=retry_exhausted failures=2 error_code=boom"
        in caplog.text
    )


def test_record_failure_caps_backoff_and_transitions_to_terminal_error(tmp_path: Path) -> None:
    _reset_singletons()
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="bounded retries", tags=[], importance=3, user_id="default")
    store.enqueue_memory_embedding(memory_id=entry.memory_id)

    worker = MemoryEmbeddingWorker(
        store=store,
        batch_size=1,
        base_backoff_s=0.2,
        max_backoff_s=0.3,
    )
    worker._max_embedding_retries = 2

    worker._record_failure(entry=entry, error_message="boom", error_code="boom")
    first_failure = worker._failures[entry.memory_id]
    first_backoff_s = first_failure.next_retry_monotonic - time.monotonic()
    assert first_failure.failures == 1
    assert 0.0 < first_backoff_s <= 0.35

    worker._record_failure(entry=entry, error_message="boom", error_code="boom")
    second_failure = worker._failures[entry.memory_id]
    second_backoff_s = second_failure.next_retry_monotonic - time.monotonic()
    assert second_failure.failures == 2
    assert 0.0 < second_backoff_s <= 0.35

    worker._record_failure(entry=entry, error_message="boom", error_code="boom")
    failed = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert failed.status == "error"
    assert failed.error == "terminal_retry_exhausted:boom"
    assert entry.memory_id not in worker._failures

    metrics = worker.get_metrics()
    assert metrics["pending_count"] == 0
    assert metrics["retry_blocked_count"] == 0


def test_worker_metrics_reset_consecutive_failures_after_success(tmp_path: Path) -> None:
    _reset_singletons()
    store = MemoryStore(db_path=tmp_path / "memories.db")
    first = store.append_memory(content="first", tags=[], importance=3, user_id="default")
    second = store.append_memory(content="second", tags=[], importance=3, user_id="default")
    store.enqueue_memory_embedding(memory_id=first.memory_id)
    store.enqueue_memory_embedding(memory_id=second.memory_id)

    provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=b"",
                dimension=0,
                model="test-model",
                model_version=None,
                vector_norm=None,
                provider="test",
                status="error",
                error_code="boom",
                error_message="boom",
            ),
            EmbeddingResult(
                vector=(1.0).hex().encode("utf-8"),
                dimension=1,
                model="test-model",
                model_version="v1",
                vector_norm=1.0,
                provider="test",
                status="ready",
            ),
        ]
    )
    worker = MemoryEmbeddingWorker(
        store=store,
        provider=provider,
        batch_size=1,
        base_backoff_s=30.0,
    )

    assert worker.run_once() == 0
    assert worker.get_metrics()["consecutive_failures"] == 1
    assert worker.run_once() == 1

    metrics = worker.get_metrics()
    assert metrics["consecutive_failures"] == 0


def test_worker_logs_batch_summary_and_metrics_on_success(tmp_path: Path, caplog) -> None:
    _reset_singletons()
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="ok", tags=[], importance=3, user_id="default")

    provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=(1.0).hex().encode("utf-8"),
                dimension=1,
                model="test-model",
                model_version="v1",
                vector_norm=1.0,
                provider="test",
                status="ready",
            )
        ]
    )
    worker = MemoryEmbeddingWorker(store=store, provider=provider, batch_size=1)

    caplog.set_level("INFO")
    worker.enqueue_memory(memory_id=entry.memory_id)
    assert worker.run_once() == 1

    assert "memory embedding batch summary processed=1 succeeded=1 failed=0" in caplog.text
    metrics = worker.get_metrics()
    assert set(("pending_count", "retry_blocked_count", "consecutive_failures", "oldest_pending_age_ms")).issubset(metrics)


def test_worker_logs_failure_classification_and_metrics_on_error(tmp_path: Path, caplog) -> None:
    _reset_singletons()
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="bad", tags=[], importance=3, user_id="default")

    provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=b"",
                dimension=0,
                model="test-model",
                model_version=None,
                vector_norm=None,
                provider="test",
                status="error",
                error_code="boom",
                error_message="boom",
            )
        ]
    )
    worker = MemoryEmbeddingWorker(
        store=store,
        provider=provider,
        batch_size=1,
        base_backoff_s=30.0,
    )

    caplog.set_level("DEBUG")
    worker.enqueue_memory(memory_id=entry.memory_id)
    assert worker.run_once() == 0

    assert "classification=retry_backoff" in caplog.text
    assert "memory embedding batch summary processed=1 succeeded=0 failed=1" in caplog.text
    metrics = worker.get_metrics()
    assert metrics["pending_count"] == 1
    assert metrics["retry_blocked_count"] == 1


def test_remember_memory_enqueues_pending_embedding_when_enabled(tmp_path: Path, monkeypatch, caplog) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "assistant_name: Theo",
                f"var_dir: {tmp_path / 'var'}",
                "memory_semantic:",
                "  enabled: true",
                "  provider: openai",
                "  background_embedding_enabled: true",
                "  openai:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    manager = MemoryManager.get_instance()
    manager._store = MemoryStore(db_path=tmp_path / "memories.db")
    manager._embedding_worker = MemoryEmbeddingWorker(store=manager._store)
    manager._default_scope = MemoryScope.USER_GLOBAL

    caplog.set_level("INFO")
    entry = manager.remember_memory(content="queue this memory", importance=3)

    embedding = manager._store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert embedding.status == "pending"
    assert f"memory_embedding_audit event=enqueued memory_id={entry.memory_id}" in caplog.text


def test_worker_run_loop_periodically_backfills_missing_rows(tmp_path: Path) -> None:
    _reset_singletons()
    store = MemoryStore(db_path=tmp_path / "memories.db")
    entry = store.append_memory(content="old row", tags=[], importance=2, user_id="default")

    provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=(0.5).hex().encode("utf-8"),
                dimension=1,
                model="test-model",
                model_version="v1",
                vector_norm=0.5,
                provider="test",
                status="ready",
            )
        ]
    )
    worker = MemoryEmbeddingWorker(
        store=store,
        provider=provider,
        idle_sleep_s=0.01,
        rolling_backfill_interval_idle_cycles=1,
        rolling_backfill_batch_size=1,
    )

    worker.start()
    try:
        deadline = time.monotonic() + 1.5
        status = None
        while time.monotonic() < deadline:
            embedding = store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id]).get(entry.memory_id)
            status = None if embedding is None else embedding.status
            if status == "ready":
                break
            time.sleep(0.02)
        assert status == "ready"
    finally:
        worker.stop(timeout_s=0.5)


def test_remember_memory_inline_embedding_ready_when_background_disabled(tmp_path: Path, monkeypatch, caplog) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "assistant_name: Theo",
                f"var_dir: {tmp_path / 'var'}",
                "memory_semantic:",
                "  enabled: true",
                "  provider: openai",
                "  background_embedding_enabled: false",
                "  inline_embedding_on_write_when_background_disabled: true",
                "  openai:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    manager = MemoryManager.get_instance()
    manager._store = MemoryStore(db_path=tmp_path / "memories.db")
    manager._default_scope = MemoryScope.USER_GLOBAL
    manager._embedding_provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=(0.75).hex().encode("utf-8"),
                dimension=1,
                model="inline-model",
                model_version="v1",
                vector_norm=0.75,
                provider="test",
                status="ready",
            )
        ]
    )

    caplog.set_level("INFO")
    entry = manager.remember_memory(content="inline this memory", importance=3)

    embedding = manager._store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert embedding.status == "ready"
    assert embedding.model_id == "inline-model"
    assert f"memory_embedding_audit event=inline-attempted memory_id={entry.memory_id}" in caplog.text
    assert "mode=inline outcome=success" in caplog.text


def test_remember_memory_inline_embedding_records_failure_when_background_disabled(tmp_path: Path, monkeypatch, caplog) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "assistant_name: Theo",
                f"var_dir: {tmp_path / 'var'}",
                "memory_semantic:",
                "  enabled: true",
                "  provider: openai",
                "  background_embedding_enabled: false",
                "  inline_embedding_on_write_when_background_disabled: true",
                "  openai:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    manager = MemoryManager.get_instance()
    manager._store = MemoryStore(db_path=tmp_path / "memories.db")
    manager._default_scope = MemoryScope.USER_GLOBAL
    manager._embedding_provider = _SequenceProvider(
        results=[
            EmbeddingResult(
                vector=b"",
                dimension=0,
                model="inline-model",
                model_version=None,
                vector_norm=None,
                provider="test",
                status="error",
                error_code="provider_request_failed",
                error_message="boom",
            )
        ]
    )

    caplog.set_level("INFO")
    entry = manager.remember_memory(content="still remember this", importance=3)

    stored_entry = manager._store.search_memories(limit=1)[0]
    assert stored_entry.memory_id == entry.memory_id
    embedding = manager._store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert embedding.status == "error"
    assert embedding.error == "inline_embedding_provider_request_failed"
    assert "mode=inline outcome=failure error_code=provider_request_failed" in caplog.text


def test_remember_memory_inline_embedding_remains_disabled_by_default(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "\n".join(
            [
                "assistant_name: Theo",
                f"var_dir: {tmp_path / 'var'}",
                "memory_semantic:",
                "  enabled: true",
                "  provider: openai",
                "  background_embedding_enabled: false",
                "  openai:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    manager = MemoryManager.get_instance()
    manager._store = MemoryStore(db_path=tmp_path / "memories.db")
    manager._default_scope = MemoryScope.USER_GLOBAL

    entry = manager.remember_memory(content="no inline embedding", importance=3)

    embedding = manager._store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])
    assert embedding == {}
