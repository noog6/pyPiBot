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


def test_remember_memory_enqueues_pending_embedding_when_enabled(tmp_path: Path, monkeypatch) -> None:
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

    entry = manager.remember_memory(content="queue this memory", importance=3)

    embedding = manager._store.fetch_embeddings_for_memories(memory_ids=[entry.memory_id])[entry.memory_id]
    assert embedding.status == "pending"


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
