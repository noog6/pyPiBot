"""Tests for memory retention/maintenance helpers."""

from __future__ import annotations

from pathlib import Path

from config.controller import ConfigController
from storage.memories import MemoryStore


NOW_MS = 2_000_000_000_000
DAY_MS = 86_400_000


def _reset_singletons() -> None:
    ConfigController._instance = None


def _configure(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        """
assistant_name: Theo
memory:
  retention:
    importance_tiers:
      low: [1, 2]
      medium: [3, 4]
      high: [5]
    max_age_days_by_source:
      manual_tool:
        low: 10
      auto_reflection:
        low: 4
        medium: 20
    protect_review_approved_strategic: true
    optimize_min_deleted_rows: 2
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()


def test_prune_policy_respects_source_tier_and_protections(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    store = MemoryStore(db_path=tmp_path / "memories.db")

    old_manual = NOW_MS - (15 * DAY_MS)
    old_auto = NOW_MS - (6 * DAY_MS)
    fresh_auto = NOW_MS - (2 * DAY_MS)

    removable = store.append_memory(
        content="old low manual",
        tags=["prefs"],
        importance=2,
        source="manual_tool",
        timestamp=old_manual,
    )
    pinned = store.append_memory(
        content="old low manual pinned",
        tags=["prefs"],
        importance=2,
        source="manual_tool",
        pinned=True,
        timestamp=old_manual,
    )
    strategic = store.append_memory(
        content="old auto strategic",
        tags=["strategic"],
        importance=2,
        source="auto_reflection",
        needs_review=False,
        timestamp=old_auto,
    )
    fresh = store.append_memory(
        content="fresh auto",
        tags=["prefs"],
        importance=2,
        source="auto_reflection",
        timestamp=fresh_auto,
    )

    deleted = store.prune_memories_by_retention_policy(now_ms=NOW_MS)
    assert deleted == 1

    remaining = {entry.memory_id for entry in store.search_memories(limit=20)}
    assert removable.memory_id not in remaining
    assert pinned.memory_id in remaining
    assert strategic.memory_id in remaining
    assert fresh.memory_id in remaining


def test_force_prune_includes_pinned_and_strategic(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    store = MemoryStore(db_path=tmp_path / "memories.db")

    old_manual = NOW_MS - (20 * DAY_MS)
    old_auto = NOW_MS - (8 * DAY_MS)

    pinned = store.append_memory(
        content="pinned old",
        tags=["prefs"],
        importance=2,
        source="manual_tool",
        pinned=True,
        timestamp=old_manual,
    )
    strategic = store.append_memory(
        content="strategic reviewed",
        tags=["strategic"],
        importance=2,
        source="auto_reflection",
        needs_review=False,
        timestamp=old_auto,
    )

    deleted = store.prune_memories_by_retention_policy(now_ms=NOW_MS, force=True)
    assert deleted == 2

    remaining = store.search_memories(limit=10)
    assert pinned.memory_id not in {entry.memory_id for entry in remaining}
    assert strategic.memory_id not in {entry.memory_id for entry in remaining}


def test_purge_orphan_embeddings_and_optimize_guard(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)
    store = MemoryStore(db_path=tmp_path / "memories.db")

    keep_entry = store.append_memory(content="kept", tags=["x"], importance=3, source="manual_tool")
    store.upsert_memory_embedding(memory_id=keep_entry.memory_id, model_id="m", dim=1, vector=b"x")
    store.upsert_memory_embedding(memory_id=999, model_id="m", dim=1, vector=b"y")

    purged = store.purge_orphan_embeddings()
    assert purged == 1
    assert set(store.fetch_embeddings_for_memories(memory_ids=[keep_entry.memory_id, 999]).keys()) == {
        keep_entry.memory_id
    }

    assert store.maybe_optimize_storage(deleted_rows=1, force=False) is False
    assert store.maybe_optimize_storage(deleted_rows=2, force=False) is True


def test_periodic_maintenance_noop_returns_zero_counters(tmp_path: Path, monkeypatch) -> None:
    _configure(tmp_path, monkeypatch)

    from services.memory_manager import MemoryManager

    MemoryManager._instance = None
    manager = MemoryManager.get_instance()

    stats = manager.run_periodic_maintenance(optimize_allowed=True)

    assert stats == {
        "pruned_rows": 0,
        "purged_rows": 0,
        "optimize_triggered": False,
    }
