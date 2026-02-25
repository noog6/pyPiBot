"""Targeted coverage for service wiring through storage helper constructors."""

from __future__ import annotations


class _FakeConfigController:
    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def get_config(self) -> dict[str, object]:
        return dict(self._config)


def test_memory_manager_initializes_via_helper_and_round_trips_memory(monkeypatch, tmp_path) -> None:
    import config.controller as config_controller

    from services.memory_manager import MemoryManager
    from storage.memories import MemoryStore

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    config = {
        "var_dir": str(var_dir),
        "log_dir": str(log_dir),
        "memory": {"default_scope": "user_global"},
        "memory_semantic": {"enabled": False, "provider": "none"},
        "active_user_id": "user-test",
    }
    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController(config),
    )

    created = {"called": False}

    def _factory() -> MemoryStore:
        created["called"] = True
        return MemoryStore(db_path=var_dir / "memories.db")

    monkeypatch.setattr("services.memory_manager.create_memory_store", _factory)

    MemoryManager._instance = None
    manager = MemoryManager.get_instance()

    try:
        assert created["called"] is True
        manager.remember_memory(content="Prefers jasmine tea", tags=["preference"], importance=4, pinned=True)
        digest = manager.retrieve_startup_digest(max_items=2, max_chars=280)
        assert digest is not None
        assert any(item.content == "Prefers jasmine tea" for item in digest.items)
    finally:
        manager.close()
        MemoryManager._instance = None


def test_profile_manager_initializes_via_helper_and_round_trips_profile(monkeypatch, tmp_path) -> None:
    import config.controller as config_controller

    from services.profile_manager import ProfileManager
    from storage.user_profiles import UserProfileStore

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController(
            {
                "var_dir": str(var_dir),
                "log_dir": str(log_dir),
                "active_user_id": "profile-user",
            }
        ),
    )

    created = {"called": False}

    def _factory() -> UserProfileStore:
        created["called"] = True
        return UserProfileStore(db_path=var_dir / "user_profiles.db")

    monkeypatch.setattr("services.profile_manager.create_user_profile_store", _factory)

    ProfileManager._instance = None
    manager = ProfileManager.get_instance()
    try:
        assert created["called"] is True
        manager.update_active_profile_fields(name="Ada", favorites=["oolong"])
        profile = manager.load_active_profile()
        assert profile.name == "Ada"
        assert profile.favorites == ["oolong"]
        assert profile.last_seen is not None
    finally:
        ProfileManager._instance = None


def test_research_budget_manager_initializes_via_helper_and_round_trips_state(monkeypatch, tmp_path) -> None:
    import config.controller as config_controller

    from services.research.budget_manager import ResearchBudgetManager
    from storage.controller import StorageController
    from storage.research_budget import ResearchBudgetStorage

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController({"var_dir": str(var_dir), "log_dir": str(log_dir)}),
    )

    created = {"called": False}

    def _factory() -> ResearchBudgetStorage:
        created["called"] = True
        return ResearchBudgetStorage(storage_controller=StorageController.get_instance())

    monkeypatch.setattr("services.research.budget_manager.create_research_budget_store", _factory)

    StorageController._instance = None
    manager = ResearchBudgetManager("unused.json", daily_limit=2)
    try:
        assert created["called"] is True
        assert manager.spend_if_allowed(1, audit_payload={"request_fingerprint": "fp-test"})
        state = manager.current_state()
        assert state["remaining"] == 1
        assert state["last_audit"]["request_fingerprint"] == "fp-test"
    finally:
        controller = StorageController._instance
        if controller is not None:
            controller.close()
        StorageController._instance = None


def test_memory_embedding_worker_initializes_via_helper(monkeypatch, tmp_path) -> None:
    import config.controller as config_controller

    from services.embedding_provider import EmbeddingResult
    from services.memory_embedding_worker import MemoryEmbeddingWorker
    from storage.memories import MemoryStore

    var_dir = tmp_path / "var"
    log_dir = tmp_path / "log"
    monkeypatch.setattr(
        config_controller.ConfigController,
        "get_instance",
        lambda: _FakeConfigController(
            {
                "var_dir": str(var_dir),
                "log_dir": str(log_dir),
                "memory_semantic": {"max_embedding_retries": 2},
            }
        ),
    )

    created = {"called": False}

    def _factory() -> MemoryStore:
        created["called"] = True
        return MemoryStore(db_path=var_dir / "memories.db")

    class _Provider:
        def embed_batch(self, texts):  # noqa: ANN001
            return [
                EmbeddingResult(
                    vector=(1.0).hex().encode("utf-8"),
                    dimension=1,
                    model="test-model",
                    model_version="v1",
                    vector_norm=1.0,
                    provider="test",
                    status="ready",
                )
                for _ in texts
            ]

    monkeypatch.setattr("services.memory_embedding_worker.create_memory_store", _factory)

    worker = MemoryEmbeddingWorker(provider=_Provider())
    assert created["called"] is True
