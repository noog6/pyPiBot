"""Factory helpers for constructing persistent storage classes consistently."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import ConfigController
from storage.controller import StorageController
from storage.memories import MemoryStore
from storage.research_budget import ResearchBudgetStorage
from storage.user_profiles import UserProfileStore


def _load_config() -> dict[str, Any]:
    return ConfigController.get_instance().get_config()


def resolve_storage_var_dir(config: dict[str, Any] | None = None) -> Path:
    """Resolve the canonical var directory used by SQLite-backed stores."""

    resolved_config = config if config is not None else _load_config()
    storage_config = resolved_config.get("storage") if isinstance(resolved_config, dict) else {}
    if not isinstance(storage_config, dict):
        storage_config = {}
    var_dir = storage_config.get("var_dir", resolved_config.get("var_dir", "./var/"))
    return Path(str(var_dir)).expanduser()


def create_memory_store(*, db_path: Path | None = None) -> MemoryStore:
    """Create a memory store using canonical config/path resolution."""

    resolved_path = db_path if db_path is not None else resolve_storage_var_dir() / "memories.db"
    return MemoryStore(db_path=resolved_path)


def create_user_profile_store(*, db_path: Path | None = None) -> UserProfileStore:
    """Create a user profile store using canonical config/path resolution."""

    resolved_path = db_path if db_path is not None else resolve_storage_var_dir() / "user_profiles.db"
    return UserProfileStore(db_path=resolved_path)


def create_research_budget_store(
    *,
    storage_controller: StorageController | None = None,
) -> ResearchBudgetStorage:
    """Create a research budget store bound to the shared storage controller connection."""

    controller = storage_controller or StorageController.get_instance()
    return ResearchBudgetStorage(storage_controller=controller)

