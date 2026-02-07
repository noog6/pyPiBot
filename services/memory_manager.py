"""Manage memory entries for long-term recall."""

from __future__ import annotations

from dataclasses import dataclass

from config import ConfigController
from storage.memories import MemoryEntry, MemoryStore

MAX_CONTENT_LENGTH = 400
MAX_TAGS = 6
MAX_TAG_LENGTH = 24
MAX_RECALL_LIMIT = 10
MIN_IMPORTANCE = 1
MAX_IMPORTANCE = 5


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


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

    @classmethod
    def from_entry(cls, entry: MemoryEntry) -> "MemorySummary":
        return cls(
            memory_id=entry.memory_id,
            content=entry.content,
            tags=entry.tags,
            importance=entry.importance,
        )


class MemoryManager:
    """Singleton manager for memory storage access."""

    _instance: "MemoryManager | None" = None

    def __init__(self) -> None:
        if MemoryManager._instance is not None:
            raise RuntimeError("You cannot create another MemoryManager class")

        config = ConfigController.get_instance().get_config()
        self._active_user_id = config.get("active_user_id", "default")
        self._active_session_id = config.get("active_session_id")
        self._store = MemoryStore()
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

    def remember_memory(
        self,
        *,
        content: str,
        tags: list[str] | None = None,
        importance: int = 3,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> MemoryEntry:
        normalized_content = _normalize_content(content)
        normalized_tags = _normalize_tags(tags)
        bounded_importance = _clamp(importance, MIN_IMPORTANCE, MAX_IMPORTANCE)
        return self._store.append_memory(
            content=normalized_content,
            tags=normalized_tags,
            importance=bounded_importance,
            user_id=user_id if user_id is not None else self._active_user_id,
            session_id=session_id if session_id is not None else self._active_session_id,
        )

    def recall_memories(
        self,
        *,
        query: str | None = None,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[MemorySummary]:
        bounded_limit = _clamp(limit, 1, MAX_RECALL_LIMIT)
        entries = self._store.search_memories(
            query=query,
            limit=bounded_limit,
            user_id=self._active_user_id,
            session_id=session_id,
        )
        return [MemorySummary.from_entry(entry) for entry in entries]

    def forget_memory(self, *, memory_id: int) -> bool:
        return self._store.delete_memory(memory_id=memory_id)
