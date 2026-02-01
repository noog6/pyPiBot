"""Manage reflection entries for instruction context."""

from __future__ import annotations

from dataclasses import dataclass

from config import ConfigController
from storage.reflections import ReflectionEntry, ReflectionStore


@dataclass(frozen=True)
class ReflectionContext:
    """Snapshot of recent reflection lessons."""

    user_id: str | None
    session_id: str | None
    recent_lessons: list[str]

    def to_instruction_block(self) -> str:
        lessons = "\n".join(f"- {lesson}" for lesson in self.recent_lessons)
        lesson_block = lessons if lessons else "None"
        user_id = self.user_id or "Unknown"
        session_id = self.session_id or "Unknown"
        return (
            "Reflection context:\n"
            f"- user_id: {user_id}\n"
            f"- session_id: {session_id}\n"
            f"- recent_lessons:\n{lesson_block}\n"
            "Use these lessons to improve guidance and avoid repeat mistakes."
        )


class ReflectionManager:
    """Singleton manager for reflection storage access."""

    _instance: "ReflectionManager | None" = None

    def __init__(self) -> None:
        if ReflectionManager._instance is not None:
            raise RuntimeError("You cannot create another ReflectionManager class")

        config = ConfigController.get_instance().get_config()
        self._active_user_id = config.get("active_user_id", "default")
        self._active_session_id = config.get("active_session_id")
        self._store = ReflectionStore()
        ReflectionManager._instance = self

    @classmethod
    def get_instance(cls) -> "ReflectionManager":
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

    def append_reflection(
        self,
        *,
        summary: str,
        lessons: list[str],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> ReflectionEntry:
        return self._store.append_reflection(
            summary=summary,
            lessons=lessons,
            user_id=user_id if user_id is not None else self._active_user_id,
            session_id=session_id if session_id is not None else self._active_session_id,
        )

    def get_recent_lessons(self, *, limit: int = 5) -> list[str]:
        return self._store.get_recent_lessons(
            limit=limit,
            user_id=self._active_user_id,
            session_id=self._active_session_id,
        )

    def get_reflection_context(self, *, limit: int = 5) -> ReflectionContext:
        return ReflectionContext(
            user_id=self._active_user_id,
            session_id=self._active_session_id,
            recent_lessons=self.get_recent_lessons(limit=limit),
        )
