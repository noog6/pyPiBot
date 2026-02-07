"""Manage active user profile data for personalization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from config import ConfigController
from services.memory_manager import MemoryManager
from services.reflection_manager import ReflectionManager
from storage.user_profiles import UserProfile, UserProfileStore


@dataclass(frozen=True)
class ProfileContext:
    """Snapshot of the active profile for AI personalization."""

    profile: UserProfile

    def to_instruction_block(self) -> str:
        preferences = (
            json.dumps(self.profile.preferences, indent=2, sort_keys=True)
            if self.profile.preferences
            else "None"
        )
        favorites = ", ".join(self.profile.favorites) if self.profile.favorites else "None"
        last_seen = self.profile.last_seen if self.profile.last_seen is not None else "Unknown"
        name = self.profile.name or "Unknown"
        return (
            "User profile context:\n"
            f"- id: {self.profile.user_id}\n"
            f"- name: {name}\n"
            f"- preferences: {preferences}\n"
            f"- favorites: {favorites}\n"
            f"- last_seen: {last_seen}\n"
            "Use this info to personalize responses. If data is missing, ask politely.\n"
        )


class ProfileManager:
    """Singleton manager for the active user profile."""

    _instance: "ProfileManager | None" = None

    def __init__(self) -> None:
        if ProfileManager._instance is not None:
            raise RuntimeError("You cannot create another ProfileManager class")

        config = ConfigController.get_instance().get_config()
        self._active_user_id = config.get("active_user_id", "default")
        self._store = UserProfileStore()
        ProfileManager._instance = self

    @classmethod
    def get_instance(cls) -> "ProfileManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_active_user_id(self, user_id: str) -> None:
        self._active_user_id = user_id
        MemoryManager.get_instance().set_active_user_id(user_id)
        ReflectionManager.get_instance().set_active_user_id(user_id)

    def get_active_user_id(self) -> str:
        return self._active_user_id

    def load_active_profile(self) -> UserProfile:
        profile = self._store.get_profile(self._active_user_id)
        if profile is None:
            profile = UserProfile(
                user_id=self._active_user_id,
                name=None,
                preferences={},
                last_seen=None,
                favorites=[],
            )
        profile = self._store.touch_last_seen(profile.user_id)
        return profile

    def update_active_profile_fields(self, **updates: Any) -> UserProfile:
        return self._store.update_profile_fields(self._active_user_id, **updates)

    def get_profile_context(self) -> ProfileContext:
        return ProfileContext(profile=self.load_active_profile())
