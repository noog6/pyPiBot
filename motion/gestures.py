"""Gesture builders for motion sequences."""

from __future__ import annotations

from motion.action import Action
from motion.gesture_library import GestureLibrary


def gesture_idle(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a gentle idle gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_idle", delay_ms=delay_ms, intensity=intensity)


def gesture_nod(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a nod gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_nod", delay_ms=delay_ms, intensity=intensity)
