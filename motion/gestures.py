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


def gesture_no(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a head shake gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_no", delay_ms=delay_ms, intensity=intensity)


def gesture_look_around(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a casual look around gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_look_around", delay_ms=delay_ms, intensity=intensity)


def gesture_look_up(
    delay_ms: int = 0,
    intensity: float = 1.0,
    style: str = "neutral",
) -> Action:
    """Build a look up gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action(
        "gesture_look_up",
        delay_ms=delay_ms,
        intensity=intensity,
        style=style,
    )


def gesture_look_left(
    delay_ms: int = 0,
    intensity: float = 1.0,
    style: str = "neutral",
) -> Action:
    """Build a look left gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action(
        "gesture_look_left",
        delay_ms=delay_ms,
        intensity=intensity,
        style=style,
    )


def gesture_look_right(
    delay_ms: int = 0,
    intensity: float = 1.0,
    style: str = "neutral",
) -> Action:
    """Build a look right gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action(
        "gesture_look_right",
        delay_ms=delay_ms,
        intensity=intensity,
        style=style,
    )


def gesture_look_down(
    delay_ms: int = 0,
    intensity: float = 1.0,
    style: str = "neutral",
) -> Action:
    """Build a look down gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action(
        "gesture_look_down",
        delay_ms=delay_ms,
        intensity=intensity,
        style=style,
    )


def gesture_look_center(delay_ms: int = 0, style: str = "neutral") -> Action:
    """Build a look center gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_look_center", delay_ms=delay_ms, style=style)


def gesture_curious_tilt(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a curious tilt gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_curious_tilt", delay_ms=delay_ms, intensity=intensity)



def gesture_speaking_posture(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a restrained speaking posture gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_speaking_posture", delay_ms=delay_ms, intensity=intensity)


def gesture_speaking_settle(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a restrained settle gesture action after speaking."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_speaking_settle", delay_ms=delay_ms, intensity=intensity)


def gesture_attention_hold(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a sustained attention hold gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_attention_hold", delay_ms=delay_ms, intensity=intensity)


def gesture_attention_release(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a subtle release back toward neutral after attention hold."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_attention_release", delay_ms=delay_ms, intensity=intensity)


def gesture_attention_snap(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a quick attention snap gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_attention_snap", delay_ms=delay_ms, intensity=intensity)


def gesture_startup_presence(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build the lifecycle startup presence gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_startup_presence", delay_ms=delay_ms, intensity=intensity)


def gesture_shutdown_rest(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build the lifecycle shutdown rest gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_shutdown_rest", delay_ms=delay_ms, intensity=intensity)
