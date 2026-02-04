"""Gesture builders for motion sequences."""

from __future__ import annotations

from motion.action import Action
from motion.gesture_library import GestureLibrary
from motion.motion_controller import MotionController, millis


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


def gesture_look_up(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a look up gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_look_up", delay_ms=delay_ms, intensity=intensity)


def gesture_look_left(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a look left gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_look_left", delay_ms=delay_ms, intensity=intensity)


def gesture_look_right(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a look right gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_look_right", delay_ms=delay_ms, intensity=intensity)


def gesture_look_down(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a look down gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_look_down", delay_ms=delay_ms, intensity=intensity)


def gesture_look_center(delay_ms: int = 0) -> Action:
    """Build a look center gesture action."""

    controller = MotionController.get_instance()
    frame = controller.generate_base_keyframe(pan_degrees=0, tilt_degrees=0)
    frame.name = "look-center"
    return Action(
        priority=2,
        timestamp=millis() + delay_ms,
        name="gesture_look_center",
        frames=frame,
    )


def gesture_curious_tilt(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a curious tilt gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_curious_tilt", delay_ms=delay_ms, intensity=intensity)


def gesture_attention_snap(delay_ms: int = 0, intensity: float = 1.0) -> Action:
    """Build a quick attention snap gesture action."""

    library = GestureLibrary.get_instance()
    return library.build_action("gesture_attention_snap", delay_ms=delay_ms, intensity=intensity)
