"""Explicit expressive gesture tilt-envelope shaping coverage."""

from __future__ import annotations

from types import SimpleNamespace

from motion.body_model import EXPRESSIVE_TILT_SOFT_LIMIT_DEGREES
from motion.gesture_library import GestureLibrary


class _FakeController:
    def __init__(self, pose: dict[str, float]) -> None:
        self._pose = pose
        servo = lambda: SimpleNamespace(min_angle=-45.0, max_angle=45.0)
        self.servo_registry = SimpleNamespace(
            servos={
                "pan": servo(),
                "tilt_left": servo(),
                "tilt_right": servo(),
                "roll": servo(),
                "ear_left": servo(),
                "ear_right": servo(),
            }
        )

    def get_current_logical_pose(self) -> dict[str, float]:
        return dict(self._pose)

    def generate_base_keyframe(
        self,
        pan_degrees: float,
        tilt_degrees: float,
        roll_degrees: float = 0.0,
        ear_left_degrees: float = 0.0,
        ear_right_degrees: float = 0.0,
    ):
        return SimpleNamespace(
            servo_destination={
                "pan": pan_degrees,
                "tilt": tilt_degrees,
                "roll": roll_degrees,
                "ear_left": ear_left_degrees,
                "ear_right": ear_right_degrees,
            },
            name="",
            final_target_time=0,
            next=None,
        )


def _library_for_pose(monkeypatch, pose: dict[str, float]) -> GestureLibrary:
    import motion.gesture_library as gesture_library

    library = GestureLibrary.__new__(GestureLibrary)
    library._definitions = {
        definition.name: definition for definition in gesture_library.DEFAULT_GESTURES
    }
    controller = _FakeController(pose)
    monkeypatch.setattr(gesture_library.MotionController, "get_instance", lambda: controller)
    return library


def _frames(action) -> list:
    frames = []
    frame = action.frames
    while frame is not None:
        frames.append(frame)
        frame = frame.next
    return frames


def _assert_composed_tilt_within_envelope(frame) -> None:
    tilt = frame.servo_destination["tilt"]
    roll = frame.servo_destination["roll"]
    assert abs(tilt + roll) <= EXPRESSIVE_TILT_SOFT_LIMIT_DEGREES
    assert abs(tilt - roll) <= EXPRESSIVE_TILT_SOFT_LIMIT_DEGREES


def test_explicit_no_at_high_tilt_preserves_pan_and_ears_but_shapes_roll(monkeypatch) -> None:
    import motion.gesture_library as gesture_library

    library = _library_for_pose(
        monkeypatch,
        {"pan": 0.0, "tilt": 44.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    )
    log_messages = []
    monkeypatch.setattr(
        gesture_library.LOGGER,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    frames = _frames(library.build_action("gesture_no", source="user"))

    assert frames[0].servo_destination["pan"] == -14.0
    assert frames[1].servo_destination["pan"] == 14.0
    assert frames[0].servo_destination["ear_left"] == 4.0
    assert frames[1].servo_destination["ear_right"] == 4.0
    assert frames[0].servo_destination["roll"] == 0.0
    assert frames[1].servo_destination["roll"] == 0.0
    for frame in frames:
        _assert_composed_tilt_within_envelope(frame)
    assert any(
        "gesture_shaped reason=tilt_envelope gesture=gesture_no frame=no-left current_tilt=44.0 requested_roll=2.5 shaped_roll=0.0 policy=preserve_pan_ears"
        in message
        for message in log_messages
    )


def test_explicit_no_near_center_keeps_authored_expression(monkeypatch) -> None:
    library = _library_for_pose(
        monkeypatch,
        {"pan": 0.0, "tilt": 25.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    )

    frames = _frames(library.build_action("gesture_no", source="user"))

    assert frames[0].servo_destination["pan"] == -14.0
    assert frames[0].servo_destination["tilt"] == 25.0
    assert frames[0].servo_destination["roll"] == 2.5
    assert frames[0].servo_destination["ear_left"] == 4.0
    assert frames[1].servo_destination["roll"] == -2.5


def test_explicit_curious_tilt_at_high_tilt_shapes_tilt_and_roll_safely(monkeypatch) -> None:
    library = _library_for_pose(
        monkeypatch,
        {"pan": 0.0, "tilt": 44.0, "roll": 0.0, "ear_left": 0.0, "ear_right": 0.0},
    )

    frames = _frames(library.build_action("gesture_curious_tilt", source="user"))

    assert frames[0].servo_destination["ear_left"] == 6.0
    assert frames[1].servo_destination["ear_right"] == 5.0
    assert frames[0].servo_destination["tilt"] == EXPRESSIVE_TILT_SOFT_LIMIT_DEGREES
    assert frames[0].servo_destination["roll"] == 0.0
    for frame in frames:
        _assert_composed_tilt_within_envelope(frame)


def test_explicit_look_down_is_exempt_from_expression_shaping(monkeypatch) -> None:
    library = _library_for_pose(monkeypatch, {"pan": 0.0, "tilt": 44.0, "roll": 0.0})

    frame = library.build_action("gesture_look_down", source="user").frames

    assert frame.servo_destination["pan"] == 0.0
    assert frame.servo_destination["tilt"] == -44.0
    assert frame.servo_destination["roll"] == 0.0
