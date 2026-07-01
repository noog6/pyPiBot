"""Pose-aware softening coverage for automatic PandaTheo gestures."""

from __future__ import annotations

from types import SimpleNamespace

from motion.gesture_library import GestureLibrary, SOFTEN_TILT_LIMIT_DEGREES


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
        frame = SimpleNamespace(
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
        return frame


def _library_for_pose(monkeypatch, pose: dict[str, float]) -> GestureLibrary:
    import motion.gesture_library as gesture_library

    library = GestureLibrary.__new__(GestureLibrary)
    library._definitions = {definition.name: definition for definition in gesture_library.DEFAULT_GESTURES}
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


def test_automatic_micro_presence_idle_near_high_tilt_softens_to_ears_only(monkeypatch) -> None:
    library = _library_for_pose(
        monkeypatch,
        {"pan": 7.0, "tilt": 44.0, "roll": 2.0, "ear_left": 0.0, "ear_right": 0.0},
    )

    import motion.gesture_library as gesture_library

    log_messages = []
    monkeypatch.setattr(
        gesture_library.LOGGER,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    action = library.build_action("gesture_idle", source="micro_presence")

    frames = _frames(action)
    assert [frame.servo_destination["pan"] for frame in frames] == [7.0, 7.0, 7.0]
    assert [frame.servo_destination["tilt"] for frame in frames] == [44.0, 44.0, 44.0]
    assert [frame.servo_destination["roll"] for frame in frames] == [0.0, 0.0, 0.0]
    assert frames[0].servo_destination["ear_left"] > 0.0
    assert frames[1].servo_destination["ear_right"] > 0.0
    assert frames[-1].servo_destination["ear_left"] == 0.0
    assert any(
        "gesture_softened reason=near_tilt_limit source=micro_presence gesture=gesture_idle current_tilt=44.0 policy=ears_only"
        in message
        for message in log_messages
    )


def test_automatic_speaking_posture_near_low_tilt_softens_to_ears_only(monkeypatch) -> None:
    library = _library_for_pose(
        monkeypatch,
        {"pan": -3.0, "tilt": -44.0, "roll": -1.0, "ear_left": 0.0, "ear_right": 0.0},
    )

    frames = _frames(library.build_action("gesture_speaking_posture", source="state_cue"))

    assert all(frame.servo_destination["pan"] == -3.0 for frame in frames)
    assert all(frame.servo_destination["tilt"] == -44.0 for frame in frames)
    assert frames[0].servo_destination["roll"] == 0.0
    assert frames[0].servo_destination["ear_left"] != 0.0
    assert frames[-1].servo_destination["ear_left"] == 0.0
    assert frames[-1].servo_destination["ear_right"] == 0.0


def test_automatic_absolute_target_gesture_is_not_softened(monkeypatch) -> None:
    library = _library_for_pose(monkeypatch, {"pan": 7.0, "tilt": 44.0, "roll": 2.0})

    frame = library.build_action("gesture_look_down", source="state_cue").frames

    assert frame.servo_destination["pan"] == 0.0
    assert frame.servo_destination["tilt"] == -44.0
    assert frame.servo_destination["roll"] == 0.0


def test_explicit_look_down_is_not_softened(monkeypatch) -> None:
    library = _library_for_pose(monkeypatch, {"pan": 0.0, "tilt": 44.0, "roll": 2.0})

    frame = library.build_action("gesture_look_down", source="user").frames

    assert frame.servo_destination["tilt"] == -44.0


def test_explicit_no_is_not_softened_near_limit(monkeypatch) -> None:
    library = _library_for_pose(monkeypatch, {"pan": 1.0, "tilt": 44.0, "roll": 2.0})

    frames = _frames(library.build_action("gesture_no", source="user"))

    assert frames[0].servo_destination["pan"] < 1.0
    assert frames[0].servo_destination["roll"] > 2.0


def test_normal_automatic_idle_away_from_limits_is_unchanged(monkeypatch) -> None:
    library = _library_for_pose(monkeypatch, {"pan": 0.0, "tilt": 25.0, "roll": 0.0})

    frames = _frames(library.build_action("gesture_idle", source="micro_presence"))

    assert frames[0].servo_destination["pan"] == -4.0
    assert frames[0].servo_destination["tilt"] == 27.5
    assert frames[0].servo_destination["roll"] == 1.5


def test_softening_threshold_is_named_constant() -> None:
    assert SOFTEN_TILT_LIMIT_DEGREES == 35.0
