"""Focused regressions for seam-local look-center motion request handling."""

from __future__ import annotations

from types import SimpleNamespace

from services import tool_runtime


class _FakeMotionController:
    def __init__(self, *, pan: float = 0.0, tilt: float = 0.0, moving: bool = False, current_action=None, action_queue=None) -> None:
        self.current_servo_position = {"pan": pan, "tilt": tilt}
        self._moving = moving
        self.current_action = current_action
        self.action_queue = list(action_queue or [])
        self.added_actions = []

    def is_moving(self) -> bool:
        return self._moving

    def add_action_to_queue(self, action) -> None:
        self.added_actions.append(action)
        self.action_queue.append(action)


def test_repeated_center_request_when_already_centered_returns_satisfied(monkeypatch) -> None:
    controller = _FakeMotionController(pan=0.0, tilt=0.0, moving=False, current_action=None, action_queue=[])
    monkeypatch.setattr(tool_runtime.MotionController, "get_instance", classmethod(lambda cls: controller))

    result = tool_runtime.enqueue_look_center_gesture()

    assert result["queued"] is False
    assert result["motion_request_state"] == "already_satisfied"
    assert result["message"] == "Already centered."
    assert result["tool_result_has_distinct_info"] is True
    assert controller.added_actions == []


def test_repeated_center_request_while_center_motion_active_returns_in_progress(monkeypatch) -> None:
    controller = _FakeMotionController(
        pan=12.0,
        tilt=0.0,
        moving=True,
        current_action=SimpleNamespace(name="gesture_look_center"),
        action_queue=[],
    )
    monkeypatch.setattr(tool_runtime.MotionController, "get_instance", classmethod(lambda cls: controller))

    result = tool_runtime.enqueue_look_center_gesture()

    assert result["queued"] is False
    assert result["motion_request_state"] == "in_progress"
    assert result["message"] == "Already moving back to center."
    assert result["tool_result_has_distinct_info"] is True
    assert controller.added_actions == []


def test_repeated_center_request_when_center_already_queued_returns_queued(monkeypatch) -> None:
    controller = _FakeMotionController(
        pan=18.0,
        tilt=0.0,
        moving=True,
        current_action=SimpleNamespace(name="gesture_look_left"),
        action_queue=[SimpleNamespace(name="gesture_look_center")],
    )
    monkeypatch.setattr(tool_runtime.MotionController, "get_instance", classmethod(lambda cls: controller))

    result = tool_runtime.enqueue_look_center_gesture()

    assert result["queued"] is False
    assert result["motion_request_state"] == "queued"
    assert result["message"] == "Center motion is already queued."
    assert result["tool_result_has_distinct_info"] is True
    assert controller.added_actions == []


def test_new_center_request_still_queues_motion(monkeypatch) -> None:
    queued_action = SimpleNamespace(name="gesture_look_center")
    controller = _FakeMotionController(
        pan=22.0,
        tilt=-3.0,
        moving=False,
        current_action=None,
        action_queue=[],
    )
    monkeypatch.setattr(tool_runtime.MotionController, "get_instance", classmethod(lambda cls: controller))
    monkeypatch.setattr(tool_runtime, "gesture_look_center", lambda delay_ms=0: queued_action)

    result = tool_runtime.enqueue_look_center_gesture(delay_ms=25)

    assert result["queued"] is True
    assert result["motion_request_state"] == "new_request"
    assert result["message"] == "Moving back to center."
    assert "tool_result_has_distinct_info" not in result
    assert controller.added_actions == [queued_action]
