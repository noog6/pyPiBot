"""Focused tests for gesture motion acceptance vs physical completion tracking."""

from __future__ import annotations

from types import SimpleNamespace

from services import tool_runtime


class _FakeMotionController:
    def __init__(self) -> None:
        self.added_actions = []
        self._lifecycle_callback = None

    def add_action_to_queue(self, action) -> None:
        self.added_actions.append(action)

    def register_action_lifecycle_callback(self, callback) -> None:
        self._lifecycle_callback = callback

    def emit(self, event: str, action) -> None:
        if callable(self._lifecycle_callback):
            self._lifecycle_callback(event, action)


def _reset_motion_tracking_state() -> None:
    tool_runtime._motion_state_by_request_key.clear()
    tool_runtime._motion_request_key_by_tool_call_id.clear()
    tool_runtime._motion_callbacks_registered_for_controller_id = None


def test_gesture_queue_state_is_tracked_before_completion(monkeypatch) -> None:
    _reset_motion_tracking_state()
    controller = _FakeMotionController()
    action = SimpleNamespace(name="gesture_nod", timestamp=123)
    monkeypatch.setattr(tool_runtime.MotionController, "get_instance", classmethod(lambda cls: controller))
    monkeypatch.setattr(tool_runtime, "gesture_nod", lambda delay_ms=0, intensity=1.0: action)

    result = tool_runtime.enqueue_nod_gesture(delay_ms=25, intensity=0.5)

    request_key = str(result.get("motion_request_key") or "")
    assert result["queued"] is True
    assert request_key
    state = tool_runtime._motion_state_by_request_key[request_key]
    assert state["status"] == "queued"
    assert state["queued_monotonic_s"] is not None
    assert state["started_monotonic_s"] is None
    assert state["completed_monotonic_s"] is None


def test_gesture_motion_state_advances_to_completed_when_controller_signals_finish(monkeypatch) -> None:
    _reset_motion_tracking_state()
    controller = _FakeMotionController()
    action = SimpleNamespace(name="gesture_no", timestamp=456)
    monkeypatch.setattr(tool_runtime.MotionController, "get_instance", classmethod(lambda cls: controller))
    monkeypatch.setattr(tool_runtime, "gesture_no", lambda delay_ms=0, intensity=1.0: action)

    result = tool_runtime.enqueue_no_gesture()
    request_key = str(result.get("motion_request_key") or "")
    tool_runtime.register_tool_call_motion_request(tool_call_id="call-gesture-1", motion_request_key=request_key)

    accepted_state = tool_runtime.get_tool_call_motion_state("call-gesture-1")
    assert accepted_state is not None
    assert accepted_state["status"] == "queued"
    assert "_action_obj_id" not in accepted_state
    assert tool_runtime.is_tool_call_motion_completed("call-gesture-1") is False

    controller.emit("started", action)
    started_state = tool_runtime.get_tool_call_motion_state("call-gesture-1")
    assert started_state is not None
    assert started_state["status"] == "started"
    assert started_state["started_monotonic_s"] is not None

    controller.emit("completed", action)
    completed_state = tool_runtime.get_tool_call_motion_state("call-gesture-1")
    assert completed_state is not None
    assert completed_state["status"] == "completed"
    assert completed_state["completed_monotonic_s"] is not None
    assert tool_runtime.is_tool_call_motion_completed("call-gesture-1") is True


def test_same_name_and_timestamp_gestures_get_distinct_request_keys(monkeypatch) -> None:
    _reset_motion_tracking_state()
    controller = _FakeMotionController()
    action_one = SimpleNamespace(name="gesture_nod", timestamp=999)
    action_two = SimpleNamespace(name="gesture_nod", timestamp=999)
    actions = [action_one, action_two]
    monkeypatch.setattr(tool_runtime.MotionController, "get_instance", classmethod(lambda cls: controller))
    monkeypatch.setattr(tool_runtime, "gesture_nod", lambda delay_ms=0, intensity=1.0: actions.pop(0))

    first = tool_runtime.enqueue_nod_gesture()
    second = tool_runtime.enqueue_nod_gesture()

    first_key = str(first.get("motion_request_key") or "")
    second_key = str(second.get("motion_request_key") or "")
    assert first_key
    assert second_key
    assert first_key != second_key
    assert tool_runtime._motion_state_by_request_key[first_key]["status"] == "queued"
    assert tool_runtime._motion_state_by_request_key[second_key]["status"] == "queued"
