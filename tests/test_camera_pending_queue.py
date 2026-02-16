"""Tests for camera pending image queue readiness observability."""

from __future__ import annotations

import threading
from collections import deque

from hardware.camera_controller import CameraController


class _RealtimeTupleStub:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def is_ready_for_injections(self, with_reason: bool = False):
        if with_reason:
            return False, self.reason
        return False


class _RealtimeBoolStub:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def is_ready_for_injections(self) -> bool:
        return self.ready


class _LoopStub:
    def __init__(self, running: bool) -> None:
        self._running = running

    def is_running(self) -> bool:
        return self._running


class _RealtimeLoopStub:
    def __init__(self, running: bool) -> None:
        self.loop = _LoopStub(running)


def _make_camera_stub(realtime_instance: object) -> CameraController:
    camera = CameraController.__new__(CameraController)
    camera.realtime_instance = realtime_instance
    camera._pending_images = deque(maxlen=3)
    camera._pending_lock = threading.Lock()
    return camera


def test_queue_pending_image_logs_response_in_progress_reason(monkeypatch) -> None:
    camera = _make_camera_stub(_RealtimeTupleStub("response_in_progress"))
    messages: list[str] = []
    monkeypatch.setattr("hardware.camera_controller.logger.info", lambda msg, reason: messages.append(msg % reason))

    camera._queue_pending_image("frame")

    assert list(camera._pending_images) == ["frame"]
    assert any("response_in_progress" in message for message in messages)


def test_queue_pending_image_logs_interaction_state_reason(monkeypatch) -> None:
    camera = _make_camera_stub(_RealtimeTupleStub("interaction_state=speaking"))
    messages: list[str] = []
    monkeypatch.setattr("hardware.camera_controller.logger.info", lambda msg, reason: messages.append(msg % reason))

    camera._queue_pending_image("frame")

    assert list(camera._pending_images) == ["frame"]
    assert any("interaction_state=speaking" in message for message in messages)


def test_queue_pending_image_logs_loop_not_running_reason(monkeypatch) -> None:
    camera = _make_camera_stub(_RealtimeLoopStub(running=False))
    messages: list[str] = []
    monkeypatch.setattr("hardware.camera_controller.logger.info", lambda msg, reason: messages.append(msg % reason))

    camera._queue_pending_image("frame")

    assert list(camera._pending_images) == ["frame"]
    assert any("loop_not_running" in message for message in messages)


def test_can_send_realtime_supports_bool_readiness_callable() -> None:
    not_ready_camera = _make_camera_stub(_RealtimeBoolStub(False))
    ready_camera = _make_camera_stub(_RealtimeBoolStub(True))

    assert not not_ready_camera._can_send_realtime()
    assert ready_camera._can_send_realtime()
