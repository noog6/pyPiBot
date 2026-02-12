import threading
from unittest.mock import Mock

from hardware.camera_controller import CameraController


def _bare_camera_controller() -> CameraController:
    controller = CameraController.__new__(CameraController)
    controller._send_in_flight = threading.Event()
    controller.take_main_pil = Mock(return_value=object())
    controller._save_image_async = Mock()
    controller._can_send_realtime = Mock(return_value=True)
    controller._queue_image_send = Mock(return_value=True)
    controller._queue_pending_image = Mock()
    return controller


def test_capture_dispatch_burst_frame_queues_when_send_in_flight() -> None:
    controller = _bare_camera_controller()
    controller._send_in_flight.set()

    controller._capture_and_dispatch_image(prefer_queue_if_in_flight=True)

    controller._queue_pending_image.assert_called_once()
    controller._queue_image_send.assert_not_called()


def test_capture_dispatch_default_path_sends_realtime() -> None:
    controller = _bare_camera_controller()

    controller._capture_and_dispatch_image()

    controller._queue_image_send.assert_called_once()
    assert controller._send_in_flight.is_set()
