"""Camera controller for capturing frames and sending vision updates."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from typing import Any

from core.logging import logger
from motion.motion_controller import MotionController, millis


LOGGER = logging.getLogger(__name__)


def _require_camera_deps() -> tuple[Any, Any, Any]:
    import importlib
    import importlib.util

    if importlib.util.find_spec("picamera2") is None:
        raise RuntimeError("picamera2 is required for CameraController")
    if importlib.util.find_spec("numpy") is None:
        raise RuntimeError("numpy is required for CameraController")
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError("Pillow is required for CameraController")

    picamera2 = importlib.import_module("picamera2")
    numpy = importlib.import_module("numpy")
    pil_image = importlib.import_module("PIL.Image")
    return picamera2.Picamera2, numpy, pil_image.Image


def _safe_connection_closed_ok() -> type[BaseException] | None:
    import importlib
    import importlib.util

    if importlib.util.find_spec("websockets") is None:
        return None
    websockets = importlib.import_module("websockets")
    return websockets.exceptions.ConnectionClosedOK


class CameraController:
    """Singleton controller for camera capture and vision loop."""

    _instance: "CameraController | None" = None

    def __init__(self) -> None:
        if CameraController._instance is not None:
            raise RuntimeError("You cannot create another CameraController class")

        Picamera2, numpy, image_cls = _require_camera_deps()
        self._np = numpy
        self._Image = image_cls

        self.picam2 = Picamera2()
        self._main_size = (640, 480)
        self._lores_size = (160, 90)
        self._last_luma = None
        self.camera_configuration = self.picam2.create_preview_configuration(
            main={"size": self._main_size, "format": "RGB888"},
            lores={"size": self._lores_size, "format": "YUV420"},
            buffer_count=2,
        )
        self.picam2.configure(self.camera_configuration)
        self.picam2.start()

        self._vision_loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._send_in_flight = threading.Event()
        self.vision_loop_function = None
        self.vision_loop_period_ms = 0
        self.vision_loop_start_time = [0] * 100
        self.vision_loop_index = 0
        self.last_image = None
        self.realtime_instance = None
        self.motion = MotionController.get_instance()

        CameraController._instance = self

    @classmethod
    def get_instance(cls) -> "CameraController":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_vision_loop(self, vision_loop_period_ms: int = 15000) -> None:
        if self._vision_loop_thread is None or not self._vision_loop_thread.is_alive():
            self._stop_event.clear()
            self.vision_loop_period_ms = vision_loop_period_ms
            self._vision_loop_thread = threading.Thread(target=self._vision_loop, daemon=True)
            self._vision_loop_thread.start()

    def stop_vision_loop(self) -> None:
        if self._vision_loop_thread is not None:
            self._stop_event.set()
            self._vision_loop_thread.join()
            self._vision_loop_thread = None
            logger.info("[CAMERA] Control loop stopped at index: %s", self.vision_loop_index)
            self.vision_loop_index = 0

    def take_image(self) -> Any:
        frame = self.picam2.capture_array("main")
        rotated_image = self._np.rot90(frame, k=3)
        final_image = self._Image.fromarray(rotated_image)
        return final_image

    def take_lores_luma(self) -> Any:
        yuv = self.picam2.capture_array("lores")
        h = self._lores_size[1]
        y = yuv[:h, :]
        return y.copy()

    def take_main_pil(self) -> Any:
        frame = self.picam2.capture_array("main")
        frame = frame[:, :, ::-1]
        rotated_image = self._np.rot90(frame, k=3)
        return self._Image.fromarray(rotated_image, mode="RGB")

    def _vision_loop(self) -> None:
        next_vision_loop_time = millis() + self.vision_loop_period_ms
        while not self._stop_event.is_set():
            current_time = millis()
            if current_time >= next_vision_loop_time:
                next_vision_loop_time = current_time + self.vision_loop_period_ms

                if self._send_in_flight.is_set():
                    time.sleep(0.01)
                    continue

                if self.motion and self.motion.is_moving():
                    logger.info("[CAMERA] skipped (motion active)")
                    time.sleep(0.01)
                    continue

                try:
                    luma = self.take_lores_luma()
                    changed, score = self.lores_changed(luma, threshold=7.0)
                    if not changed:
                        time.sleep(0.01)
                        continue

                    logger.info("[CAMERA] change detected (mad=%.2f)", score)

                    if self.realtime_instance:
                        self._send_in_flight.set()
                        new_image = self.take_main_pil()
                        future = asyncio.run_coroutine_threadsafe(
                            self.realtime_instance.send_image_to_assistant(new_image),
                            self.realtime_instance.loop,
                        )
                        future.add_done_callback(self._clear_send_flag)
                    else:
                        self._send_in_flight.clear()
                        logger.warning("[CAMERA] Unable to take image - realtime instance not available")

                except Exception as exc:
                    self._send_in_flight.clear()
                    logger.exception("[CAMERA] Error in control loop (retrying): %s", exc)
                    traceback.print_exc()
            else:
                time.sleep(0.01)

    def is_vision_loop_alive(self) -> bool:
        return self._vision_loop_thread is not None and self._vision_loop_thread.is_alive()

    def toggle_vision_loop(self) -> None:
        if self.is_vision_loop_alive():
            self.stop_vision_loop()
        else:
            self.start_vision_loop()

    def set_realtime_instance(self, realtime_instance: Any) -> None:
        self.realtime_instance = realtime_instance

    def _clear_send_flag(self, fut: Any) -> None:
        ConnectionClosedOK = _safe_connection_closed_ok()
        try:
            fut.result()
        except Exception as exc:
            if ConnectionClosedOK is not None and isinstance(exc, ConnectionClosedOK):
                pass
            else:
                logger.exception("[CAMERA] [WARN] Image send failed: %s", exc)
        finally:
            self._send_in_flight.clear()

    def lores_changed(self, luma: Any, threshold: float = 7.0) -> tuple[bool, float]:
        if self._last_luma is None:
            self._last_luma = luma
            return True, 999.0

        d = luma.astype(self._np.int16) - self._last_luma.astype(self._np.int16)
        mad = float(self._np.abs(d).mean())
        self._last_luma = luma
        return (mad >= threshold), mad
