"""Camera controller for capturing frames and sending vision updates."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
import traceback
from typing import Any
from collections import deque
from datetime import datetime
from pathlib import Path

from config import ConfigController
from core.logging import logger
from motion.motion_controller import MotionController, millis
from storage.controller import StorageController
from vision.attention import AttentionController, AttentionState
from vision.detections import DetectionEvent


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
    return picamera2.Picamera2, numpy, pil_image


def _safe_connection_closed_ok() -> type[BaseException] | None:
    import importlib
    import importlib.util

    if importlib.util.find_spec("websockets") is None:
        return None
    websockets = importlib.import_module("websockets")
    exceptions_module = getattr(websockets, "exceptions", None)
    if exceptions_module is not None:
        return getattr(exceptions_module, "ConnectionClosedOK", None)
    return getattr(websockets, "ConnectionClosedOK", None)


class CameraController:
    """Singleton controller for camera capture and vision loop."""

    _instance: "CameraController | None" = None

    def __init__(self) -> None:
        if CameraController._instance is not None:
            raise RuntimeError("You cannot create another CameraController class")

        Picamera2, numpy, pil_image = _require_camera_deps()
        self._np = numpy
        self._pil_image = pil_image

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
        self._image_save_index = 0
        self._image_save_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._image_save_enabled = False
        self._image_save_dir = None
        self._pending_images: deque[Any] = deque(maxlen=3)
        self._pending_lock = threading.Lock()
        self._attention_controller: AttentionController | None = None
        self._imx500_controller: Any = None
        self._configure_image_saving()
        self._warmup_frames = 0
        self._warmup_ms = 0
        self._warmup_start_ms = 0
        self._warmup_frames_seen = 0
        self._warmup_done = True
        self._configure_warmup()

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
            self._reset_warmup()
            self._vision_loop_thread = threading.Thread(target=self._vision_loop, daemon=True)
            self._vision_loop_thread.start()

    def stop_vision_loop(self) -> None:
        if self._vision_loop_thread is not None:
            self._stop_event.set()
            self._vision_loop_thread.join(timeout=2.0)
            if self._vision_loop_thread.is_alive():
                logger.warning("[CAMERA] Control loop did not stop within timeout")
                self._shutdown_image_saver()
                return
            self._vision_loop_thread = None
            logger.info("[CAMERA] Control loop stopped at index: %s", self.vision_loop_index)
            self.vision_loop_index = 0
            self._shutdown_image_saver()

    def take_image(self) -> Any:
        frame = self.picam2.capture_array("main")
        rotated_image = self._np.rot90(frame, k=3)
        final_image = self._pil_image.fromarray(rotated_image)
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
        return self._pil_image.fromarray(rotated_image, mode="RGB")

    def _vision_loop(self) -> None:
        next_vision_loop_time = millis() + self.vision_loop_period_ms
        while not self._stop_event.is_set():
            current_time = millis()
            if current_time >= next_vision_loop_time:
                self.vision_loop_index += 1
                effective_period_ms = self.vision_loop_period_ms
                next_vision_loop_time = current_time + effective_period_ms

                if self._send_in_flight.is_set():
                    time.sleep(0.01)
                    continue

                if self._drain_pending_images():
                    time.sleep(0.01)
                    continue

                if self.motion and self.motion.is_moving():
                    logger.info("[CAMERA] skipped (motion active)")
                    time.sleep(0.01)
                    continue

                try:
                    luma = self.take_lores_luma()
                    changed, score = self.lores_changed(luma, threshold=7.0)
                    imx500_event = self._get_latest_imx500_event()
                    attention = self._get_attention_controller()
                    attention_state = AttentionState.IDLE
                    if attention is not None:
                        attention_state = attention.update(
                            now_ms=current_time,
                            mad_changed=changed,
                            detections=imx500_event,
                        )
                        effective_period_ms = attention.get_capture_period_ms(
                            attention_state,
                            self.vision_loop_period_ms,
                        )

                    interesting_detection = self._is_interesting_detection(
                        attention,
                        imx500_event,
                    )
                    should_send = changed or interesting_detection
                    if attention is not None:
                        should_send = should_send or attention.should_send_image(
                            attention_state,
                            changed,
                            imx500_event,
                        )

                    next_vision_loop_time = current_time + effective_period_ms
                    if not should_send:
                        time.sleep(0.01)
                        continue

                    logger.info(
                        "[CAMERA] trigger state=%s mad=%.2f changed=%s detection=%s",
                        attention_state.value,
                        score,
                        changed,
                        interesting_detection,
                    )

                    self._capture_and_dispatch_image()

                    if attention is not None and attention.should_burst(attention_state):
                        burst_count = attention.get_burst_count()
                        for _ in range(max(0, burst_count - 1)):
                            if self._stop_event.is_set():
                                break
                            time.sleep(0.08)
                            self._capture_and_dispatch_image(prefer_queue_if_in_flight=True)

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

    def _can_send_realtime(self) -> bool:
        if not self.realtime_instance:
            return False
        is_ready = getattr(self.realtime_instance, "is_ready_for_injections", None)
        if callable(is_ready):
            return is_ready()
        loop = getattr(self.realtime_instance, "loop", None)
        return loop is not None and loop.is_running()

    def _queue_pending_image(self, image: Any) -> None:
        if not self.realtime_instance:
            logger.warning("[CAMERA] Unable to take image - realtime instance not available")
            return
        with self._pending_lock:
            if len(self._pending_images) == self._pending_images.maxlen:
                self._pending_images.popleft()
            self._pending_images.append(image)
        logger.info("[CAMERA] Queued image until realtime loop is ready.")

    def _drain_pending_images(self) -> bool:
        if not self._can_send_realtime():
            return False
        with self._pending_lock:
            if not self._pending_images:
                return False
            image = self._pending_images.popleft()
        self._send_in_flight.set()
        if not self._queue_image_send(image):
            self._send_in_flight.clear()
        return True

    def _queue_image_send(self, image: Any) -> bool:
        if not self.realtime_instance:
            logger.warning("[CAMERA] Unable to send image - realtime instance not available")
            return False
        loop = getattr(self.realtime_instance, "loop", None)
        if loop is None or not loop.is_running():
            logger.warning("[CAMERA] Unable to send image - realtime event loop not running")
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.realtime_instance.send_image_to_assistant(image),
                loop,
            )
        except Exception as exc:
            logger.exception("[CAMERA] Failed to queue image send: %s", exc)
            return False
        future.add_done_callback(self._clear_send_flag)
        return True

    def _capture_and_dispatch_image(self, prefer_queue_if_in_flight: bool = False) -> None:
        new_image = self.take_main_pil()
        self._save_image_async(new_image)

        if prefer_queue_if_in_flight and self._send_in_flight.is_set():
            self._queue_pending_image(new_image)
            return

        if self._can_send_realtime():
            self._send_in_flight.set()
            if not self._queue_image_send(new_image):
                self._send_in_flight.clear()
        else:
            self._queue_pending_image(new_image)

    def _get_attention_controller(self) -> AttentionController | None:
        if self._attention_controller is not None:
            return self._attention_controller
        try:
            controller = AttentionController.get_instance()
        except Exception:
            logger.exception("[CAMERA] Attention controller unavailable; using MAD-only gating")
            return None
        if not controller.config.enabled:
            return None
        self._attention_controller = controller
        return self._attention_controller

    def _get_latest_imx500_event(self) -> DetectionEvent | None:
        try:
            if self._imx500_controller is None:
                from hardware.imx500_controller import Imx500Controller

                self._imx500_controller = Imx500Controller.get_instance()
            return self._imx500_controller.get_latest_event()
        except Exception:
            return None

    def _is_interesting_detection(
        self,
        attention: AttentionController | None,
        event: DetectionEvent | None,
    ) -> bool:
        if attention is not None:
            return attention.is_interesting_event(event)
        return event is not None and len(event.detections) > 0

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

    def _configure_image_saving(self) -> None:
        config = ConfigController.get_instance().get_config()
        self._image_save_enabled = bool(config.get("save_camera_images", False))
        if not self._image_save_enabled:
            return
        storage_controller = StorageController.get_instance()
        self._image_save_dir = storage_controller.get_run_image_dir()
        self._image_save_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="camera-image-save",
        )

    def _configure_warmup(self) -> None:
        config = ConfigController.get_instance().get_config()
        self._warmup_frames = max(int(config.get("camera_warmup_frames", 5)), 0)
        self._warmup_ms = max(int(config.get("camera_warmup_ms", 1000)), 0)
        self._reset_warmup()

    def _reset_warmup(self) -> None:
        self._warmup_start_ms = millis()
        self._warmup_frames_seen = 0
        self._warmup_done = self._warmup_frames == 0 and self._warmup_ms == 0
        self._last_luma = None

    def _shutdown_image_saver(self) -> None:
        if self._image_save_executor is not None:
            self._image_save_executor.shutdown(wait=False)
            self._image_save_executor = None

    def _save_image_async(self, image: Any) -> None:
        if not self._image_save_enabled or self._image_save_executor is None:
            return
        if self._image_save_dir is None:
            return
        self._image_save_index += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        millis_part = int(time.time() * 1000) % 1000
        filename = f"image_{timestamp}_{millis_part:03d}_{self._image_save_index:06d}.jpg"
        path = self._image_save_dir / filename
        detection_json_path = path.with_suffix(".detections.json")
        detection_payload = self._latest_detection_payload()
        image_copy = image.copy()
        self._image_save_executor.submit(self._write_image, image_copy, path)
        if detection_payload is not None:
            self._image_save_executor.submit(
                self._write_detection_artifact,
                detection_json_path,
                detection_payload,
            )

    def _write_image(self, image: Any, path: Any) -> None:
        try:
            image.save(path, format="JPEG", quality=85)
        except Exception as exc:
            logger.exception("[CAMERA] Failed to save image to %s: %s", path, exc)

    def _latest_detection_payload(self) -> dict[str, Any] | None:
        try:
            from hardware.imx500_controller import Imx500Controller
        except Exception:
            return None

        controller = Imx500Controller.get_instance()
        event = controller.get_latest_event()
        if event is None:
            return None
        return {
            "timestamp_ms": int(event.timestamp_ms),
            "frame_id": event.frame_id,
            "source": event.source,
            "detections": [
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "bbox": list(item.bbox),
                    "metadata": item.metadata,
                }
                for item in event.detections
            ],
        }

    def _write_detection_artifact(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        except Exception as exc:
            logger.exception("[CAMERA] Failed to save detections to %s: %s", path, exc)

    def lores_changed(self, luma: Any, threshold: float = 7.0) -> tuple[bool, float]:
        if not self._warmup_done:
            self._warmup_frames_seen += 1
            elapsed_ms = millis() - self._warmup_start_ms
            if self._warmup_frames_seen < self._warmup_frames or elapsed_ms < self._warmup_ms:
                self._last_luma = luma
                return False, 0.0
            self._warmup_done = True
            self._last_luma = luma
            return False, 0.0

        if self._last_luma is None:
            self._last_luma = luma
            return False, 0.0

        d = luma.astype(self._np.int16) - self._last_luma.astype(self._np.int16)
        mad = float(self._np.abs(d).mean())
        self._last_luma = luma
        return (mad >= threshold), mad
