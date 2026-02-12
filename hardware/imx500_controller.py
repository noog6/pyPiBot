"""IMX500 object-detection controller skeleton with safe no-op behavior."""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import importlib
import importlib.util
import math
import threading
import time
from typing import Any, Callable

from config import ConfigController
from core.logging import logger
from vision.detections import Detection, DetectionEvent


@dataclass(frozen=True)
class Imx500Settings:
    """Runtime settings for IMX500 detection."""

    enabled: bool = False
    model: str = "yolo11n_pp"
    fps_cap: int = 5
    min_confidence: float = 0.4
    event_buffer_size: int = 50
    interesting_classes: tuple[str, ...] = (
        "person",
        "cat",
        "dog",
        "cell phone",
        "cup",
        "keyboard",
    )
    status_log_period_s: float = 15.0
    startup_grace_s: float = 30.0


class Imx500Controller:
    """Singleton controller that owns IMX500 detection state and lifecycle."""

    _instance: "Imx500Controller | None" = None

    def __init__(self) -> None:
        if Imx500Controller._instance is not None:
            raise RuntimeError("You cannot create another Imx500Controller class")

        self._lock = threading.RLock()
        self._started = False
        self._available = False
        self._disabled_reason = ""
        self._warning_logged = False

        self.settings = self._load_settings()

        self._latest_event: DetectionEvent | None = None
        self._event_buffer: deque[DetectionEvent] = deque(
            maxlen=max(self.settings.event_buffer_size, 1)
        )
        self._subscribers: set[Callable[[list[Detection], float], None]] = set()
        self._worker_thread: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._frame_id = 0
        self._events_published = 0
        self._detection_events_published = 0
        self._last_publish_monotonic: float | None = None
        self._last_detection_monotonic: float | None = None
        self._last_detection_classes: tuple[str, ...] = ()
        self._start_monotonic: float | None = None
        self._last_status_log_monotonic: float = 0.0
        self._no_detections_warning_logged = False

        Imx500Controller._instance = self

    @classmethod
    def get_instance(cls) -> "Imx500Controller":
        """Return the singleton IMX500 controller."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self) -> None:
        """Start the IMX500 subsystem (safe to call repeatedly)."""

        with self._lock:
            if self._started:
                return
            self._started = True

            if not self.settings.enabled:
                self._available = False
                self._disabled_reason = "imx500_enabled is false"
                self._log_warning_once(
                    "[IMX500] Controller disabled in config; running as no-op."
                )
                return

            backend_ok, reason = self._check_backend_available()
            self._available = backend_ok
            self._disabled_reason = reason
            if not backend_ok:
                self._log_warning_once(
                    "[IMX500] Dependencies unavailable; running as no-op "
                    f"({reason})."
                )
                return

            logger.info(
                "[IMX500] Initialized skeleton controller "
                "(model=%s fps_cap=%s min_confidence=%.2f)",
                self.settings.model,
                self.settings.fps_cap,
                self.settings.min_confidence,
            )
            now = time.monotonic()
            self._start_monotonic = now
            self._last_status_log_monotonic = now
            self._start_worker_locked()

    def stop(self) -> None:
        """Stop the IMX500 subsystem (safe to call repeatedly)."""

        worker: threading.Thread | None = None
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._available = False
            self._worker_stop.set()
            worker = self._worker_thread

        if worker is not None:
            worker.join(timeout=2.0)
            if worker.is_alive():
                logger.warning("[IMX500] Worker did not stop within timeout")

        with self._lock:
            if self._worker_thread is worker and (worker is None or not worker.is_alive()):
                self._worker_thread = None

    def _start_worker_locked(self) -> None:
        if self._worker_thread is not None:
            if self._worker_thread.is_alive():
                logger.warning("[IMX500] Worker is still running; restart deferred")
                return
            self._worker_thread = None
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="imx500-detection-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        camera: Any = None
        model_stack: Any = None
        period_s = 1.0 / max(1, self.settings.fps_cap)
        model = self.settings.model

        try:
            camera, model_stack = self._create_imx500_stack(model)
            logger.info("[IMX500] Worker attached to camera stack (model=%s)", model)
            while not self._worker_stop.is_set():
                loop_start = time.monotonic()
                timestamp_s = time.time()
                raw_detections = []
                try:
                    raw_detections, timestamp_s = self._read_raw_detections(camera, model_stack)
                except Exception:
                    logger.exception("[IMX500] Failed to read frame detections")

                detections, frame_id = self._convert_raw_detections(raw_detections)
                self._publish_detections(detections, timestamp=timestamp_s, frame_id=frame_id)
                self._maybe_log_status(time.monotonic())

                elapsed_s = time.monotonic() - loop_start
                sleep_s = max(0.0, period_s - elapsed_s)
                self._worker_stop.wait(sleep_s)
        except Exception as exc:
            logger.exception("[IMX500] Worker initialization failed: %s", exc)
        finally:
            self._shutdown_imx500_stack(camera, model_stack)

    def _create_imx500_stack(self, model: str) -> tuple[Any, Any]:
        picamera2_module = importlib.import_module("picamera2")
        picamera = picamera2_module.Picamera2()

        model_stack = None
        try:
            imx500_module = importlib.import_module("picamera2.devices.imx500")
            model_stack_cls = getattr(imx500_module, "IMX500", None)
            if model_stack_cls is not None:
                model_stack = model_stack_cls(model=model)
        except Exception:
            model_stack = None

        if model_stack is not None and hasattr(model_stack, "create_preview_configuration"):
            config = model_stack.create_preview_configuration()
            picamera.configure(config)
        else:
            picamera.configure(picamera.create_preview_configuration())

        picamera.start()
        return picamera, model_stack

    def _shutdown_imx500_stack(self, camera: Any, model_stack: Any) -> None:
        if model_stack is not None and hasattr(model_stack, "close"):
            try:
                model_stack.close()
            except Exception:
                logger.exception("[IMX500] Failed to close IMX500 model stack")

        if camera is not None:
            for method_name in ("stop", "close"):
                method = getattr(camera, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        logger.exception("[IMX500] Failed to %s camera", method_name)

    def _read_raw_detections(self, camera: Any, model_stack: Any) -> tuple[list[Any], float]:
        timestamp_s = time.time()

        if model_stack is not None and hasattr(model_stack, "get_outputs"):
            outputs = model_stack.get_outputs()
            return list(outputs) if outputs is not None else [], timestamp_s

        metadata = camera.capture_metadata()
        if isinstance(metadata, dict):
            timestamp_raw = metadata.get("SensorTimestamp")
            if isinstance(timestamp_raw, (int, float)):
                timestamp_s = float(timestamp_raw) / 1_000_000_000.0

            for key in (
                "imx500_detections",
                "detections",
                "objects",
                "ai_outputs",
                "imx500",
            ):
                candidate = metadata.get(key)
                if candidate is not None:
                    return list(candidate) if isinstance(candidate, list) else [candidate], timestamp_s

        return [], timestamp_s

    def _convert_raw_detections(self, raw_detections: list[Any]) -> tuple[list[Detection], int]:
        normalized: list[Detection] = []
        for raw in raw_detections:
            detection = self._convert_single_detection(raw)
            if detection is None:
                continue
            normalized.append(detection)

        with self._lock:
            self._frame_id += 1
            frame_id = self._frame_id

        return normalized, frame_id

    def _convert_single_detection(self, raw: Any) -> Detection | None:
        if isinstance(raw, Detection):
            detection = raw
        else:
            payload = self._to_mapping(raw)
            if payload is None:
                return None

            confidence = self._extract_confidence(payload)
            if confidence < self.settings.min_confidence:
                return None

            label = self._extract_label(payload)
            bbox = self._extract_bbox(payload)
            if bbox is None:
                return None
            metadata = {k: v for k, v in payload.items() if k not in {"label", "class", "class_name", "name", "score", "confidence", "bbox", "box", "rect", "rectangle", "x", "y", "w", "h", "width", "height", "xmin", "ymin", "xmax", "ymax"}}
            metadata["model"] = self.settings.model

            detection = Detection(
                label=label,
                confidence=confidence,
                bbox=bbox,
                metadata=metadata,
            )

        if detection.confidence < self.settings.min_confidence:
            return None

        return detection

    def _to_mapping(self, raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, dict):
            return raw

        mapping: dict[str, Any] = {}
        for field in (
            "label",
            "class",
            "class_name",
            "name",
            "score",
            "confidence",
            "bbox",
            "box",
            "rect",
            "rectangle",
            "x",
            "y",
            "w",
            "h",
            "width",
            "height",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ):
            if hasattr(raw, field):
                mapping[field] = getattr(raw, field)

        return mapping or None

    def _extract_confidence(self, payload: dict[str, Any]) -> float:
        value = payload.get("score", payload.get("confidence", 0.0))
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(confidence) or math.isinf(confidence):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _extract_label(self, payload: dict[str, Any]) -> str:
        value = payload.get("label", payload.get("class_name", payload.get("class", payload.get("name", "unknown"))))
        label = str(value).strip() if value is not None else "unknown"
        return label or "unknown"

    def _extract_bbox(self, payload: dict[str, Any]) -> tuple[float, float, float, float] | None:
        raw_bbox = payload.get("bbox", payload.get("box", payload.get("rect", payload.get("rectangle"))))

        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            x, y, w, h = raw_bbox[:4]
            x_f = self._to_finite_float(x)
            y_f = self._to_finite_float(y)
            w_f = self._to_finite_float(w)
            h_f = self._to_finite_float(h)
            if None in (x_f, y_f, w_f, h_f):
                return None
            return self._normalize_bbox(x_f, y_f, w_f, h_f)

        if {"xmin", "ymin", "xmax", "ymax"}.issubset(payload.keys()):
            xmin = self._to_finite_float(payload.get("xmin", 0.0))
            ymin = self._to_finite_float(payload.get("ymin", 0.0))
            xmax = self._to_finite_float(payload.get("xmax", xmin))
            ymax = self._to_finite_float(payload.get("ymax", ymin))
            if None in (xmin, ymin, xmax, ymax):
                return None
            return self._normalize_bbox(xmin, ymin, xmax - xmin, ymax - ymin)

        x = self._to_finite_float(payload.get("x", 0.0))
        y = self._to_finite_float(payload.get("y", 0.0))
        w = self._to_finite_float(payload.get("w", payload.get("width", 1.0)))
        h = self._to_finite_float(payload.get("h", payload.get("height", 1.0)))
        if None in (x, y, w, h):
            return None
        return self._normalize_bbox(x, y, w, h)

    def _to_finite_float(self, value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    def _normalize_bbox(self, x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))
        return (x, y, w, h)

    def get_latest_detections(self) -> list[Detection]:
        """Return the most recent detection snapshot."""

        latest = self.get_latest_event()
        if latest is None:
            return []
        return [self._clone_detection(item) for item in latest.detections]

    def get_latest_event(self) -> DetectionEvent | None:
        """Return latest stable detection event snapshot for non-blocking readers."""

        with self._lock:
            event = self._latest_event
        if event is None:
            return None
        return self._clone_event(event)

    def get_recent_events(self, n: int = 10) -> list[DetectionEvent]:
        """Return up to ``n`` events ordered from oldest to newest."""

        with self._lock:
            events = list(self._event_buffer)
        if n <= 0:
            return []
        selected = events[-n:]
        return [self._clone_event(event) for event in selected]

    def subscribe(self, callback: Callable[[list[Detection], float], None]) -> None:
        """Subscribe to future detection updates."""

        with self._lock:
            self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[list[Detection], float], None]) -> None:
        """Unsubscribe from detection updates."""

        with self._lock:
            self._subscribers.discard(callback)

    def is_available(self) -> bool:
        """Return whether IMX500 backend dependencies are currently available."""

        with self._lock:
            return self._available

    def get_runtime_status(self) -> dict[str, int | float | str]:
        """Return runtime IMX500 status metrics for ops health and diagnostics."""

        now = time.monotonic()
        with self._lock:
            worker_alive = bool(self._worker_thread and self._worker_thread.is_alive())
            enabled = self.settings.enabled
            backend_available = self._available
            events_published = self._events_published
            detection_events_published = self._detection_events_published
            last_publish_age_s = (
                now - self._last_publish_monotonic
                if self._last_publish_monotonic is not None
                else -1.0
            )
            last_detection_age_s = (
                now - self._last_detection_monotonic
                if self._last_detection_monotonic is not None
                else -1.0
            )
            last_classes = ", ".join(self._last_detection_classes)

        return {
            "enabled": int(enabled),
            "backend_available": int(backend_available),
            "loop_alive": int(worker_alive),
            "events_published": events_published,
            "detection_events_published": detection_events_published,
            "last_event_age_s": round(last_publish_age_s, 3) if last_publish_age_s >= 0.0 else -1.0,
            "last_detection_age_s": round(last_detection_age_s, 3)
            if last_detection_age_s >= 0.0
            else -1.0,
            "last_classes_confidences": last_classes,
        }

    def _publish_detections(
        self,
        detections: list[Detection],
        timestamp: float | None = None,
        frame_id: int | None = None,
    ) -> None:
        """Publish a new detection snapshot to subscribers.

        Note: this internal helper is for future integration points.
        """

        timestamp_s = timestamp if timestamp is not None else time.time()
        timestamp_ms = int(timestamp_s * 1000)
        event = DetectionEvent(
            timestamp_ms=timestamp_ms,
            detections=[self._clone_detection(item) for item in detections],
            frame_id=frame_id,
            source="imx500",
        )
        event_for_callbacks = self._clone_event(event)

        with self._lock:
            self._latest_event = event
            self._event_buffer.append(event)
            subscribers = list(self._subscribers)
            self._record_publish_metrics_locked(event)

        self._log_interesting_detection(event)

        for callback in subscribers:
            try:
                callback(
                    [self._clone_detection(item) for item in event_for_callbacks.detections],
                    timestamp_s,
                )
            except Exception:
                logger.exception("[IMX500] Subscriber callback failed")

    def _load_settings(self) -> Imx500Settings:
        config = ConfigController.get_instance().get_config()

        classes_value = config.get("imx500_interesting_classes")
        if isinstance(classes_value, list):
            classes = tuple(str(item) for item in classes_value)
        else:
            classes = Imx500Settings().interesting_classes

        return Imx500Settings(
            enabled=bool(config.get("imx500_enabled", False)),
            model=str(config.get("imx500_model", "yolo11n_pp")),
            fps_cap=int(config.get("imx500_fps_cap", 5)),
            min_confidence=float(config.get("imx500_min_confidence", 0.4)),
            event_buffer_size=int(config.get("imx500_event_buffer_size", 50)),
            interesting_classes=classes,
            status_log_period_s=float(config.get("imx500_status_log_period_s", 15.0)),
            startup_grace_s=float(config.get("imx500_startup_grace_s", 30.0)),
        )

    def _record_publish_metrics_locked(self, event: DetectionEvent) -> None:
        now = time.monotonic()
        self._events_published += 1
        self._last_publish_monotonic = now
        if not event.detections:
            return
        self._detection_events_published += 1
        self._last_detection_monotonic = now
        self._last_detection_classes = tuple(
            f"{detection.label}:{detection.confidence:.2f}"
            for detection in event.detections[:3]
        )

    def _maybe_log_status(self, now_monotonic: float) -> None:
        status_log_period_s = max(10.0, min(30.0, self.settings.status_log_period_s))
        startup_grace_s = max(0.0, self.settings.startup_grace_s)

        should_log_status = False
        warning_message = ""
        with self._lock:
            if (now_monotonic - self._last_status_log_monotonic) >= status_log_period_s:
                self._last_status_log_monotonic = now_monotonic
                should_log_status = True

            if (
                self.settings.enabled
                and self._available
                and self._start_monotonic is not None
                and not self._no_detections_warning_logged
                and self._detection_events_published <= 0
                and (now_monotonic - self._start_monotonic) >= startup_grace_s
            ):
                elapsed_s = int(now_monotonic - self._start_monotonic)
                warning_message = (
                    "IMX500 enabled and backend available, but no detection "
                    f"events published in {elapsed_s}s."
                )
                self._no_detections_warning_logged = True

        if should_log_status:
            status = self.get_runtime_status()
            logger.info(
                "[IMX500] Status: backend_available=%s loop_alive=%s "
                "events_published=%s detection_events_published=%s "
                "last_event_age_s=%s last_detection_age_s=%s "
                "last_classes_confidences=%s",
                status["backend_available"],
                status["loop_alive"],
                status["events_published"],
                status["detection_events_published"],
                status["last_event_age_s"],
                status["last_detection_age_s"],
                status["last_classes_confidences"] or "none",
            )

        if warning_message:
            logger.warning("[IMX500] %s", warning_message)

    def _check_backend_available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("picamera2") is None:
            return False, "missing picamera2"

        # IMX500 support may be exposed via this helper module on Raspberry Pi stacks.
        try:
            if importlib.util.find_spec("picamera2.devices.imx500") is not None:
                return True, ""
        except ModuleNotFoundError:
            # picamera2 may not be installed as a package with importable submodules.
            pass

        # Some environments package IMX500 support under top-level module names.
        if importlib.util.find_spec("imx500") is not None:
            return True, ""

        return False, "missing IMX500 extras"

    def _log_interesting_detection(self, event: DetectionEvent) -> None:
        interesting_labels = {label.lower() for label in self.settings.interesting_classes}
        matches = [
            det
            for det in event.detections
            if det.label.lower() in interesting_labels
            and det.confidence >= self.settings.min_confidence
        ]
        if not matches:
            return
        labels = ", ".join(
            f"{det.label}:{det.confidence:.2f}" for det in matches[:3]
        )
        logger.info(
            "[IMX500] Interesting detections (%d total): %s",
            len(matches),
            labels,
        )

    def _clone_detection(self, detection: Detection) -> Detection:
        return Detection(
            label=detection.label,
            confidence=float(detection.confidence),
            bbox=tuple(detection.bbox),
            metadata=copy.deepcopy(detection.metadata),
        )

    def _clone_event(self, event: DetectionEvent) -> DetectionEvent:
        return DetectionEvent(
            timestamp_ms=int(event.timestamp_ms),
            detections=[self._clone_detection(item) for item in event.detections],
            frame_id=event.frame_id,
            source=event.source,
        )

    def _log_warning_once(self, message: str) -> None:
        if self._warning_logged:
            return
        logger.warning(message)
        self._warning_logged = True
