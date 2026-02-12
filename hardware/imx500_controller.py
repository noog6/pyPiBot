"""IMX500 object-detection controller skeleton with safe no-op behavior."""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import importlib.util
import threading
import time
from typing import Callable

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

    def stop(self) -> None:
        """Stop the IMX500 subsystem (safe to call repeatedly)."""

        with self._lock:
            if not self._started:
                return
            self._started = False
            self._available = False

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

    def _publish_detections(self, detections: list[Detection], timestamp: float | None = None) -> None:
        """Publish a new detection snapshot to subscribers.

        Note: this internal helper is for future integration points.
        """

        timestamp_s = timestamp if timestamp is not None else time.time()
        timestamp_ms = int(timestamp_s * 1000)
        event = DetectionEvent(
            timestamp_ms=timestamp_ms,
            detections=[self._clone_detection(item) for item in detections],
            source="imx500",
        )
        event_for_callbacks = self._clone_event(event)

        with self._lock:
            self._latest_event = event
            self._event_buffer.append(event)
            subscribers = list(self._subscribers)

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
        )

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
