"""IMX500 object-detection controller skeleton with safe no-op behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import threading
import time
from typing import Any, Callable

from config import ConfigController
from core.logging import logger


@dataclass(frozen=True)
class Detection:
    """Temporary detection schema for IMX500 object detection snapshots."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Imx500Settings:
    """Runtime settings for IMX500 detection."""

    enabled: bool = False
    model: str = "yolo11n_pp"
    fps_cap: int = 5
    min_confidence: float = 0.4
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

        self._latest_detections: list[Detection] = []
        self._latest_timestamp: float = 0.0
        self._subscribers: set[Callable[[list[Detection], float], None]] = set()

        self.settings = self._load_settings()

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

        with self._lock:
            return list(self._latest_detections)

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

        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            self._latest_detections = list(detections)
            self._latest_timestamp = ts
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(list(detections), ts)
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

    def _log_warning_once(self, message: str) -> None:
        if self._warning_logged:
            return
        logger.warning(message)
        self._warning_logged = True
