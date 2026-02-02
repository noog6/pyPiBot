"""Battery monitor for periodic voltage status events."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable, Iterable

from hardware import ADS1015Sensor


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatteryStatusEvent:
    """Derived battery status event for local behavior or LLM context."""

    timestamp: float
    voltage: float
    percent_of_range: float
    severity: str
    event_type: str = "status"


class BatteryMonitor:
    """Singleton battery monitor with a background sampling loop."""

    _instance: "BatteryMonitor | None" = None

    def __init__(self) -> None:
        if BatteryMonitor._instance is not None:
            raise RuntimeError("You cannot create another BatteryMonitor class")

        self._sensor = ADS1015Sensor.get_instance()
        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_event: BatteryStatusEvent | None = None
        self._event_handlers: set[Callable[[BatteryStatusEvent], None]] = set()
        self._min_voltage = 7.0
        self._max_voltage = 8.4
        self._loop_period_s = 60.0
        self._low_battery_period_s = 30.0
        BatteryMonitor._instance = self

    @classmethod
    def get_instance(cls) -> "BatteryMonitor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_loop(self, loop_period_s: float = 60.0, low_battery_period_s: float = 30.0) -> None:
        if self._loop_thread is None or not self._loop_thread.is_alive():
            self._loop_period_s = max(loop_period_s, 10.0)
            self._low_battery_period_s = max(low_battery_period_s, 10.0)
            self._stop_event.clear()
            self._loop_thread = threading.Thread(target=self._loop, daemon=True)
            self._loop_thread.start()

    def stop_loop(self) -> None:
        if self._loop_thread is not None:
            self._stop_event.set()
            self._loop_thread.join()
            self._loop_thread = None

    def is_loop_alive(self) -> bool:
        return self._loop_thread is not None and self._loop_thread.is_alive()

    def get_latest_event(self) -> BatteryStatusEvent | None:
        with self._lock:
            return self._latest_event

    def register_event_handler(self, handler: Callable[[BatteryStatusEvent], None]) -> None:
        self._event_handlers.add(handler)

    def unregister_event_handler(self, handler: Callable[[BatteryStatusEvent], None]) -> None:
        self._event_handlers.discard(handler)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                voltage = self._sensor.read_battery_voltage()
                event = self._build_event(voltage)
                self._store_event(event)
            except Exception as exc:
                LOGGER.exception("[Battery] Error in loop (retrying): %s", exc)
                event = None

            sleep_for = self._loop_period_s
            if event and event.percent_of_range <= 0.25:
                sleep_for = self._low_battery_period_s
            self._stop_event.wait(timeout=sleep_for)

    def _build_event(self, voltage: float) -> BatteryStatusEvent:
        percent = (voltage - self._min_voltage) / (self._max_voltage - self._min_voltage)
        percent = max(0.0, min(1.0, percent))
        if percent <= 0.25:
            severity = "critical"
        elif percent <= 0.5:
            severity = "warning"
        else:
            severity = "info"
        return BatteryStatusEvent(
            timestamp=time.time(),
            voltage=voltage,
            percent_of_range=percent,
            severity=severity,
            event_type="status",
        )

    def _store_event(self, event: BatteryStatusEvent) -> None:
        clear_event = None
        with self._lock:
            previous_event = self._latest_event
            self._latest_event = event
        if (
            previous_event
            and previous_event.severity in {"warning", "critical"}
            and event.severity == "info"
        ):
            clear_event = BatteryStatusEvent(
                timestamp=event.timestamp,
                voltage=event.voltage,
                percent_of_range=event.percent_of_range,
                severity="info",
                event_type="clear",
            )
        self._emit_events([clear_event, event] if clear_event else [event])

    def _emit_events(self, events: Iterable[BatteryStatusEvent]) -> None:
        handlers = list(self._event_handlers)
        if not handlers:
            return
        for event in events:
            LOGGER.info(
                "[Battery] Emitting event: voltage=%.2fV severity=%s percent=%.2f",
                event.voltage,
                event.severity,
                event.percent_of_range,
            )
            for handler in handlers:
                try:
                    handler(event)
                except Exception as exc:
                    LOGGER.exception("[Battery] Event handler failed: %s", exc)
