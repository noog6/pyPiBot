"""Battery monitor for periodic voltage status events."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Iterable

from ai.event_bus import Event, EventBus
from config import ConfigController
from core.logging import logger as LOGGER
from hardware import ADS1015Sensor


@dataclass(frozen=True)
class BatteryStatusEvent:
    """Derived battery status event for local behavior or LLM context."""

    timestamp: float
    voltage: float
    percent_of_range: float
    severity: str
    event_type: str = "status"
    transition: str = "steady"
    delta_percent: float = 0.0
    rapid_drop: bool = False


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
        self._warning_percent = 50.0
        self._critical_percent = 25.0
        self._hysteresis_percent = 0.0
        self._response_enabled = True
        self._response_cooldown_s = 60.0
        self._response_allow_warning = True
        self._response_allow_critical = True
        self._response_require_transition = False
        self._load_config()
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

    def stop_loop(self, timeout_s: float = 2.0) -> None:
        if self._loop_thread is not None:
            self._stop_event.set()
            self._loop_thread.join(timeout=timeout_s)
            if self._loop_thread.is_alive():
                LOGGER.warning(
                    "[Battery] Loop thread did not exit within %.2fs; continuing shutdown.",
                    timeout_s,
                )
                return
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

    def create_event_bus_handler(self, event_bus: EventBus) -> Callable[[BatteryStatusEvent], None]:
        last_response_time = 0.0
        last_response_severity: str | None = None

        def _handle_battery_event(event: BatteryStatusEvent) -> None:
            nonlocal last_response_time, last_response_severity

            if event.severity == "critical":
                priority = "critical"
            elif event.severity == "warning":
                priority = "high"
            else:
                priority = "low"

            request_response = self._should_request_response(
                event,
                last_response_severity=last_response_severity,
                last_response_time=last_response_time,
            )
            if request_response:
                last_response_time = time.time()
                last_response_severity = event.severity
            elif event.severity == "info" or event.event_type == "clear":
                last_response_severity = None

            event_bus.publish(
                Event(
                    source="battery",
                    kind="status",
                    priority=priority,
                    dedupe_key="battery_status",
                    cooldown_s=self._response_cooldown_s,
                    request_response=request_response,
                    metadata={
                        "voltage": event.voltage,
                        "percent_of_range": event.percent_of_range,
                        "severity": event.severity,
                        "event_type": event.event_type,
                        "transition": event.transition,
                        "delta_percent": event.delta_percent,
                        "rapid_drop": event.rapid_drop,
                    },
                ),
                coalesce=True,
            )

        return _handle_battery_event

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                voltage = self._sensor.read_battery_voltage()
                with self._lock:
                    previous_event = self._latest_event
                event = self._build_event(voltage, previous_event)
                self._store_event(event)
            except Exception as exc:
                LOGGER.exception("[Battery] Error in loop (retrying): %s", exc)
                event = None

            sleep_for = self._loop_period_s
            if event and event.percent_of_range <= self._critical_threshold():
                sleep_for = self._low_battery_period_s
            self._stop_event.wait(timeout=sleep_for)

    def _build_event(
        self,
        voltage: float,
        previous_event: BatteryStatusEvent | None,
    ) -> BatteryStatusEvent:
        percent = (voltage - self._min_voltage) / (self._max_voltage - self._min_voltage)
        percent = max(0.0, min(1.0, percent))
        severity = self._derive_severity(percent, previous_event)

        previous_percent = previous_event.percent_of_range if previous_event else percent
        delta_percent = (percent - previous_percent) * 100.0
        transition = self._derive_transition(severity, previous_event, delta_percent)

        return BatteryStatusEvent(
            timestamp=time.time(),
            voltage=voltage,
            percent_of_range=percent,
            severity=severity,
            event_type="status",
            transition=transition,
            delta_percent=delta_percent,
            rapid_drop=self._is_rapid_drop(delta_percent),
        )

    def _derive_severity(
        self,
        percent_of_range: float,
        previous_event: BatteryStatusEvent | None,
    ) -> str:
        warning = self._warning_threshold()
        critical = self._critical_threshold()
        hysteresis = max(self._hysteresis_percent, 0.0) / 100.0

        if previous_event is None:
            if percent_of_range <= critical:
                return "critical"
            if percent_of_range <= warning:
                return "warning"
            return "info"

        previous_severity = previous_event.severity

        if previous_severity == "critical":
            if percent_of_range <= critical + hysteresis:
                return "critical"
            if percent_of_range <= warning:
                return "warning"
            return "info"

        if previous_severity == "warning":
            if percent_of_range <= critical:
                return "critical"
            if percent_of_range <= warning + hysteresis:
                return "warning"
            return "info"

        # info state: require additional downward movement before entering warning.
        if percent_of_range <= critical:
            return "critical"
        if percent_of_range <= max(0.0, warning - hysteresis):
            return "warning"
        return "info"

    def _derive_transition(
        self,
        severity: str,
        previous_event: BatteryStatusEvent | None,
        delta_percent: float,
    ) -> str:
        if previous_event is None:
            return f"initial_{severity}"

        previous_severity = previous_event.severity
        if previous_severity != severity:
            if severity == "warning":
                return "enter_warning"
            if severity == "critical":
                return "enter_critical"
            return "recover_info"

        if delta_percent <= -self._rapid_drop_threshold_percent():
            return "delta_drop"
        return f"steady_{severity}"

    def _store_event(self, event: BatteryStatusEvent) -> None:
        clear_event = None
        with self._lock:
            previous_event = self._latest_event
            self._latest_event = event

        if previous_event and previous_event.severity in {"warning", "critical"} and event.severity == "info":
            clear_event = BatteryStatusEvent(
                timestamp=event.timestamp,
                voltage=event.voltage,
                percent_of_range=event.percent_of_range,
                severity="info",
                event_type="clear",
                transition="recover_info",
                delta_percent=event.delta_percent,
                rapid_drop=event.rapid_drop,
            )

        self._emit_events([clear_event, event] if clear_event else [event])

    def _emit_events(self, events: Iterable[BatteryStatusEvent]) -> None:
        handlers = list(self._event_handlers)
        if not handlers:
            return
        for event in events:
            LOGGER.info(
                "[Battery] Emitting event: voltage=%.2fV severity=%s percent=%.2f transition=%s delta=%.2f rapid_drop=%s",
                event.voltage,
                event.severity,
                event.percent_of_range,
                event.transition,
                event.delta_percent,
                event.rapid_drop,
            )
            for handler in handlers:
                try:
                    handler(event)
                except Exception as exc:
                    LOGGER.exception("[Battery] Event handler failed: %s", exc)

    def _warning_threshold(self) -> float:
        return min(max(self._warning_percent / 100.0, 0.0), 1.0)

    def _critical_threshold(self) -> float:
        critical = min(max(self._critical_percent / 100.0, 0.0), 1.0)
        return min(critical, self._warning_threshold())

    def _rapid_drop_threshold_percent(self) -> float:
        return max(self._hysteresis_percent, 5.0)

    def _is_rapid_drop(self, delta_percent: float) -> bool:
        return delta_percent <= -self._rapid_drop_threshold_percent()

    def _load_config(self) -> None:
        config = ConfigController.get_instance().get_config()
        battery_cfg = config.get("battery") or {}
        response_cfg = battery_cfg.get("response") or {}

        self._min_voltage = float(battery_cfg.get("voltage_min", self._min_voltage))
        self._max_voltage = float(battery_cfg.get("voltage_max", self._max_voltage))
        if self._max_voltage <= self._min_voltage:
            LOGGER.warning("[Battery] Invalid voltage range; falling back to defaults.")
            self._min_voltage = 7.0
            self._max_voltage = 8.4

        self._warning_percent = float(battery_cfg.get("warning_percent", self._warning_percent))
        self._critical_percent = float(battery_cfg.get("critical_percent", self._critical_percent))
        self._hysteresis_percent = float(
            battery_cfg.get("hysteresis_percent", self._hysteresis_percent)
        )

        self._response_enabled = bool(response_cfg.get("enabled", self._response_enabled))
        self._response_cooldown_s = max(
            0.0,
            float(response_cfg.get("cooldown_s", self._response_cooldown_s)),
        )
        self._response_allow_warning = bool(
            response_cfg.get("allow_warning", self._response_allow_warning)
        )
        self._response_allow_critical = bool(
            response_cfg.get("allow_critical", self._response_allow_critical)
        )
        self._response_require_transition = bool(
            response_cfg.get("require_transition", self._response_require_transition)
        )

    def _should_request_response(
        self,
        event: BatteryStatusEvent,
        *,
        last_response_severity: str | None,
        last_response_time: float,
    ) -> bool:
        if not self._response_enabled:
            return False
        if event.event_type == "clear" or event.severity == "info":
            return False
        if event.severity == "warning" and not self._response_allow_warning:
            return False
        if event.severity == "critical" and not self._response_allow_critical:
            return False

        # Avoid repeated chat-triggering events when status is unchanged.
        if event.transition.startswith("steady_"):
            return False

        if self._response_require_transition and event.severity == last_response_severity:
            return False

        if self._response_cooldown_s <= 0.0:
            return True
        return time.time() - last_response_time >= self._response_cooldown_s
