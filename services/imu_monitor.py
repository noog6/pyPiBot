"""IMU monitor for sampling and deriving motion events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Callable, Deque, Iterable

from hardware.icm20948_sensor import ICM20948Sensor


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImuSample:
    """Thread-safe snapshot of the most recent IMU sample."""

    timestamp: float
    roll: float
    pitch: float
    yaw: float
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]
    mag: tuple[float, float, float]


@dataclass(frozen=True)
class ImuMotionEvent:
    """Derived IMU event for local behavior or LLM context."""

    timestamp: float
    event_type: str
    severity: str
    details: dict[str, float | str]


class ImuMonitor:
    """Singleton IMU monitor with a background sampling loop."""

    _instance: "ImuMonitor | None" = None

    def __init__(self) -> None:
        if ImuMonitor._instance is not None:
            raise RuntimeError("You cannot create another ImuMonitor class")

        self._sensor = ICM20948Sensor.get_instance()
        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_sample: ImuSample | None = None
        self._event_history: Deque[ImuMotionEvent] = deque(maxlen=50)
        self._event_handlers: set[Callable[[ImuMotionEvent], None]] = set()
        self._last_event_times: dict[str, float] = {}
        self._loop_period_s = 0.05
        self._mag_period_s = 0.2
        self._tilt_threshold_deg = 45.0
        self._gyro_threshold_dps = 180.0
        self._roll_rate_threshold = 30.0
        self._min_event_interval_s = 0.5
        ImuMonitor._instance = self

    @classmethod
    def get_instance(cls) -> "ImuMonitor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_loop(self, loop_period_s: float = 0.05, mag_period_s: float = 0.2) -> None:
        if self._loop_thread is None or not self._loop_thread.is_alive():
            self._loop_period_s = max(loop_period_s, 0.01)
            self._mag_period_s = max(mag_period_s, self._loop_period_s)
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

    def get_latest_sample(self) -> ImuSample | None:
        with self._lock:
            return self._latest_sample

    def get_recent_events(self, limit: int = 10) -> list[ImuMotionEvent]:
        with self._lock:
            return list(self._event_history)[-limit:]

    def register_event_handler(self, handler: Callable[[ImuMotionEvent], None]) -> None:
        self._event_handlers.add(handler)

    def unregister_event_handler(self, handler: Callable[[ImuMotionEvent], None]) -> None:
        self._event_handlers.discard(handler)

    def get_context_block(self, event_limit: int = 5) -> str:
        sample = self.get_latest_sample()
        if sample is None:
            return "IMU context: no samples available."

        events = self.get_recent_events(limit=event_limit)
        event_lines = "\n".join(
            f"- {event.event_type} ({event.severity}) @ {event.timestamp:.2f}s {event.details}"
            for event in events
        )
        if not event_lines:
            event_lines = "- None"
        return (
            "IMU context:\n"
            f"- roll/pitch/yaw: {sample.roll:.2f}, {sample.pitch:.2f}, {sample.yaw:.2f}\n"
            f"- accel: {sample.accel}\n"
            f"- gyro: {sample.gyro}\n"
            f"- mag: {sample.mag}\n"
            "- recent_events:\n"
            f"{event_lines}"
        )

    def _loop(self) -> None:
        next_sample_time = time.monotonic()
        next_mag_time = time.monotonic()
        last_sample: ImuSample | None = None

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= next_sample_time:
                next_sample_time = now + self._loop_period_s
                try:
                    sample = self._read_sample(now, next_mag_time)
                    if now >= next_mag_time:
                        next_mag_time = now + self._mag_period_s
                    events = self._detect_events(sample, last_sample)
                    last_sample = sample
                    self._store_sample(sample, events)
                except Exception as exc:
                    LOGGER.exception("[IMU] Error in loop (retrying): %s", exc)
            else:
                time.sleep(0.005)

    def _read_sample(self, now: float, next_mag_time: float) -> ImuSample:
        self._sensor.icm20948_Gyro_Accel_Read()
        if now >= next_mag_time:
            self._sensor.icm20948MagRead()
        self._sensor.icm20948CalAvgValue()
        self._sensor.imuAHRSupdate(
            self._sensor.MotionVal[0] * 0.0175,
            self._sensor.MotionVal[1] * 0.0175,
            self._sensor.MotionVal[2] * 0.0175,
            self._sensor.MotionVal[3],
            self._sensor.MotionVal[4],
            self._sensor.MotionVal[5],
            self._sensor.MotionVal[6],
            self._sensor.MotionVal[7],
            self._sensor.MotionVal[8],
        )
        roll = self._sensor.calc_roll_degrees()
        pitch = self._sensor.calc_pitch_degrees()
        yaw = self._sensor.calc_yaw_degrees()
        accel = (
            float(self._sensor.MotionVal[3]),
            float(self._sensor.MotionVal[4]),
            float(self._sensor.MotionVal[5]),
        )
        gyro = (
            float(self._sensor.MotionVal[0]),
            float(self._sensor.MotionVal[1]),
            float(self._sensor.MotionVal[2]),
        )
        mag = (
            float(self._sensor.MotionVal[6]),
            float(self._sensor.MotionVal[7]),
            float(self._sensor.MotionVal[8]),
        )
        return ImuSample(
            timestamp=time.time(),
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            accel=accel,
            gyro=gyro,
            mag=mag,
        )

    def _detect_events(self, sample: ImuSample, previous: ImuSample | None) -> Iterable[ImuMotionEvent]:
        events: list[ImuMotionEvent] = []
        now = sample.timestamp

        if abs(sample.roll) > self._tilt_threshold_deg or abs(sample.pitch) > self._tilt_threshold_deg:
            event = ImuMotionEvent(
                timestamp=now,
                event_type="tilt",
                severity="warning",
                details={"roll": sample.roll, "pitch": sample.pitch},
            )
            LOGGER.debug(
                "[IMU] Tilt detected: roll=%.2f pitch=%.2f threshold=%.2f",
                sample.roll,
                sample.pitch,
                self._tilt_threshold_deg,
            )
            events.append(event)

        gyro_mag = math.sqrt(sum(axis * axis for axis in sample.gyro))
        if gyro_mag > self._gyro_threshold_dps:
            event = ImuMotionEvent(
                timestamp=now,
                event_type="spin",
                severity="notice",
                details={"gyro_dps": gyro_mag},
            )
            LOGGER.debug(
                "[IMU] Spin detected: gyro_dps=%.2f threshold=%.2f",
                gyro_mag,
                self._gyro_threshold_dps,
            )
            events.append(event)

        if previous is not None:
            roll_rate = abs(sample.roll - previous.roll)
            pitch_rate = abs(sample.pitch - previous.pitch)
            if roll_rate > self._roll_rate_threshold or pitch_rate > self._roll_rate_threshold:
                event = ImuMotionEvent(
                    timestamp=now,
                    event_type="shake",
                    severity="notice",
                    details={"roll_delta": roll_rate, "pitch_delta": pitch_rate},
                )
                LOGGER.debug(
                    "[IMU] Shake detected: roll_delta=%.2f pitch_delta=%.2f threshold=%.2f",
                    roll_rate,
                    pitch_rate,
                    self._roll_rate_threshold,
                )
                events.append(event)

        return self._rate_limit_events(events)

    def _rate_limit_events(self, events: Iterable[ImuMotionEvent]) -> list[ImuMotionEvent]:
        filtered: list[ImuMotionEvent] = []
        for event in events:
            last_time = self._last_event_times.get(event.event_type)
            if last_time is None or (event.timestamp - last_time) >= self._min_event_interval_s:
                filtered.append(event)
                self._last_event_times[event.event_type] = event.timestamp
        return filtered

    def _store_sample(self, sample: ImuSample, events: Iterable[ImuMotionEvent]) -> None:
        with self._lock:
            self._latest_sample = sample
            for event in events:
                self._event_history.append(event)
        if events:
            self._emit_events(events)

    def _emit_events(self, events: Iterable[ImuMotionEvent]) -> None:
        handlers = list(self._event_handlers)
        if not handlers:
            return
        for event in events:
            LOGGER.info(
                "[IMU] Emitting event: type=%s severity=%s details=%s",
                event.event_type,
                event.severity,
                event.details,
            )
            for handler in handlers:
                try:
                    handler(event)
                except Exception as exc:
                    LOGGER.exception("[IMU] Event handler failed: %s", exc)
