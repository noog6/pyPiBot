"""Operational orchestrator for background coordination tasks."""

from __future__ import annotations

from dataclasses import replace
import threading
import time

from core.logging import logger as LOGGER
from core.ops_models import BudgetCounters, HealthSnapshot, HealthStatus, ModeState, OpsEvent


class OpsOrchestrator:
    """Singleton orchestrator with a heartbeat tick loop."""

    _instance: "OpsOrchestrator | None" = None

    def __init__(self) -> None:
        if OpsOrchestrator._instance is not None:
            raise RuntimeError("You cannot create another OpsOrchestrator class")

        self._stop_event = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._loop_period_s = 1.0
        self._heartbeat_period_s = 30.0
        self._next_heartbeat = time.monotonic()
        self._counters = BudgetCounters()
        self._mode = ModeState.STARTUP
        self._latest_health: HealthSnapshot | None = None
        self._recent_events: list[OpsEvent] = []
        OpsOrchestrator._instance = self

    @classmethod
    def get_instance(cls) -> "OpsOrchestrator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_loop(self, loop_period_s: float = 1.0, heartbeat_period_s: float = 30.0) -> None:
        if self._loop_thread is None or not self._loop_thread.is_alive():
            self._loop_period_s = max(loop_period_s, 0.2)
            self._heartbeat_period_s = max(heartbeat_period_s, self._loop_period_s)
            self._next_heartbeat = time.monotonic() + self._heartbeat_period_s
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

    def get_latest_health(self) -> HealthSnapshot | None:
        with self._lock:
            return self._latest_health

    def get_counters(self) -> BudgetCounters:
        with self._lock:
            return replace(self._counters)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                LOGGER.exception("[Ops] Error in tick loop (retrying): %s", exc)
                with self._lock:
                    self._counters.errors += 1
            self._stop_event.wait(timeout=self._loop_period_s)

    def _tick(self) -> None:
        now = time.monotonic()
        timestamp = time.time()
        with self._lock:
            self._counters.ticks += 1
            self._latest_health = HealthSnapshot(
                timestamp=timestamp,
                status=HealthStatus.OK,
                summary="Orchestrator running",
                details={"mode": self._mode.value, "tick": self._counters.ticks},
            )
            self._mode = ModeState.ACTIVE

        if now >= self._next_heartbeat:
            self._next_heartbeat = now + self._heartbeat_period_s
            self._emit_heartbeat(timestamp)

    def _emit_heartbeat(self, timestamp: float) -> None:
        """Heartbeat log stub for future operational events."""

        with self._lock:
            self._counters.heartbeats += 1
            event = OpsEvent(
                timestamp=timestamp,
                event_type="heartbeat",
                message="Orchestrator heartbeat",
                metadata={
                    "mode": self._mode.value,
                    "ticks": self._counters.ticks,
                    "heartbeats": self._counters.heartbeats,
                },
            )
            self._recent_events = (self._recent_events + [event])[-20:]

        LOGGER.info(
            "[Ops] Heartbeat: mode=%s ticks=%s heartbeats=%s",
            event.metadata["mode"],
            event.metadata["ticks"],
            event.metadata["heartbeats"],
        )
