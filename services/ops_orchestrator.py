"""Operational orchestrator for background coordination tasks."""

from __future__ import annotations

from dataclasses import replace
import threading
import time

from config import ConfigController
from core.logging import logger as LOGGER
from core.ops_models import (
    BudgetCounters,
    DebouncedState,
    HealthSnapshot,
    HealthStatus,
    ModeState,
    OpsEvent,
)
from services.health_probes import (
    HealthProbeResult,
    probe_audio,
    probe_battery,
    probe_motion,
    probe_network,
    probe_realtime_session,
)


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
        self._health_states: dict[str, DebouncedState] = {}
        self._health_pending: dict[str, DebouncedState] = {}
        self._health_debounce_s = 2.0
        self._network_probe_enabled = False
        self._network_probe_host = "api.openai.com"
        self._network_probe_timeout_s = 2.0
        self._realtime_api: object | None = None
        OpsOrchestrator._instance = self

    @classmethod
    def get_instance(cls) -> "OpsOrchestrator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_loop(self, loop_period_s: float = 1.0, heartbeat_period_s: float = 30.0) -> None:
        if self._loop_thread is None or not self._loop_thread.is_alive():
            self._load_probe_config()
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

    def set_realtime_api(self, realtime_api: object | None) -> None:
        with self._lock:
            self._realtime_api = realtime_api

    def enable_network_probe(self, enabled: bool) -> None:
        with self._lock:
            self._network_probe_enabled = enabled

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
        probe_results = self._run_health_probes()
        changed, overall_status, details = self._apply_health_results(probe_results, now)
        with self._lock:
            self._counters.ticks += 1
            if changed or self._latest_health is None:
                summary = self._summarize_health(overall_status, probe_results)
                self._latest_health = HealthSnapshot(
                    timestamp=timestamp,
                    status=overall_status,
                    summary=summary,
                    details={
                        "mode": self._mode.value,
                        "tick": self._counters.ticks,
                        **details,
                    },
                )
                self._emit_health_snapshot(self._latest_health)
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

    def _load_probe_config(self) -> None:
        config = ConfigController.get_instance().get_config()
        health_cfg = config.get("health") or {}
        self._health_debounce_s = float(health_cfg.get("debounce_s", self._health_debounce_s))
        network_cfg = health_cfg.get("network") or {}
        self._network_probe_enabled = bool(network_cfg.get("enabled", self._network_probe_enabled))
        self._network_probe_host = str(network_cfg.get("host", self._network_probe_host))
        self._network_probe_timeout_s = float(
            network_cfg.get("timeout_s", self._network_probe_timeout_s)
        )

    def _run_health_probes(self) -> list[HealthProbeResult]:
        with self._lock:
            realtime_api = self._realtime_api
            network_enabled = self._network_probe_enabled
            network_host = self._network_probe_host
            network_timeout = self._network_probe_timeout_s

        results = [
            probe_audio(realtime_api),
            probe_battery(),
            probe_motion(),
            probe_realtime_session(realtime_api),
        ]
        if network_enabled:
            results.append(probe_network(network_host, network_timeout))
        return results

    def _apply_health_results(
        self,
        results: list[HealthProbeResult],
        now: float,
    ) -> tuple[bool, HealthStatus, dict[str, str | float | int]]:
        changed = False
        details: dict[str, str | float | int] = {}
        for result in results:
            stable = self._health_states.get(result.name)
            pending = self._health_pending.get(result.name)
            if stable is None:
                self._health_states[result.name] = DebouncedState(
                    status=result.status,
                    since=now,
                    last_update=now,
                )
                changed = True
            elif result.status == stable.status:
                stable.last_update = now
                if pending:
                    self._health_pending.pop(result.name, None)
            else:
                if pending is None or pending.status != result.status:
                    self._health_pending[result.name] = DebouncedState(
                        status=result.status,
                        since=now,
                        last_update=now,
                    )
                else:
                    pending.last_update = now
                pending = self._health_pending.get(result.name)
                if pending and (now - pending.since) >= self._health_debounce_s:
                    self._health_states[result.name] = DebouncedState(
                        status=pending.status,
                        since=now,
                        last_update=now,
                    )
                    self._health_pending.pop(result.name, None)
                    changed = True

            stable = self._health_states.get(result.name)
            status_value = stable.status.value if stable else result.status.value
            details[f"{result.name}_status"] = status_value
            details[f"{result.name}_summary"] = result.summary
            for key, value in result.details.items():
                if isinstance(value, (int, float, str)):
                    details[f"{result.name}_{key}"] = value

        overall_status = self._derive_overall_status()
        return changed, overall_status, details

    def _derive_overall_status(self) -> HealthStatus:
        states = [state.status for state in self._health_states.values()]
        if not states:
            return HealthStatus.DEGRADED
        if any(status == HealthStatus.FAILING for status in states):
            return HealthStatus.FAILING
        if any(status == HealthStatus.DEGRADED for status in states):
            return HealthStatus.DEGRADED
        return HealthStatus.OK

    def _summarize_health(
        self,
        status: HealthStatus,
        results: list[HealthProbeResult],
    ) -> str:
        if status == HealthStatus.OK:
            return "All systems nominal"
        impacted = [result.name for result in results if result.status != HealthStatus.OK]
        if not impacted:
            return "System health pending"
        if status == HealthStatus.FAILING:
            return f"Critical issues: {', '.join(impacted)}"
        return f"Degraded: {', '.join(impacted)}"

    def _emit_health_snapshot(self, snapshot: HealthSnapshot) -> None:
        event = OpsEvent(
            timestamp=snapshot.timestamp,
            event_type="health_snapshot",
            message=snapshot.summary,
            metadata={
                "status": snapshot.status.value,
                "summary": snapshot.summary,
            },
        )
        self._recent_events = (self._recent_events + [event])[-20:]
        LOGGER.info(
            "[Ops] Health snapshot: status=%s summary=%s",
            snapshot.status.value,
            snapshot.summary,
        )
