"""Operational orchestrator for background coordination tasks."""

from __future__ import annotations

from dataclasses import replace
import random
import threading
import time

from config import ConfigController
from core.logging import logger as LOGGER
from core.alert_policy import Alert, AlertPolicy
from core.budgeting import RollingWindowBudget
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
    probe_imx500,
)
from motion.gestures import gesture_idle
from motion.motion_controller import MotionController


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
        self._event_bus = None
        self._alert_policy = AlertPolicy()
        self._sensor_budget = RollingWindowBudget(0, 60.0, name="sensor_reads")
        self._micro_presence_budget = RollingWindowBudget(0, 3600.0, name="micro_presence")
        self._log_budget = RollingWindowBudget(0, 60.0, name="logs")
        self._last_probe_results: list[HealthProbeResult] = []
        self._micro_presence_enabled = False
        self._micro_presence_min_s = 60.0
        self._micro_presence_max_s = 180.0
        self._micro_presence_battery_min = 0.2
        self._micro_presence_health_allowed = {HealthStatus.OK, HealthStatus.DEGRADED}
        self._micro_presence_next_ts = time.monotonic()
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

    def stop_loop(self, timeout_s: float = 2.0) -> None:
        if self._loop_thread is not None:
            self._stop_event.set()
            self._loop_thread.join(timeout=timeout_s)
            if self._loop_thread.is_alive():
                LOGGER.warning(
                    "[Ops] Loop thread did not exit within %.2fs; continuing shutdown.",
                    timeout_s,
                )
                return
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

    def set_event_bus(self, event_bus) -> None:
        with self._lock:
            self._event_bus = event_bus

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

        self._maybe_run_micro_presence(now)

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
        if self._should_log_event("heartbeat"):
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
        ops_cfg = config.get("ops") or {}
        budgets_cfg = ops_cfg.get("budgets") or {}
        self._sensor_budget = RollingWindowBudget(
            int(budgets_cfg.get("sensor_reads_per_minute", 0)),
            60.0,
            name="sensor_reads",
        )
        self._micro_presence_budget = RollingWindowBudget(
            int(budgets_cfg.get("micro_presence_per_hour", 0)),
            3600.0,
            name="micro_presence",
        )
        self._log_budget = RollingWindowBudget(
            int(budgets_cfg.get("logs_per_minute", 0)),
            60.0,
            name="logs",
        )
        self._alert_policy = AlertPolicy.from_config(config)
        micro_cfg = ops_cfg.get("micro_presence") or {}
        self._micro_presence_enabled = bool(micro_cfg.get("enabled", self._micro_presence_enabled))
        self._micro_presence_min_s = float(
            micro_cfg.get("min_interval_s", self._micro_presence_min_s)
        )
        self._micro_presence_max_s = float(
            micro_cfg.get("max_interval_s", self._micro_presence_max_s)
        )
        self._micro_presence_battery_min = float(
            micro_cfg.get("battery_min_percent", self._micro_presence_battery_min)
        )
        allowed = micro_cfg.get("health_allowed") or [
            HealthStatus.OK.value,
            HealthStatus.DEGRADED.value,
        ]
        resolved: set[HealthStatus] = set()
        for status in allowed:
            if not isinstance(status, str):
                continue
            try:
                resolved.add(HealthStatus(status))
            except ValueError:
                continue
        if resolved:
            self._micro_presence_health_allowed = resolved
        self._schedule_next_micro_presence(time.monotonic())

    def _run_health_probes(self) -> list[HealthProbeResult]:
        now = time.monotonic()
        if not self._sensor_budget.allow(now):
            self._emit_budget_alert("sensor_reads", "Sensor read budget exhausted.")
            return list(self._last_probe_results)

        self._sensor_budget.record(now)
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
            probe_imx500(),
        ]
        if network_enabled:
            results.append(probe_network(network_host, network_timeout))
        self._last_probe_results = list(results)
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
        if self._should_log_event("health_snapshot"):
            LOGGER.info(
                "[Ops] Health snapshot: status=%s summary=%s",
                snapshot.status.value,
                snapshot.summary,
            )
        self._emit_health_alert(snapshot)

    def _emit_health_alert(self, snapshot: HealthSnapshot) -> None:
        if snapshot.status == HealthStatus.OK:
            return
        severity = "critical" if snapshot.status == HealthStatus.FAILING else "warning"
        self._emit_alert(
            Alert(
                key=f"health_{snapshot.status.value}",
                message=f"Health status {snapshot.status.value}: {snapshot.summary}",
                severity=severity,
                metadata={"summary": snapshot.summary},
                cooldown_s=120.0,
            )
        )

    def _emit_budget_alert(self, key: str, message: str) -> None:
        self._emit_alert(Alert(key=f"budget_{key}", message=message, severity="warning"))

    def _emit_alert(self, alert: Alert) -> None:
        event_bus = None
        with self._lock:
            event_bus = self._event_bus
        if event_bus is None:
            return
        self._alert_policy.emit(event_bus, alert)

    def _should_log_event(self, event_type: str) -> bool:
        now = time.monotonic()
        if self._log_budget.allow(now):
            self._log_budget.record(now)
            return True
        self._emit_budget_alert("logs", f"Log budget exhausted (event={event_type}).")
        return False

    def _schedule_next_micro_presence(self, now: float) -> None:
        min_s = max(0.1, self._micro_presence_min_s)
        max_s = max(min_s, self._micro_presence_max_s)
        self._micro_presence_next_ts = now + random.uniform(min_s, max_s)

    def _maybe_run_micro_presence(self, now: float) -> None:
        if not self._micro_presence_enabled:
            return
        if now < self._micro_presence_next_ts:
            return
        self._schedule_next_micro_presence(now)
        if not self._micro_presence_budget.allow(now):
            self._emit_budget_alert("micro_presence", "Micro-presence budget exhausted.")
            return

        with self._lock:
            snapshot = self._latest_health

        if snapshot and snapshot.status not in self._micro_presence_health_allowed:
            return

        battery_percent = None
        if snapshot:
            battery_value = snapshot.details.get("battery_percent")
            if isinstance(battery_value, (int, float)):
                battery_percent = float(battery_value)
        if battery_percent is not None and battery_percent < self._micro_presence_battery_min:
            return

        try:
            controller = MotionController.get_instance()
        except Exception as exc:
            LOGGER.debug("[Ops] Micro-presence skipped: motion controller unavailable (%s).", exc)
            return

        if not controller.is_control_loop_alive():
            LOGGER.debug("[Ops] Micro-presence skipped: motion loop inactive.")
            return

        if controller.is_moving():
            LOGGER.debug("[Ops] Micro-presence skipped: motion busy.")
            return

        with controller._queue_lock:
            if controller.action_queue:
                LOGGER.debug("[Ops] Micro-presence skipped: action queue not empty.")
                return

        delay_ms = random.randint(100, 350)
        intensity = random.uniform(0.6, 1.0)
        action = gesture_idle(delay_ms=delay_ms, intensity=intensity)
        controller.add_action_to_queue(action)
        self._micro_presence_budget.record(now)
        if self._should_log_event("micro_presence"):
            LOGGER.info(
                "[Ops] Micro-presence gesture queued: delay_ms=%s intensity=%.2f",
                delay_ms,
                intensity,
            )
