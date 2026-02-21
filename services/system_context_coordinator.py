"""Coordinator that injects one-time startup system context into realtime."""

from __future__ import annotations

import threading
import time
from typing import Any

from core.logging import logger


class SystemContextCoordinator:
    """Inject startup context once after ops startup and realtime readiness."""

    def __init__(
        self,
        *,
        realtime_api: Any,
        ops_orchestrator: Any,
        run_id: str,
        boot_time: str,
        semantic_state: str,
        semantic_reason: str,
        battery_monitor: Any | None = None,
        poll_interval_s: float = 0.2,
    ) -> None:
        self._realtime_api = realtime_api
        self._ops_orchestrator = ops_orchestrator
        self._battery_monitor = battery_monitor
        self._run_id = run_id
        self._boot_time = boot_time
        self._semantic_state = semantic_state
        self._semantic_reason = semantic_reason
        self._poll_interval_s = max(0.05, poll_interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._injected = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 0.5) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop_event.is_set() and not self._injected:
            if not self._startup_snapshot_ready():
                self._stop_event.wait(self._poll_interval_s)
                continue
            if not self._realtime_api.is_ready_for_injections():
                self._stop_event.wait(self._poll_interval_s)
                continue
            payload = self._build_payload()
            self._realtime_api.inject_system_context(payload)
            self._injected = True

    def _startup_snapshot_ready(self) -> bool:
        if self._ops_orchestrator.has_startup_snapshot_emitted():
            return True
        return self._ops_orchestrator.get_latest_health() is not None

    def _build_payload(self) -> dict[str, Any]:
        health = self._ops_orchestrator.get_latest_health()
        if health is None:
            health_status = "unknown"
            health_summary = "no health snapshot yet"
        else:
            health_status = getattr(health.status, "value", str(health.status))
            health_summary = health.summary

        battery_voltage = None
        if self._battery_monitor is not None:
            event = self._battery_monitor.get_latest_event()
            if event is not None:
                battery_voltage = round(float(event.voltage), 3)

        logger.info(
            "[SYSCTX] injected run_id=%s health=%s semantic=%s battery_v=%s",
            self._run_id,
            health_status,
            self._semantic_state,
            f"{battery_voltage:.3f}" if isinstance(battery_voltage, float) else "unknown",
        )

        return {
            "source": "system_context",
            "run_id": self._run_id,
            "boot_time": self._boot_time,
            "startup_health": {
                "status": health_status,
                "summary": health_summary,
            },
            "semantic_readiness": {
                "state": self._semantic_state,
                "reason": self._semantic_reason,
            },
            "battery_startup_v": battery_voltage,
        }
