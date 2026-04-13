"""Coordinator that injects startup system context and health transition updates."""

from __future__ import annotations

import threading
import time
from typing import Any

from core.logging import logger
from services import tool_runtime


class SystemContextCoordinator:
    """Inject startup context and an optional follow-up when ops reaches OK."""

    def __init__(
        self,
        *,
        realtime_api: Any,
        ops_orchestrator: Any,
        run_id: str,
        boot_time: str,
        semantic_state: str,
        semantic_reason: str,
        prior_run_startup_fact: str | None = None,
        battery_monitor: Any | None = None,
        inject_startup_time_context: bool = True,
        poll_interval_s: float = 0.2,
    ) -> None:
        self._realtime_api = realtime_api
        self._ops_orchestrator = ops_orchestrator
        self._battery_monitor = battery_monitor
        self._run_id = run_id
        self._boot_time = boot_time
        self._semantic_state = semantic_state
        self._semantic_reason = semantic_reason
        self._prior_run_startup_fact = (
            str(prior_run_startup_fact).strip() if prior_run_startup_fact is not None else None
        ) or None
        self._inject_startup_time_context = bool(inject_startup_time_context)
        self._poll_interval_s = max(0.05, poll_interval_s)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._startup_injected = False
        self._ok_transition_injected = False
        self._startup_health_status: str | None = None

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
        while not self._stop_event.is_set():
            if not self._startup_injected:
                if not self._startup_snapshot_ready():
                    self._stop_event.wait(self._poll_interval_s)
                    continue
                ready = self._realtime_api.is_ready_for_injections()
                if not ready:
                    self._stop_event.wait(self._poll_interval_s)
                    continue
                payload = self._build_startup_payload()
                # RealtimeAPI may defer non-critical startup injections until first-turn output settles.
                self._realtime_api.inject_system_context(payload)
                self._startup_injected = True
                self._startup_health_status = str(payload["startup_health"]["status"])
                if self._startup_health_status == "ok":
                    break
                continue

            if self._ok_transition_injected:
                break

            if not self._realtime_api.is_ready_for_injections():
                self._stop_event.wait(self._poll_interval_s)
                continue

            health = self._ops_orchestrator.get_latest_health()
            current_status = self._health_status_value(health)
            if current_status == "ok" and self._startup_health_status != "ok":
                payload = self._build_ops_ok_transition_payload(health)
                self._realtime_api.inject_system_context(payload)
                self._ok_transition_injected = True
                break

            self._stop_event.wait(self._poll_interval_s)

    def _startup_snapshot_ready(self) -> bool:
        if self._ops_orchestrator.has_startup_snapshot_emitted():
            return True
        return self._ops_orchestrator.get_latest_health() is not None

    def _health_status_value(self, health: Any | None) -> str:
        if health is None:
            return "unknown"
        return str(getattr(health.status, "value", str(health.status)))

    def _build_startup_payload(self) -> dict[str, Any]:
        health = self._ops_orchestrator.get_latest_health()
        if health is None:
            health_status = "unknown"
            health_summary = "no health snapshot yet"
        else:
            health_status = self._health_status_value(health)
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

        payload: dict[str, Any] = {
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
        if self._prior_run_startup_fact is not None:
            payload["prior_runtime_session"] = {"fact": self._prior_run_startup_fact}
        if self._inject_startup_time_context:
            startup_time_context = self._build_startup_time_context()
            if startup_time_context:
                payload["startup_time_context"] = startup_time_context
        return payload

    def _build_startup_time_context(self) -> dict[str, Any] | None:
        """Build a one-shot startup time grounding note.

        This payload is low-authority orientation context only and can become stale.
        Live turn-time reasoning should still call get_current_time on demand.
        """

        time_payload = tool_runtime.get_current_time(context={"include_period_of_day": True})
        if not isinstance(time_payload, dict) or str(time_payload.get("status")) != "ok":
            return None
        return {
            "type": "startup_time_context",
            "source": "startup_time_context",
            "local_datetime_iso": time_payload.get("local_datetime_iso"),
            "local_date": time_payload.get("local_date"),
            "local_time": time_payload.get("local_time"),
            "weekday": time_payload.get("weekday"),
            "period_of_day": time_payload.get("period_of_day"),
            "timezone_name": time_payload.get("timezone_name"),
            "utc_offset": time_payload.get("utc_offset"),
            "startup_unix_epoch_ms": time_payload.get("unix_epoch_ms"),
            "freshness": "startup_only_snapshot_can_be_stale_use_get_current_time_for_live_reasoning",
        }

    def _build_ops_ok_transition_payload(self, health: Any) -> dict[str, Any]:
        health_summary = str(getattr(health, "summary", "All systems nominal"))
        logger.info(
            "[SYSCTX] injected run_id=%s update=ops_transition_ok health=ok",
            self._run_id,
        )
        return {
            "source": "system_context",
            "run_id": self._run_id,
            "update": "ops_transition_ok",
            "ops_health": {
                "status": "ok",
                "summary": health_summary,
            },
        }
