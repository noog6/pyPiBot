"""Tests for startup system context injection."""

from __future__ import annotations

import asyncio
import json
import time

from ai.realtime_api import RealtimeAPI
from services.system_context_coordinator import SystemContextCoordinator


class _RealtimeStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def is_ready_for_injections(self) -> bool:
        return True

    def inject_system_context(self, payload: dict[str, object]) -> None:
        self.calls.append(payload)


class _Health:
    def __init__(self, status_value: str, summary: str) -> None:
        self.status = type("S", (), {"value": status_value})()
        self.summary = summary


class _OpsStub:
    def __init__(self, *, startup_emitted: bool = True, health: _Health | None = None) -> None:
        self._startup_emitted = startup_emitted
        self._health = health

    def has_startup_snapshot_emitted(self) -> bool:
        return self._startup_emitted

    def get_latest_health(self) -> _Health | None:
        return self._health

    def set_latest_health(self, health: _Health | None) -> None:
        self._health = health


class _BatteryMonitorStub:
    def __init__(self, voltage: float | None) -> None:
        self._voltage = voltage

    def get_latest_event(self):
        if self._voltage is None:
            return None
        return type("BatteryEvent", (), {"voltage": self._voltage})()


def test_system_context_coordinator_injects_once_per_run() -> None:
    realtime = _RealtimeStub()
    coordinator = SystemContextCoordinator(
        realtime_api=realtime,
        ops_orchestrator=_OpsStub(startup_emitted=True),
        run_id="run-123",
        boot_time="2026-01-01T00:00:00+00:00",
        semantic_state="ready",
        semantic_reason="provider_ready",
    )

    coordinator.start()
    time.sleep(0.15)
    coordinator.stop()

    assert len(realtime.calls) == 1


def test_system_context_payload_includes_run_id_and_startup_health() -> None:
    coordinator = SystemContextCoordinator(
        realtime_api=_RealtimeStub(),
        ops_orchestrator=_OpsStub(
            startup_emitted=True,
            health=_Health("degraded", "Degraded: audio unavailable"),
        ),
        battery_monitor=_BatteryMonitorStub(7.8912),
        run_id="run-777",
        boot_time="2026-01-01T00:00:00+00:00",
        semantic_state="timeout",
        semantic_reason="startup_canary_timeout",
    )

    payload = coordinator._build_startup_payload()

    assert payload["run_id"] == "run-777"
    assert payload["startup_health"]["status"] == "degraded"
    assert payload["startup_health"]["summary"] == "Degraded: audio unavailable"


def test_system_context_coordinator_injects_second_update_on_transition_to_ok() -> None:
    realtime = _RealtimeStub()
    ops = _OpsStub(
        startup_emitted=True,
        health=_Health("degraded", "Degraded: audio unavailable"),
    )
    coordinator = SystemContextCoordinator(
        realtime_api=realtime,
        ops_orchestrator=ops,
        run_id="run-abc",
        boot_time="2026-01-01T00:00:00+00:00",
        semantic_state="ready",
        semantic_reason="provider_ready",
        poll_interval_s=0.05,
    )

    coordinator.start()
    time.sleep(0.1)
    ops.set_latest_health(_Health("ok", "All systems nominal"))
    time.sleep(0.2)
    coordinator.stop()

    assert len(realtime.calls) == 2
    assert realtime.calls[0]["startup_health"]["status"] == "degraded"
    assert realtime.calls[1]["update"] == "ops_transition_ok"
    assert realtime.calls[1]["ops_health"]["status"] == "ok"


class _Ws:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.events.append(json.loads(payload))


def test_send_system_context_enforces_no_response_create() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.websocket = _Ws()
    api._track_outgoing_event = lambda *args, **kwargs: None

    asyncio.run(api.send_system_context({"source": "system_context", "run_id": "run-5"}))

    event_types = [event["type"] for event in api.websocket.events]
    assert event_types == ["conversation.item.create"]
