"""Focused tests for the runtime diagnostics tool bridge."""

from __future__ import annotations

import sys
import subprocess
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI
from services import tool_runtime
from services.battery_monitor import BatteryStatusEvent


def test_read_runtime_diagnostics_delegates_to_registered_provider_and_adds_host_status(monkeypatch) -> None:
    original_provider = tool_runtime._runtime_diagnostics_provider
    source_payload = {
        "connected": True,
        "ready": False,
        "continuity": {
            "stance": "awaiting_user",
            "stance_detail": "question_pending",
            "settlement": "unresolved_followup",
            "settlement_detail": "question_pending",
            "counts": {
                "current": 1,
                "ongoing": 0,
                "commitments": 0,
                "unresolved": 1,
                "blockers": 0,
                "constraints": 0,
                "recently_closed": 0,
            },
        },
    }

    try:
        monkeypatch.setattr(
            tool_runtime,
            "_collect_host_status",
            lambda: {"wifi_ssid": "TheoWiFi", "primary_ip": "192.168.1.42", "system_uptime": "up 1 hour"},
        )
        tool_runtime.set_runtime_diagnostics_provider(lambda: source_payload)

        payload = tool_runtime.read_runtime_diagnostics()

        assert payload is not source_payload
        assert payload["connected"] is True
        assert payload["ready"] is False
        assert payload["host_status"] == {
            "wifi_ssid": "TheoWiFi",
            "primary_ip": "192.168.1.42",
            "system_uptime": "up 1 hour",
        }
        assert "host_status" not in source_payload
        assert set(payload["continuity"]) == {
            "stance",
            "stance_detail",
            "settlement",
            "settlement_detail",
            "counts",
        }
        assert set(payload["continuity"]["counts"]) == {
            "current",
            "ongoing",
            "commitments",
            "unresolved",
            "blockers",
            "constraints",
            "recently_closed",
        }
    finally:
        tool_runtime.set_runtime_diagnostics_provider(original_provider)


def test_read_runtime_diagnostics_reports_unavailable_without_provider() -> None:
    original_provider = tool_runtime._runtime_diagnostics_provider

    try:
        tool_runtime.set_runtime_diagnostics_provider(None)

        payload = tool_runtime.read_runtime_diagnostics()

        assert payload == {
            "status": "unavailable",
            "message": "Runtime diagnostics are not currently available.",
        }
    finally:
        tool_runtime.set_runtime_diagnostics_provider(original_provider)


def test_realtime_api_registers_session_health_as_runtime_diagnostics_provider(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.get_session_health = lambda: {"ready": True}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tool_runtime,
        "set_runtime_diagnostics_provider",
        lambda provider: captured.setdefault("provider", provider),
    )

    RealtimeAPI._register_runtime_diagnostics_provider(api)

    provider = captured.get("provider")
    assert callable(provider)
    assert provider() == {"ready": True}


def test_collect_host_status_includes_stable_unknown_reasons(monkeypatch) -> None:
    monkeypatch.setattr(tool_runtime, "_read_wifi_ssid", lambda: ("unknown", "wifi_missing"))
    monkeypatch.setattr(tool_runtime, "_read_primary_ip", lambda: ("unknown", "ip_missing"))
    monkeypatch.setattr(tool_runtime, "_read_uptime_pretty", lambda: ("unknown", "uptime_missing"))
    monkeypatch.setattr(
        tool_runtime,
        "_read_load_average",
        lambda: ({"1m": "unknown", "5m": "unknown", "15m": "unknown"}, "load_missing"),
    )
    monkeypatch.setattr(
        tool_runtime,
        "_read_memory_snapshot",
        lambda: {"ram_total": "unknown", "ram_used": "unknown", "ram_available": "unknown", "swap_total": "unknown", "swap_used": "unknown"},
    )
    monkeypatch.setattr(
        tool_runtime,
        "_read_root_disk_snapshot",
        lambda: {"mount": "/", "total": "unknown", "used": "unknown", "available": "unknown", "percent_used": "unknown"},
    )
    monkeypatch.setattr(tool_runtime, "_read_battery_snapshot", lambda: {"amperage": "unknown"})

    payload = tool_runtime._collect_host_status()

    assert payload["wifi_ssid"] == "unknown"
    assert payload["wifi_ssid_reason"] == "wifi_missing"
    assert payload["primary_ip_reason"] == "ip_missing"
    assert payload["system_uptime"] == "unknown"
    assert payload["system_uptime_reason"] == "uptime_missing"
    assert payload["load_average_reason"] == "load_missing"
    assert payload["battery"] == {"amperage": "unknown"}


def test_collect_host_status_includes_battery_amperage_and_power_from_monitor(monkeypatch) -> None:
    event = BatteryStatusEvent(
        timestamp=123.0,
        voltage=7.95,
        percent_of_range=60.0,
        severity="ok",
        amperage=0.58,
        power_watts=4.61,
    )

    class _MonitorStub:
        def get_latest_event(self) -> BatteryStatusEvent:
            return event

    monkeypatch.setattr(tool_runtime, "_read_wifi_ssid", lambda: ("TheoWiFi", None))
    monkeypatch.setattr(tool_runtime, "_read_primary_ip", lambda: ("192.168.1.42", None))
    monkeypatch.setattr(tool_runtime, "_read_uptime_pretty", lambda: ("up 1 hour", None))
    monkeypatch.setattr(tool_runtime, "_read_load_average", lambda: ({"1m": "0.10", "5m": "0.12", "15m": "0.20"}, None))
    monkeypatch.setattr(tool_runtime, "_read_memory_snapshot", lambda: {"ram_total": "1Gi"})
    monkeypatch.setattr(tool_runtime, "_read_root_disk_snapshot", lambda: {"mount": "/"})
    monkeypatch.setattr(tool_runtime.BatteryMonitor, "get_instance", lambda: _MonitorStub())

    payload = tool_runtime._collect_host_status()

    assert payload["battery"] == {
        "voltage": 7.95,
        "amperage": 0.58,
        "power_watts": 4.61,
        "telemetry_source": "battery_monitor",
    }


def test_run_command_returns_none_on_timeout(monkeypatch) -> None:
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=("uptime", "-p"), timeout=1.0)

    monkeypatch.setattr(tool_runtime.subprocess, "run", _raise_timeout)

    assert tool_runtime._run_command(("uptime", "-p")) is None


def test_run_command_returns_none_on_command_failure(monkeypatch) -> None:
    def _raise_called_process_error(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=("iwgetid", "-r"))

    monkeypatch.setattr(tool_runtime.subprocess, "run", _raise_called_process_error)

    assert tool_runtime._run_command(("iwgetid", "-r")) is None


def test_read_primary_ip_prefers_local_route_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        tool_runtime,
        "_run_command",
        lambda args: "1.1.1.1 via 192.168.1.1 dev wlan0 src 192.168.1.42 uid 1000"
        if args == ("ip", "route", "get", "1.1.1.1")
        else None,
    )

    class _SocketShouldNotBeUsed:
        def __init__(self, *args, **kwargs):
            raise AssertionError("socket fallback should not run when route probe succeeded")

    monkeypatch.setattr(tool_runtime.socket, "socket", _SocketShouldNotBeUsed)

    ip, reason = tool_runtime._read_primary_ip()

    assert ip == "192.168.1.42"
    assert reason is None
