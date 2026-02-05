"""Health probes for operational monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import socket
import time
from typing import Any, Mapping

from core.ops_models import HealthStatus
from services.battery_monitor import BatteryMonitor


@dataclass(frozen=True)
class HealthProbeResult:
    """Result of a single subsystem health probe."""

    name: str
    status: HealthStatus
    summary: str
    details: Mapping[str, str | float | int] = field(default_factory=dict)


def probe_audio(realtime_api: Any | None) -> HealthProbeResult:
    """Probe audio input/output readiness."""

    if realtime_api is None:
        return HealthProbeResult(
            name="audio",
            status=HealthStatus.DEGRADED,
            summary="Audio unavailable (realtime API not initialized)",
        )

    audio_player = getattr(realtime_api, "audio_player", None)
    mic = getattr(realtime_api, "mic", None)
    input_ready = mic is not None
    output_ready = audio_player is not None
    if input_ready and output_ready:
        status = HealthStatus.OK
        summary = "Audio input/output ready"
    elif input_ready or output_ready:
        status = HealthStatus.DEGRADED
        summary = "Audio partially available"
    else:
        status = HealthStatus.FAILING
        summary = "Audio unavailable"

    details: dict[str, str | float | int] = {
        "input_ready": int(input_ready),
        "output_ready": int(output_ready),
    }
    input_device = getattr(realtime_api, "_audio_input_device_name", None)
    output_device = getattr(realtime_api, "_audio_output_device_name", None)
    if isinstance(input_device, str) and input_device:
        details["input_device"] = input_device
    if isinstance(output_device, str) and output_device:
        details["output_device"] = output_device
    if mic is not None:
        details["recording"] = int(getattr(mic, "is_recording", False))
        details["receiving"] = int(getattr(mic, "is_receiving", False))

    return HealthProbeResult(
        name="audio",
        status=status,
        summary=summary,
        details=details,
    )


def probe_battery() -> HealthProbeResult:
    """Probe battery monitoring health."""

    if importlib.util.find_spec("smbus") is None:
        return HealthProbeResult(
            name="battery",
            status=HealthStatus.DEGRADED,
            summary="Battery sensor unavailable (smbus missing)",
            details={"smbus": 0},
        )

    try:
        monitor = BatteryMonitor.get_instance()
    except Exception as exc:  # noqa: BLE001 - probe should not raise
        return HealthProbeResult(
            name="battery",
            status=HealthStatus.DEGRADED,
            summary="Battery monitor unavailable",
            details={"error": str(exc)},
        )

    latest = monitor.get_latest_event()
    loop_alive = monitor.is_loop_alive()
    details: dict[str, str | float | int] = {
        "loop_alive": int(loop_alive),
    }
    if latest is None:
        return HealthProbeResult(
            name="battery",
            status=HealthStatus.DEGRADED if loop_alive else HealthStatus.FAILING,
            summary="Battery monitor inactive" if not loop_alive else "Battery monitor warming up",
            details=details,
        )

    details.update(
        {
            "voltage": latest.voltage,
            "percent": latest.percent_of_range,
            "severity": latest.severity,
        }
    )
    if latest.severity == "critical":
        status = HealthStatus.FAILING
        summary = "Battery critical"
    elif latest.severity == "warning":
        status = HealthStatus.DEGRADED
        summary = "Battery low"
    else:
        status = HealthStatus.OK
        summary = "Battery nominal"
    return HealthProbeResult(
        name="battery",
        status=status,
        summary=summary,
        details=details,
    )


def probe_motion() -> HealthProbeResult:
    """Probe motion controller health."""

    if importlib.util.find_spec("smbus") is None:
        return HealthProbeResult(
            name="motion",
            status=HealthStatus.DEGRADED,
            summary="Motion hardware unavailable (smbus missing)",
            details={"smbus": 0},
        )

    try:
        from motion.motion_controller import MotionController

        controller = MotionController.get_instance()
    except Exception as exc:  # noqa: BLE001 - probe should not raise
        return HealthProbeResult(
            name="motion",
            status=HealthStatus.DEGRADED,
            summary="Motion controller unavailable",
            details={"error": str(exc)},
        )

    alive = controller.is_control_loop_alive()
    status = HealthStatus.OK if alive else HealthStatus.DEGRADED
    summary = "Motion loop active" if alive else "Motion loop inactive"
    return HealthProbeResult(
        name="motion",
        status=status,
        summary=summary,
        details={"loop_alive": int(alive)},
    )


def probe_network(host: str, timeout_s: float) -> HealthProbeResult:
    """Probe outbound network connectivity."""

    start = time.monotonic()
    try:
        with socket.create_connection((host, 443), timeout=timeout_s):
            latency_ms = int((time.monotonic() - start) * 1000)
            return HealthProbeResult(
                name="network",
                status=HealthStatus.OK,
                summary=f"Network reachable ({host})",
                details={"latency_ms": latency_ms},
            )
    except Exception as exc:  # noqa: BLE001 - probe should not raise
        return HealthProbeResult(
            name="network",
            status=HealthStatus.DEGRADED,
            summary=f"Network probe failed ({host})",
            details={"error": str(exc)},
        )


def probe_realtime_session(realtime_api: Any | None) -> HealthProbeResult:
    """Probe realtime session connectivity health."""

    if realtime_api is None:
        return HealthProbeResult(
            name="realtime",
            status=HealthStatus.DEGRADED,
            summary="Realtime API not initialized",
        )

    session_health = getattr(realtime_api, "get_session_health", lambda: {})()
    connected = bool(session_health.get("connected"))
    ready = bool(session_health.get("ready"))
    failures = int(session_health.get("failures", 0))
    reconnects = int(session_health.get("reconnects", 0))

    if connected and ready:
        status = HealthStatus.OK
        summary = "Realtime session connected"
    elif connected:
        status = HealthStatus.DEGRADED
        summary = "Realtime session connected (not ready)"
    elif failures > 0:
        status = HealthStatus.FAILING
        summary = "Realtime session disconnected"
    else:
        status = HealthStatus.DEGRADED
        summary = "Realtime session offline"

    details: dict[str, str | float | int] = {
        "connected": int(connected),
        "ready": int(ready),
        "failures": failures,
        "reconnects": reconnects,
    }
    for key in (
        "connection_attempts",
        "connections",
        "last_connect_time",
    ):
        value = session_health.get(key)
        if isinstance(value, (int, float)):
            details[key] = value
    for key in ("last_disconnect_reason", "last_failure_reason"):
        value = session_health.get(key)
        if isinstance(value, str) and value:
            details[key] = value

    return HealthProbeResult(
        name="realtime",
        status=status,
        summary=summary,
        details=details,
    )
