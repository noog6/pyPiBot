"""Shared hardware and motion operations used by AI tool handlers."""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from collections.abc import Callable
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from hardware import ADS1015Sensor, LPS22HBSensor
from motion import (
    MotionController,
    gesture_attention_snap,
    gesture_curious_tilt,
    gesture_idle,
    gesture_look_around,
    gesture_look_center,
    gesture_look_down,
    gesture_look_left,
    gesture_look_right,
    gesture_look_up,
    gesture_no,
    gesture_nod,
)
from services.battery_monitor import BatteryMonitor
from services.output_volume import OutputVolumeController


logger = logging.getLogger(__name__)


_runtime_diagnostics_provider: Callable[[], dict[str, Any]] | None = None
_MOTION_TRACKABLE_GESTURES = frozenset(
    {
        "gesture_idle",
        "gesture_nod",
        "gesture_no",
        "gesture_look_around",
        "gesture_look_up",
        "gesture_look_left",
        "gesture_look_right",
        "gesture_look_down",
        "gesture_look_center",
        "gesture_curious_tilt",
        "gesture_attention_snap",
    }
)
_motion_state_lock = threading.Lock()
_motion_state_by_request_key: dict[str, dict[str, Any]] = {}
_motion_request_key_by_tool_call_id: dict[str, str] = {}
_motion_callbacks_registered_for_controller_id: int | None = None


def set_runtime_diagnostics_provider(provider: Callable[[], dict[str, Any]] | None) -> None:
    """Register the read-only runtime diagnostics provider used by AI tools."""

    global _runtime_diagnostics_provider
    _runtime_diagnostics_provider = provider


def read_runtime_diagnostics() -> dict[str, Any]:
    """Return the current runtime diagnostics bundle if a provider is registered."""

    provider = _runtime_diagnostics_provider
    if not callable(provider):
        return {
            "status": "unavailable",
            "message": "Runtime diagnostics are not currently available.",
        }
    payload = provider()
    if isinstance(payload, dict):
        diagnostics = dict(payload)
        diagnostics["host_status"] = _collect_host_status()
        return diagnostics
    return {
        "status": "unavailable",
        "message": "Runtime diagnostics provider returned an unexpected payload.",
    }


def get_session_ledger_status(
    lookback_runs: int = 1,
    *,
    include_current: bool = False,
) -> dict[str, Any]:
    """Return a compact, deterministic snapshot of recent durable session-ledger runs."""

    requested_lookback = int(lookback_runs)
    clamped_lookback = max(1, min(requested_lookback, 5))

    from storage.controller import StorageController

    storage = StorageController.get_instance()
    records = storage.get_recent_session_records(
        lookback_runs=clamped_lookback,
        include_current=bool(include_current),
    )

    run_items = [
        {
            "run_id": record.canonical_run_id,
            "run_number": record.run_number,
            "shutdown_clean": record.shutdown_clean,
            "lifecycle_state": record.lifecycle_state,
            "started_at": record.started_at,
            "ready_at": record.ready_at,
            "last_seen_at": record.last_seen_at,
            "shutdown_completed_at": record.shutdown_completed_at,
            "interpretation": _session_ledger_run_interpretation(shutdown_clean=record.shutdown_clean),
        }
        for record in records
    ]

    return {
        "status": "ok",
        "lookback_runs": clamped_lookback,
        "include_current": bool(include_current),
        "returned_runs": len(run_items),
        "runs": run_items,
        "summary_text": _build_session_ledger_summary(run_items),
        "interpretation_notes": _session_ledger_interpretation_notes(),
    }


def _session_ledger_run_interpretation(*, shutdown_clean: bool) -> str:
    return "clean shutdown recorded" if shutdown_clean else "no clean shutdown recorded"


def _build_session_ledger_summary(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return "no recorded prior run history available"
    return "; ".join(f"{item['run_id']}: {item['interpretation']}" for item in runs)


def _session_ledger_interpretation_notes() -> dict[str, str]:
    return {
        "shutdown_clean_true": "appears to have ended cleanly",
        "shutdown_clean_false": "appears to have ended unexpectedly or has no clean shutdown marker",
        "empty_history": "no recorded prior run history available",
        "lifecycle_state_guardrail": "treat lifecycle_state as ledger state only; do not infer beyond recorded markers",
    }


def _collect_host_status() -> dict[str, Any]:
    wifi_ssid, wifi_reason = _read_wifi_ssid()
    primary_ip, ip_reason = _read_primary_ip()
    uptime_pretty, uptime_reason = _read_uptime_pretty()
    load_average, load_reason = _read_load_average()

    status: dict[str, Any] = {
        "wifi_ssid": wifi_ssid,
        "primary_ip": primary_ip,
        "system_uptime": uptime_pretty,
        "load_average": load_average,
        "memory": _read_memory_snapshot(),
        "disk_root": _read_root_disk_snapshot(),
        "battery": _read_battery_snapshot(),
    }
    if wifi_ssid == "unknown" and wifi_reason:
        status["wifi_ssid_reason"] = wifi_reason
    if primary_ip == "unknown" and ip_reason:
        status["primary_ip_reason"] = ip_reason
    if uptime_pretty == "unknown" and uptime_reason:
        status["system_uptime_reason"] = uptime_reason
    if any(value == "unknown" for value in load_average.values()) and load_reason:
        status["load_average_reason"] = load_reason
    return status


def _read_battery_snapshot() -> dict[str, Any]:
    return read_cached_battery_status()


def read_cached_battery_status() -> dict[str, Any]:
    """Return cached battery-monitor telemetry without triggering fresh sensor reads."""

    latest_event = BatteryMonitor.get_instance().get_latest_event()
    if latest_event is None:
        return {
            "voltage": "unknown",
            "amperage": "unknown",
            "power_watts": "unknown",
            "telemetry_source": "battery_monitor",
            "reason": "no_sample_available",
        }
    return {
        "voltage": latest_event.voltage,
        "amperage": latest_event.amperage if latest_event.amperage is not None else "unavailable",
        "power_watts": latest_event.power_watts if latest_event.power_watts is not None else "unavailable",
        "telemetry_source": "battery_monitor",
    }


def _motion_request_key(action: Any) -> str:
    existing = str(getattr(action, "_motion_request_key", "") or "").strip()
    if existing:
        return existing
    key = (
        f"{str(getattr(action, 'name', 'unknown') or 'unknown')}:"
        f"{int(getattr(action, 'timestamp', 0) or 0)}"
    )
    setattr(action, "_motion_request_key", key)
    return key


def _register_motion_queued(action: Any) -> str:
    now = time.monotonic()
    action_name = str(getattr(action, "name", "") or "").strip()
    action_obj_id = id(action)
    base_request_key = _motion_request_key(action)
    with _motion_state_lock:
        request_key = base_request_key
        state = _motion_state_by_request_key.get(request_key)
        if state is not None and int(state.get("_action_obj_id", -1)) != action_obj_id:
            request_key = f"{base_request_key}:{action_obj_id}"
            state = _motion_state_by_request_key.get(request_key)
        setattr(action, "_motion_request_key", request_key)
        if state is None:
            state = {
                "request_key": request_key,
                "gesture": action_name or "unknown",
                "status": "queued",
                "queued_monotonic_s": now,
                "started_monotonic_s": None,
                "completed_monotonic_s": None,
                "tool_call_id": None,
                "correlation_source": "action.name+action.timestamp",
                "_action_obj_id": action_obj_id,
            }
            _motion_state_by_request_key[request_key] = state
        else:
            state["status"] = "queued"
            state["queued_monotonic_s"] = now
            state["_action_obj_id"] = action_obj_id
    logger.info(
        "gesture_motion_registered state=queued request_key=%s gesture=%s correlation_source=%s",
        request_key,
        action_name or "unknown",
        "action.name+action.timestamp",
    )
    return request_key


def register_tool_call_motion_request(*, tool_call_id: str | None, motion_request_key: str | None) -> None:
    normalized_call_id = str(tool_call_id or "").strip()
    normalized_request_key = str(motion_request_key or "").strip()
    if not normalized_call_id or not normalized_request_key:
        return
    with _motion_state_lock:
        _motion_request_key_by_tool_call_id[normalized_call_id] = normalized_request_key
        state = _motion_state_by_request_key.get(normalized_request_key)
        if state is not None:
            state["tool_call_id"] = normalized_call_id
    logger.info(
        "gesture_motion_registered state=bound_tool_call request_key=%s tool_call_id=%s",
        normalized_request_key,
        normalized_call_id,
    )


def get_tool_call_motion_state(tool_call_id: str | None) -> dict[str, Any] | None:
    normalized_call_id = str(tool_call_id or "").strip()
    if not normalized_call_id:
        return None
    with _motion_state_lock:
        request_key = _motion_request_key_by_tool_call_id.get(normalized_call_id)
        if not request_key:
            return None
        state = _motion_state_by_request_key.get(request_key)
        if not isinstance(state, dict):
            return None
        return {key: value for key, value in state.items() if not str(key).startswith("_")}




def read_motion_status(*, limit: int = 20) -> dict[str, Any]:
    """Return a read-only summary of tracked motion requests."""

    bounded_limit = max(0, int(limit))
    with _motion_state_lock:
        states = [
            {key: value for key, value in state.items() if not str(key).startswith("_")}
            for state in _motion_state_by_request_key.values()
            if isinstance(state, dict)
        ]
    states.sort(key=lambda item: float(item.get("queued_monotonic_s") or 0.0), reverse=True)
    active = [item for item in states if str(item.get("status") or "") in {"queued", "started"}]
    return {
        "active_request_count": len(active),
        "is_busy": bool(active),
        "active_requests": active[:bounded_limit],
    }
def is_tool_call_motion_completed(tool_call_id: str | None) -> bool | None:
    state = get_tool_call_motion_state(tool_call_id)
    if state is None:
        return None
    return str(state.get("status") or "").strip() == "completed"


def _on_motion_action_lifecycle(event: str, action: Any) -> None:
    # Contract note: "started" reflects dequeue/enter-execution in controller flow.
    # It is not a guarantee that the first servo increment has already been emitted.
    action_name = str(getattr(action, "name", "") or "").strip()
    if action_name not in _MOTION_TRACKABLE_GESTURES:
        return
    request_key = _motion_request_key(action)
    now = time.monotonic()
    with _motion_state_lock:
        state = _motion_state_by_request_key.get(request_key)
        if state is None:
            state = {
                "request_key": request_key,
                "gesture": action_name,
                "status": "unknown",
                "queued_monotonic_s": None,
                "started_monotonic_s": None,
                "completed_monotonic_s": None,
                "tool_call_id": None,
                "correlation_source": "action.name+action.timestamp",
                "_action_obj_id": id(action),
            }
            _motion_state_by_request_key[request_key] = state
        if event == "started":
            state["status"] = "started"
            state["started_monotonic_s"] = now
        elif event == "completed":
            state["status"] = "completed"
            state["completed_monotonic_s"] = now
    if event in {"started", "completed"}:
        logger.info(
            "gesture_motion_registered state=%s request_key=%s gesture=%s",
            event,
            request_key,
            action_name,
        )


def _ensure_motion_callbacks_registered(controller: MotionController) -> None:
    global _motion_callbacks_registered_for_controller_id
    controller_id = id(controller)
    if _motion_callbacks_registered_for_controller_id == controller_id:
        return
    register_fn = getattr(controller, "register_action_lifecycle_callback", None)
    if callable(register_fn):
        register_fn(_on_motion_action_lifecycle)
        _motion_callbacks_registered_for_controller_id = controller_id


def _read_wifi_ssid() -> tuple[str, str | None]:
    ssid = _run_command(("iwgetid", "-r"))
    if ssid:
        return ssid, None

    nmcli_output = _run_command(("nmcli", "-t", "-f", "active,ssid", "dev", "wifi"))
    if nmcli_output:
        for line in nmcli_output.splitlines():
            if line.startswith("yes:"):
                active_ssid = line.partition(":")[2].strip()
                if active_ssid:
                    return active_ssid, None
    return "unknown", "wireless_ssid_unavailable"


def _read_primary_ip() -> tuple[str, str | None]:
    route_probe = _run_command(("ip", "route", "get", "1.1.1.1"))
    if route_probe:
        route_match = re.search(r"\bsrc\s+(\S+)", route_probe)
        if route_match:
            return route_match.group(1), None

    # Route-resolution trick via UDP socket connect; no shell command execution.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip:
                return ip, None
    except OSError:
        pass

    hostname_ips = _run_command(("hostname", "-I"))
    if hostname_ips:
        first_ip = hostname_ips.split()[0].strip()
        if first_ip:
            return first_ip, None
    return "unknown", "primary_ip_unavailable"


def _read_uptime_pretty() -> tuple[str, str | None]:
    uptime_pretty = _run_command(("uptime", "-p"))
    if uptime_pretty:
        return uptime_pretty, None
    try:
        with open("/proc/uptime", encoding="utf-8") as uptime_file:
            uptime_seconds = int(float(uptime_file.read().split()[0]))
    except (OSError, ValueError, IndexError):
        return "unknown", "uptime_unavailable"
    return _format_uptime_from_seconds(uptime_seconds), None


def _read_load_average() -> tuple[dict[str, float | str], str | None]:
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
        return {
            "1m": round(load_1m, 2),
            "5m": round(load_5m, 2),
            "15m": round(load_15m, 2),
        }, None
    except OSError:
        pass
    try:
        with open("/proc/loadavg", encoding="utf-8") as loadavg_file:
            values = loadavg_file.read().split()[:3]
        if len(values) == 3:
            return {
                "1m": round(float(values[0]), 2),
                "5m": round(float(values[1]), 2),
                "15m": round(float(values[2]), 2),
            }, None
    except (OSError, ValueError):
        pass
    unknown_load = {"1m": "unknown", "5m": "unknown", "15m": "unknown"}
    return unknown_load, "load_average_unavailable"


def _read_memory_snapshot() -> dict[str, str]:
    meminfo = _read_meminfo()
    if meminfo is None:
        return {
            "ram_total": "unknown",
            "ram_used": "unknown",
            "ram_available": "unknown",
            "swap_total": "unknown",
            "swap_used": "unknown",
        }
    ram_total = meminfo.get("MemTotal", 0)
    ram_available = meminfo.get("MemAvailable", 0)
    swap_total = meminfo.get("SwapTotal", 0)
    swap_free = meminfo.get("SwapFree", 0)
    ram_used = max(ram_total - ram_available, 0)
    swap_used = max(swap_total - swap_free, 0)
    return {
        "ram_total": _format_bytes(ram_total),
        "ram_used": _format_bytes(ram_used),
        "ram_available": _format_bytes(ram_available),
        "swap_total": _format_bytes(swap_total),
        "swap_used": _format_bytes(swap_used),
    }


def _read_root_disk_snapshot() -> dict[str, str]:
    try:
        total, used, free = shutil.disk_usage("/")
    except OSError:
        return {
            "mount": "/",
            "total": "unknown",
            "used": "unknown",
            "available": "unknown",
            "percent_used": "unknown",
        }
    percent_used = int(round((used / total) * 100)) if total > 0 else 0
    return {
        "mount": "/",
        "total": _format_bytes(total),
        "used": _format_bytes(used),
        "available": _format_bytes(free),
        "percent_used": f"{percent_used}%",
    }


def _run_command(args: tuple[str, ...]) -> str | None:
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value or None


def _read_meminfo() -> dict[str, int] | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo_file:
            lines = meminfo_file.read().splitlines()
    except OSError:
        return None
    values: dict[str, int] = {}
    for line in lines:
        key, _, rest = line.partition(":")
        if not rest:
            continue
        amount_text = rest.strip().split()[0]
        try:
            amount_kib = int(amount_text)
        except ValueError:
            continue
        values[key] = amount_kib * 1024
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if not required.issubset(values):
        return None
    return values


def _format_uptime_from_seconds(total_seconds: int) -> str:
    minutes = total_seconds // 60
    days, rem_minutes = divmod(minutes, 24 * 60)
    hours, mins = divmod(rem_minutes, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days} day" + ("" if days == 1 else "s"))
    if hours:
        parts.append(f"{hours} hour" + ("" if hours == 1 else "s"))
    if mins or not parts:
        parts.append(f"{mins} minute" + ("" if mins == 1 else "s"))
    return "up " + ", ".join(parts)


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(max(value, 0))
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0 or size >= 10:
        return f"{int(round(size))} {units[unit_index]}"
    return f"{size:.1f} {units[unit_index]}"


_LOOK_CENTER_PAN_DEGREES = 0.0
_LOOK_CENTER_TILT_DEGREES = 0.0
_LOOK_CENTER_EPS_DEGREES = 0.5


def _is_pose_centered(controller: MotionController, *, eps_degrees: float = _LOOK_CENTER_EPS_DEGREES) -> bool:
    current_position = getattr(controller, "current_servo_position", {}) or {}
    try:
        pan = float(current_position.get("pan", 0.0))
        tilt = float(current_position.get("tilt", 0.0))
    except (TypeError, ValueError, AttributeError):
        return False
    return (
        abs(pan - _LOOK_CENTER_PAN_DEGREES) <= eps_degrees
        and abs(tilt - _LOOK_CENTER_TILT_DEGREES) <= eps_degrees
    )


def _has_queued_motion_action(controller: MotionController, *, action_name: str) -> bool:
    action_queue = getattr(controller, "action_queue", None) or ()
    return any(str(getattr(action, "name", "") or "").strip() == action_name for action in action_queue)


def _classify_look_center_request(controller: MotionController) -> str:
    current_action = getattr(controller, "current_action", None)
    current_action_name = str(getattr(current_action, "name", "") or "").strip()
    if current_action_name == "gesture_look_center" and bool(controller.is_moving()):
        return "in_progress"
    if _has_queued_motion_action(controller, action_name="gesture_look_center"):
        return "queued"
    if not bool(controller.is_moving()) and current_action is None and _is_pose_centered(controller):
        return "already_satisfied"
    return "new_request"


def read_battery_voltage() -> dict[str, Any]:
    """Return the current LiPo battery voltage via the ADS1015 sensor."""

    latest_event = BatteryMonitor.get_instance().get_latest_event()
    voltage: float
    amperage: float | None = None
    power_watts: float | None = None
    telemetry_source = "battery_monitor"
    if latest_event is not None:
        voltage = latest_event.voltage
        amperage = latest_event.amperage
        power_watts = latest_event.power_watts
    else:
        telemetry_source = "ads1015_direct_read"
        sensor = ADS1015Sensor.get_instance()
        voltage = sensor.read_battery_voltage()
        try:
            amperage = sensor.read_system_amperage()
        except Exception:
            amperage = None
        power_watts = round(voltage * amperage, 2) if amperage is not None else None

    inferred_charger_connected = False
    inference_reason = "no_prior_sample"
    if latest_event is not None:
        inferred_charger_connected = latest_event.inferred_charger_connected
        inference_reason = latest_event.inference_reason
    return {
        "voltage": voltage,
        "amperage": amperage,
        "power_watts": power_watts,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
        "inferred_charger_connected": inferred_charger_connected,
        "inference_reason": inference_reason,
        "telemetry_source": telemetry_source,
    }


def read_environment() -> dict[str, Any]:
    """Return the current onboard air pressure and temperature."""

    sensor = LPS22HBSensor.get_instance()
    air_pressure, air_temperature = sensor.read_value()
    cpu_temperature = None
    cpu_status = "unavailable"
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", result.stdout)
        if match:
            cpu_temperature = float(match.group(1))
            cpu_status = "ok"
    return {
        "air_pressure": air_pressure,
        "air_temperature": air_temperature,
        "air_temperature_context": "LPS22HB onboard ambient sensor inside the chassis",
        "cpu_temperature": cpu_temperature,
        "cpu_temperature_context": "Broadcom SoC core temperature (vcgencmd)",
        "cpu_temperature_status": cpu_status,
        "pressure_unit": "hPa",
        "temperature_unit": "C",
    }


def get_current_time(context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return read-only local time/date grounding for the current instant.

    Usage policy: on-demand primitive only; do not poll repeatedly in a turn.
    """

    normalized_context = context if isinstance(context, dict) else {}
    timezone_value = normalized_context.get("timezone", normalized_context.get("tz"))
    timezone_requested = str(timezone_value).strip() if timezone_value is not None else ""
    include_period_of_day = bool(normalized_context.get("include_period_of_day", True))

    timezone_source = "system_local"
    requested_timezone: str | None = None
    try:
        if timezone_requested:
            tzinfo = ZoneInfo(timezone_requested)
            requested_timezone = timezone_requested
            timezone_source = "request"
        else:
            tzinfo = datetime.now().astimezone().tzinfo
            if tzinfo is None:
                return {
                    "status": "error",
                    "error_code": "timezone_unavailable",
                    "message": "Unable to determine system local timezone.",
                    "requested_timezone": None,
                    "timezone_source": "system_local",
                }
    except ZoneInfoNotFoundError:
        return {
            "status": "error",
            "error_code": "invalid_timezone",
            "message": f"Unknown timezone: {timezone_requested}",
            "requested_timezone": timezone_requested or None,
            "timezone_source": "request",
        }

    now = datetime.now(tzinfo)
    timezone_name = getattr(now.tzinfo, "key", None) or now.tzname() or "unknown"
    utc_offset = now.strftime("%z")
    formatted_utc_offset = (
        f"{utc_offset[:3]}:{utc_offset[3:]}" if len(utc_offset) == 5 else "unknown"
    )

    result: dict[str, Any] = {
        "status": "ok",
        "local_datetime_iso": now.isoformat(timespec="seconds"),
        "local_date": now.strftime("%Y-%m-%d"),
        "local_time": now.strftime("%H:%M:%S"),
        "timezone_name": timezone_name,
        "utc_offset": formatted_utc_offset,
        "weekday": now.strftime("%A"),
        "timezone_source": timezone_source,
        "unix_epoch_ms": int(now.timestamp() * 1000),
    }
    if requested_timezone:
        result["requested_timezone"] = requested_timezone
    if include_period_of_day:
        hour = now.hour
        if 5 <= hour < 12:
            result["period_of_day"] = "morning"
        elif 12 <= hour < 17:
            result["period_of_day"] = "afternoon"
        elif 17 <= hour < 21:
            result["period_of_day"] = "evening"
        else:
            result["period_of_day"] = "night"
    return result


def enqueue_idle_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue an idle gesture action on the motion controller."""

    return _enqueue_gesture(gesture_idle(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_nod_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a nod gesture action on the motion controller."""

    return _enqueue_gesture(gesture_nod(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_no_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a head shake gesture action on the motion controller."""

    return _enqueue_gesture(gesture_no(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_around_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a casual look around gesture action on the motion controller."""

    return _enqueue_gesture(
        gesture_look_around(delay_ms=delay_ms, intensity=float(intensity)),
        delay_ms,
        intensity,
    )


def enqueue_look_up_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look up gesture action on the motion controller."""

    return _enqueue_gesture(gesture_look_up(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_left_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look left gesture action on the motion controller."""

    return _enqueue_gesture(gesture_look_left(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_right_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look right gesture action on the motion controller."""

    return _enqueue_gesture(gesture_look_right(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_down_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look down gesture action on the motion controller."""

    return _enqueue_gesture(gesture_look_down(delay_ms=delay_ms, intensity=float(intensity)), delay_ms, intensity)


def enqueue_look_center_gesture(delay_ms: int = 0) -> dict[str, Any]:
    """Queue a look center gesture action when the body state actually needs it."""

    controller = MotionController.get_instance()
    motion_request_state = _classify_look_center_request(controller)
    result: dict[str, Any] = {
        "queued": False,
        "gesture": "gesture_look_center",
        "delay_ms": delay_ms,
        "motion_request_state": motion_request_state,
    }
    if motion_request_state == "already_satisfied":
        result["message"] = "Already centered."
        result["tool_result_has_distinct_info"] = True
    elif motion_request_state == "in_progress":
        result["message"] = "Already moving back to center."
        result["tool_result_has_distinct_info"] = True
    elif motion_request_state == "queued":
        result["message"] = "Center motion is already queued."
        result["tool_result_has_distinct_info"] = True
    else:
        action = gesture_look_center(delay_ms=delay_ms)
        motion_request_key = _add_action_to_motion_queue(action)
        result["queued"] = True
        result["gesture"] = action.name
        result["motion_request_key"] = motion_request_key
        result["message"] = "Moving back to center."
    logger.info(
        "motion_request_eval tool=%s state=%s no_duplicate_motion_queued=%s delay_ms=%s",
        result["gesture"],
        motion_request_state,
        str(not bool(result.get("queued", False))).lower(),
        delay_ms,
    )
    return result


def enqueue_curious_tilt_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a curious tilt gesture action on the motion controller."""

    return _enqueue_gesture(
        gesture_curious_tilt(delay_ms=delay_ms, intensity=float(intensity)),
        delay_ms,
        intensity,
    )


def enqueue_attention_snap_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a quick attention snap gesture action on the motion controller."""

    return _enqueue_gesture(
        gesture_attention_snap(delay_ms=delay_ms, intensity=float(intensity)),
        delay_ms,
        intensity,
    )


def get_output_volume() -> dict[str, Any]:
    """Return the current output audio volume."""

    controller = OutputVolumeController.get_instance()
    status = controller.get_volume()
    return {
        "percent": status.percent,
        "muted": status.muted,
    }


def set_output_volume(percent: int, emergency: bool = False) -> dict[str, Any]:
    """Set the output audio volume within safe bounds."""

    controller = OutputVolumeController.get_instance()
    status = controller.set_volume(percent=int(percent), emergency=bool(emergency))
    return {
        "percent": status.percent,
        "muted": status.muted,
    }


def _enqueue_gesture(action: Any, delay_ms: int, intensity: float) -> dict[str, Any]:
    motion_request_key = _add_action_to_motion_queue(action)
    return {
        "queued": True,
        "gesture": action.name,
        "delay_ms": delay_ms,
        "intensity": intensity,
        "motion_request_key": motion_request_key,
    }


def _add_action_to_motion_queue(action: Any) -> str:
    controller = MotionController.get_instance()
    _ensure_motion_callbacks_registered(controller)
    motion_request_key = _register_motion_queued(action)
    controller.add_action_to_queue(action)
    return motion_request_key
