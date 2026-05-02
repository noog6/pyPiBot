"""Read-only runtime situation snapshot projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from enum import Enum
import time
from typing import Any

from services import tool_runtime


@dataclass(frozen=True)
class InteractionSnapshot:
    state: str
    active_input_event_key: str | None
    active_canonical_key: str | None
    listening: bool


@dataclass(frozen=True)
class ResponseSnapshot:
    active_response_id: str | None
    active_response_origin: str
    response_in_flight: bool
    pending_response_create: bool
    pending_response_create_queue_depth: int
    last_response_create_ts: float | None


@dataclass(frozen=True)
class ContinuitySnapshot:
    continuity: dict[str, Any]


@dataclass(frozen=True)
class ToolFollowupSnapshot:
    tool_followup_state_count: int
    tool_followup_states: dict[str, str]


@dataclass(frozen=True)
class MotionSnapshot:
    active_request_count: int
    is_busy: bool
    active_requests: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class VisionSnapshot:
    camera_controller_present: bool
    vision_queue_depth: int


@dataclass(frozen=True)
class StartupSnapshot:
    injection_ready: bool
    injection_ready_reason: str | None
    session_ready: bool


@dataclass(frozen=True)
class ModelSnapshot:
    realtime_model: str | None
    voice: str | None


@dataclass(frozen=True)
class SessionSnapshot:
    connected: bool
    connection_attempts: int
    connections: int
    reconnects: int
    failures: int


def _startup_summary_token(startup: StartupSnapshot) -> str:
    if startup.injection_ready:
        return "ready"
    if startup.injection_ready_reason == "response_in_progress":
        return "busy"
    if startup.injection_ready_reason:
        return f"blocked:{startup.injection_ready_reason}"
    return "not_ready"


@dataclass(frozen=True)
class SituationSnapshot:
    timestamp: float
    source: str
    monotonic_time: float
    run_id: str | None
    interaction: InteractionSnapshot
    response: ResponseSnapshot
    continuity: ContinuitySnapshot
    tools: ToolFollowupSnapshot
    motion: MotionSnapshot
    vision: VisionSnapshot
    battery: dict[str, Any]
    startup: StartupSnapshot
    model: ModelSnapshot
    session: SessionSnapshot

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compact_summary(self) -> str:
        obligation = "none"
        continuity = self.continuity.continuity
        if isinstance(continuity, dict):
            counts = continuity.get("counts")
            if isinstance(counts, dict):
                unresolved = int(counts.get("unresolved", 0) or 0)
                commitments = int(counts.get("commitments", 0) or 0)
                if unresolved > 0:
                    obligation = f"unresolved:{unresolved}"
                elif commitments > 0:
                    obligation = f"commitments:{commitments}"
        queue_depth = int(self.response.pending_response_create_queue_depth or 0)
        tool_count = int(self.tools.tool_followup_state_count or 0)
        model_name = self.model.realtime_model or "unknown"
        battery_voltage = self.battery.get("voltage") if isinstance(self.battery, dict) else "unknown"
        if isinstance(battery_voltage, bool):
            battery_token = "unknown"
        elif isinstance(battery_voltage, (int, float)):
            battery_token = f"{battery_voltage:.2f}".rstrip("0").rstrip(".")
        elif isinstance(battery_voltage, str) and battery_voltage.strip():
            battery_token = battery_voltage.strip()
        else:
            battery_token = "unknown"
        return (
            f"state={self.interaction.state} "
            f"response_in_flight={str(self.response.response_in_flight).lower()} "
            f"queue={queue_depth} "
            f"obligation={obligation} "
            f"tools={tool_count} "
            f"motion={'busy' if self.motion.is_busy else 'idle'} "
            f"battery={battery_token} "
            f"startup={_startup_summary_token(self.startup)} "
            f"model={model_name}"
        )


def _copy_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): deepcopy(v) for k, v in value.items()}
    return {}


def _normalize_runtime_identifier(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _resolve_run_id(runtime: Any) -> str | None:
    for candidate in ("_run_id", "run_id", "session_id"):
        resolved = _normalize_runtime_identifier(getattr(runtime, candidate, None))
        if resolved:
            return resolved
    current_run_id = getattr(runtime, "_current_run_id", None)
    if callable(current_run_id):
        try:
            resolved = _normalize_runtime_identifier(current_run_id())
        except Exception:
            resolved = None
        if resolved:
            return resolved
    return None


def _resolve_interaction_state(runtime: Any) -> str:
    state_manager = getattr(runtime, "state_manager", None)
    managed_state = getattr(state_manager, "state", None)
    if managed_state is not None:
        if isinstance(managed_state, Enum):
            return str(managed_state.value or "unknown")
        return str(managed_state or "unknown")
    state = getattr(runtime, "state", None)
    if isinstance(state, Enum):
        return str(state.value or "unknown")
    return str(state or "unknown")


def _resolve_model_voice(runtime: Any) -> str | None:
    for candidate in ("_session_output_voice", "_voice", "voice"):
        resolved = _normalize_runtime_identifier(getattr(runtime, candidate, None))
        if resolved:
            return resolved
    return None


def build_situation_snapshot(runtime: Any, *, health: dict[str, Any] | None = None) -> SituationSnapshot:
    now = time.time()
    monotonic_now = time.monotonic()

    pending = getattr(runtime, "_pending_response_create", None)
    queue = list(getattr(runtime, "_response_create_queue", ()) or ())
    tool_states = _copy_mapping(getattr(runtime, "_tool_followup_state_by_canonical_key", {}))
    if health is None:
        health = runtime.get_session_health() if callable(getattr(runtime, "get_session_health", None)) else {}

    battery_payload = _copy_mapping(tool_runtime.read_cached_battery_status())
    motion_payload = tool_runtime.read_motion_status(limit=20)

    return SituationSnapshot(
        timestamp=now,
        source="ai.situation_snapshot.build_situation_snapshot",
        monotonic_time=monotonic_now,
        run_id=_resolve_run_id(runtime),
        interaction=InteractionSnapshot(
            state=_resolve_interaction_state(runtime),
            active_input_event_key=getattr(runtime, "_active_response_input_event_key", None),
            active_canonical_key=getattr(runtime, "_active_response_canonical_key", None),
            listening=bool(getattr(runtime, "_is_listening", False)),
        ),
        response=ResponseSnapshot(
            active_response_id=getattr(runtime, "_active_response_id", None),
            active_response_origin=str(getattr(runtime, "_active_response_origin", "unknown") or "unknown"),
            response_in_flight=bool(getattr(runtime, "_response_in_flight", False)),
            pending_response_create=pending is not None,
            pending_response_create_queue_depth=len(queue),
            last_response_create_ts=getattr(runtime, "_last_response_create_ts", None),
        ),
        continuity=ContinuitySnapshot(
            continuity=_copy_mapping((health or {}).get("continuity", {})),
        ),
        tools=ToolFollowupSnapshot(
            tool_followup_state_count=len(tool_states),
            tool_followup_states={str(k): str(v) for k, v in tool_states.items()},
        ),
        motion=MotionSnapshot(
            active_request_count=int(motion_payload.get("active_request_count", 0) or 0),
            is_busy=bool(motion_payload.get("is_busy", False)),
            active_requests=tuple(_copy_mapping(item) for item in list(motion_payload.get("active_requests", []) or [])),
        ),
        vision=VisionSnapshot(
            camera_controller_present=getattr(runtime, "camera_controller", None) is not None,
            vision_queue_depth=len(getattr(runtime, "_vision_input_queue", ()) or ()),
        ),
        battery=battery_payload,
        startup=StartupSnapshot(
            injection_ready=bool((health or {}).get("injection_ready", False)),
            injection_ready_reason=(str((health or {}).get("injection_ready_reason")) if (health or {}).get("injection_ready_reason") is not None else None),
            session_ready=bool((health or {}).get("session_ready", False)),
        ),
        model=ModelSnapshot(
            realtime_model=str(getattr(runtime, "_realtime_model", "") or "").strip() or None,
            voice=_resolve_model_voice(runtime),
        ),
        session=SessionSnapshot(
            connected=bool((health or {}).get("connected", False)),
            connection_attempts=int((health or {}).get("connection_attempts", 0) or 0),
            connections=int((health or {}).get("connections", 0) or 0),
            reconnects=int((health or {}).get("reconnects", 0) or 0),
            failures=int((health or {}).get("failures", 0) or 0),
        ),
    )
