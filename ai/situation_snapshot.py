"""Read-only runtime situation snapshot projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
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


def _copy_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(k): deepcopy(v) for k, v in value.items()}
    return {}


def build_situation_snapshot(runtime: Any) -> SituationSnapshot:
    now = time.time()
    monotonic_now = time.monotonic()

    pending = getattr(runtime, "_pending_response_create", None)
    queue = list(getattr(runtime, "_response_create_queue", ()) or ())
    tool_states = _copy_mapping(getattr(runtime, "_tool_followup_state_by_canonical_key", {}))
    health = runtime.get_session_health() if callable(getattr(runtime, "get_session_health", None)) else {}

    battery_payload = _copy_mapping(tool_runtime.read_cached_battery_status())
    motion_payload = tool_runtime.read_motion_status(limit=20)

    return SituationSnapshot(
        timestamp=now,
        source="ai.situation_snapshot.build_situation_snapshot",
        monotonic_time=monotonic_now,
        run_id=str(getattr(runtime, "_run_id", "") or "").strip() or None,
        interaction=InteractionSnapshot(
            state=str(getattr(runtime, "state", "unknown") or "unknown"),
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
            voice=str(getattr(runtime, "_voice", "") or "").strip() or None,
        ),
        session=SessionSnapshot(
            connected=bool((health or {}).get("connected", False)),
            connection_attempts=int((health or {}).get("connection_attempts", 0) or 0),
            connections=int((health or {}).get("connections", 0) or 0),
            reconnects=int((health or {}).get("reconnects", 0) or 0),
            failures=int((health or {}).get("failures", 0) or 0),
        ),
    )
