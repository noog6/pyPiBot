from __future__ import annotations

from dataclasses import asdict

from ai.situation_snapshot import SituationSnapshot, build_situation_snapshot
from services import tool_runtime


class _StubRuntime:
    def __init__(self) -> None:
        self._run_id = "run-123"
        self.state = "listening"
        self._is_listening = True
        self._active_response_input_event_key = "evt-1"
        self._active_response_canonical_key = "turn-1"
        self._active_response_id = "resp-1"
        self._active_response_origin = "user"
        self._response_in_flight = True
        self._pending_response_create = {"id": "pending"}
        self._response_create_queue = [{"id": 1, "meta": {"k": "v"}}, {"id": 2}]
        self._last_response_create_ts = 10.5
        self._tool_followup_state_by_canonical_key = {"turn-1": "pending"}
        self._vision_input_queue = ["a"]
        self.camera_controller = object()
        self._realtime_model = "gpt-realtime"
        self._voice = "alloy"

    def get_session_health(self) -> dict[str, object]:
        return {
            "connected": True,
            "connection_attempts": 1,
            "connections": 1,
            "reconnects": 0,
            "failures": 0,
            "injection_ready": True,
            "injection_ready_reason": "ready",
            "session_ready": True,
            "continuity": {"final_report_owed": False},
        }


def test_build_snapshot_from_stub_runtime(monkeypatch) -> None:
    runtime = _StubRuntime()
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": 12.1})
    monkeypatch.setattr(
        tool_runtime,
        "read_motion_status",
        lambda limit=20: {"active_request_count": 1, "is_busy": True, "active_requests": [{"request_key": "r1"}]},
    )

    snapshot = build_situation_snapshot(runtime)

    assert isinstance(snapshot, SituationSnapshot)
    assert snapshot.run_id == "run-123"
    assert snapshot.response.pending_response_create_queue_depth == 2
    assert snapshot.motion.active_request_count == 1
    assert snapshot.battery["voltage"] == 12.1


def test_build_snapshot_does_not_mutate_source_state(monkeypatch) -> None:
    runtime = _StubRuntime()
    queue_before = list(runtime._response_create_queue)
    tool_before = dict(runtime._tool_followup_state_by_canonical_key)
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": "unknown"})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 0, "is_busy": False, "active_requests": []})

    _ = build_situation_snapshot(runtime)

    assert runtime._response_create_queue == queue_before
    assert runtime._tool_followup_state_by_canonical_key == tool_before


def test_snapshot_serialization_and_copy_semantics(monkeypatch) -> None:
    runtime = _StubRuntime()
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": 11.9})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 1, "is_busy": True, "active_requests": [{"request_key": "r1", "meta": {"nested": 1}}]})

    snapshot = build_situation_snapshot(runtime)
    payload = asdict(snapshot)

    assert payload["source"] == "ai.situation_snapshot.build_situation_snapshot"
    runtime._response_create_queue.append({"id": 3})
    runtime._response_create_queue[0]["meta"]["k"] = "changed"
    runtime._tool_followup_state_by_canonical_key["turn-2"] = "held"
    assert snapshot.response.pending_response_create_queue_depth == 2
    assert snapshot.tools.tool_followup_state_count == 1
    assert snapshot.motion.active_requests[0]["meta"]["nested"] == 1
