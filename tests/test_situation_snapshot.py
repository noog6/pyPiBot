from __future__ import annotations

from dataclasses import asdict
import json
from enum import Enum

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


def test_snapshot_compact_summary_is_deterministic(monkeypatch) -> None:
    runtime = _StubRuntime()
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": 11.9, "raw": {"nested": "ignore"}})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 0, "is_busy": False, "active_requests": []})

    snapshot = build_situation_snapshot(runtime)
    summary = snapshot.compact_summary()

    assert "state=listening" in summary
    assert "queue=2" in summary
    assert "tools=1" in summary
    assert "model=gpt-realtime" in summary
    assert "battery=11.9" in summary
    assert "active_requests" not in summary
    assert "raw" not in summary


def test_snapshot_compact_summary_uses_unknown_battery_token(monkeypatch) -> None:
    runtime = _StubRuntime()
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": "unknown", "samples": [1, 2, 3]})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 0, "is_busy": False, "active_requests": []})

    summary = build_situation_snapshot(runtime).compact_summary()

    assert "battery=unknown" in summary
    assert "samples" not in summary


def test_build_snapshot_uses_precomputed_health_without_runtime_callback(monkeypatch) -> None:
    runtime = _StubRuntime()
    runtime.get_session_health = lambda: (_ for _ in ()).throw(AssertionError("health callback should not be used"))
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": "unknown"})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 0, "is_busy": False, "active_requests": []})

    snapshot = build_situation_snapshot(runtime, health={"connected": True, "session_ready": False})
    payload = snapshot.to_dict()
    json.dumps(payload)
    assert payload["session"]["connected"] is True


def test_build_snapshot_reads_interaction_state_from_state_manager(monkeypatch) -> None:
    class _State(Enum):
        IDLE = "idle"

    runtime = _StubRuntime()
    runtime.state = "unknown"
    runtime.state_manager = type("StateManager", (), {"state": _State.IDLE})()
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": "unknown"})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 0, "is_busy": False, "active_requests": []})

    snapshot = build_situation_snapshot(runtime)
    assert snapshot.interaction.state == "idle"


def test_build_snapshot_run_id_falls_back_to_current_run_id(monkeypatch) -> None:
    runtime = _StubRuntime()
    runtime._run_id = ""
    runtime._current_run_id = lambda: "run-1011"
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": "unknown"})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 0, "is_busy": False, "active_requests": []})

    snapshot = build_situation_snapshot(runtime)
    assert snapshot.run_id == "run-1011"


def test_build_snapshot_reads_session_output_voice(monkeypatch) -> None:
    runtime = _StubRuntime()
    runtime._voice = ""
    runtime._session_output_voice = "ballad"
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": "unknown"})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 0, "is_busy": False, "active_requests": []})

    snapshot = build_situation_snapshot(runtime)
    assert snapshot.model.voice == "ballad"


def test_snapshot_compact_summary_marks_response_in_progress_startup_as_busy(monkeypatch) -> None:
    runtime = _StubRuntime()
    monkeypatch.setattr(tool_runtime, "read_cached_battery_status", lambda: {"voltage": "unknown"})
    monkeypatch.setattr(tool_runtime, "read_motion_status", lambda limit=20: {"active_request_count": 0, "is_busy": False, "active_requests": []})

    summary = build_situation_snapshot(
        runtime,
        health={
            "injection_ready": False,
            "injection_ready_reason": "response_in_progress",
            "connected": True,
        },
    ).compact_summary()
    assert "startup=busy" in summary
    assert "startup=not_ready" not in summary
