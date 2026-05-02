from __future__ import annotations

from services import tool_runtime


def _reset_motion_tracking_state() -> None:
    tool_runtime._motion_state_by_request_key.clear()
    tool_runtime._motion_request_key_by_tool_call_id.clear()


def test_read_motion_status_returns_copied_summary_and_respects_limit() -> None:
    _reset_motion_tracking_state()
    tool_runtime._motion_state_by_request_key["a"] = {
        "request_key": "a",
        "status": "queued",
        "queued_monotonic_s": 2.0,
        "_private": "hidden",
    }
    tool_runtime._motion_state_by_request_key["b"] = {
        "request_key": "b",
        "status": "started",
        "queued_monotonic_s": 1.0,
    }
    before = {k: dict(v) for k, v in tool_runtime._motion_state_by_request_key.items()}

    payload = tool_runtime.read_motion_status(limit=1)

    assert payload["active_request_count"] == 2
    assert payload["is_busy"] is True
    assert len(payload["active_requests"]) == 1
    assert "_private" not in payload["active_requests"][0]
    payload["active_requests"][0]["status"] = "mutated"
    assert tool_runtime._motion_state_by_request_key["a"]["status"] == "queued"
    assert tool_runtime._motion_state_by_request_key == before
