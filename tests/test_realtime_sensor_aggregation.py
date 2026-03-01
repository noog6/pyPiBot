"""Tests for low-priority sensor aggregation in injected response requests."""

from __future__ import annotations

import asyncio
import json
from collections import deque

from ai.realtime_api import RealtimeAPI
from interaction import InteractionState


class _Ws:
    async def send(self, payload: str) -> None:  # pragma: no cover - not used directly
        return None


def _build_api(window_s: float = 0.05) -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.websocket = _Ws()
    api._injection_response_triggers = {}
    api._injection_response_cooldown_s = 0.0
    api._max_injection_responses_per_minute = 0
    api._injection_response_trigger_timestamps = {}
    api._injection_response_timestamps = deque()
    api.rate_limits = None
    api.response_in_progress = False
    api.state_manager = type("State", (), {"state": InteractionState.IDLE})()
    api.orchestration_state = type("S", (), {"phase": None})()
    api._allow_ai_call = lambda *_args, **_kwargs: True
    api._has_active_confirmation_token = lambda: False
    api._consume_pending_memory_brief_note = lambda: None
    api._sensor_event_aggregation_window_s = window_s
    api._sensor_event_aggregate_sources = {"battery", "imu", "camera", "ops", "health"}
    api._sensor_event_aggregation_windows = {}
    api._sensor_event_aggregation_tasks = {}
    api._sensor_event_aggregation_lock = asyncio.Lock()
    api._sensor_event_aggregation_metrics = {"dropped": 0, "coalesced": 0, "immediate": 0}
    return api


def test_critical_battery_event_bypasses_aggregation() -> None:
    api = _build_api()
    sent: list[dict[str, object]] = []

    async def _send_response_create(_ws, event, **_kwargs):
        sent.append(event)
        return True

    api._send_response_create = _send_response_create

    asyncio.run(api.maybe_request_response("text_message", {"source": "battery", "severity": "critical"}))

    assert len(sent) == 1
    stimulus = json.loads(sent[0]["response"]["metadata"]["stimulus"])
    assert stimulus["source"] == "battery"
    assert api._sensor_event_aggregation_metrics["immediate"] == 1


def test_non_critical_sensor_events_are_coalesced() -> None:
    api = _build_api(window_s=0.05)
    sent: list[dict[str, object]] = []

    async def _send_response_create(_ws, event, **_kwargs):
        sent.append(event)
        return True

    api._send_response_create = _send_response_create

    async def _exercise() -> None:
        await api.maybe_request_response("text_message", {"source": "camera", "severity": "warning"})
        await api.maybe_request_response("text_message", {"source": "camera", "severity": "warning"})
        await asyncio.sleep(0.08)

    asyncio.run(_exercise())

    assert len(sent) == 1
    stimulus = json.loads(sent[0]["response"]["metadata"]["stimulus"])
    assert stimulus["sensor_aggregate_summary"] is True
    assert stimulus["event_count"] == 2
    assert api._sensor_event_aggregation_metrics["dropped"] == 1
    assert api._sensor_event_aggregation_metrics["coalesced"] == 1
