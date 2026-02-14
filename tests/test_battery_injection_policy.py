"""Tests for battery event request-response policy in realtime API."""

from __future__ import annotations

import time

from ai.event_bus import Event
from ai.realtime_api import RealtimeAPI


class _StimuliRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def enqueue(self, *, trigger: str, metadata: dict[str, object], priority: int) -> None:
        self.calls.append({"trigger": trigger, "metadata": metadata, "priority": priority})


class _FakeWs:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def send(self, payload: str) -> None:
        self.events.append(payload)


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._battery_response_enabled = True
    api._battery_response_allow_warning = True
    api._battery_response_allow_critical = True
    api._battery_response_require_transition = False
    api._battery_query_context_window_s = 45.0
    api._last_user_input_text = None
    api._last_user_input_time = None
    api._last_user_input_source = None
    api._last_user_battery_query_time = None
    api._injection_response_cooldown_s = 0.0
    api._max_injection_responses_per_minute = 0
    api._injection_response_triggers = {}
    api._injection_response_trigger_timestamps = {}
    api._injection_response_timestamps = []
    return api


def test_format_battery_event_defaults_to_no_response() -> None:
    api = _make_api_stub()
    message, request_response = api._format_event_for_injection(
        Event(
            source="battery",
            kind="status",
            metadata={
                "voltage": 7.5,
                "percent_of_range": 0.4,
                "severity": "warning",
                "event_type": "status",
                "transition": "steady_warning",
                "delta_percent": -0.5,
                "rapid_drop": False,
            },
        )
    )

    assert "severity=warning" in message
    assert "transition=steady_warning" in message
    assert request_response is False


def test_battery_response_only_for_qualifying_transition_or_critical() -> None:
    api = _make_api_stub()

    steady_warning = Event(
        source="battery",
        kind="status",
        metadata={"severity": "warning", "event_type": "status", "transition": "steady_warning"},
    )
    assert api._should_request_battery_response(steady_warning, fallback=False) is False

    enter_warning = Event(
        source="battery",
        kind="status",
        metadata={"severity": "warning", "event_type": "status", "transition": "enter_warning"},
    )
    assert api._should_request_battery_response(enter_warning, fallback=False) is True

    critical = Event(
        source="battery",
        kind="status",
        metadata={"severity": "critical", "event_type": "status", "transition": "steady_critical"},
    )
    assert api._should_request_battery_response(critical, fallback=False) is True


def test_battery_query_intent_detection_and_context_window() -> None:
    api = _make_api_stub()
    assert api._is_battery_status_query("how's battery?") is True
    assert api._is_battery_status_query("tell me a joke") is False

    api._record_user_input("how's battery?", source="text_message")
    assert api._is_battery_query_context_active() is True


def test_battery_query_context_allows_response() -> None:
    api = _make_api_stub()
    api._record_user_input("what is the battery voltage right now?", source="text_message")

    info_event = Event(
        source="battery",
        kind="status",
        metadata={"severity": "info", "event_type": "status", "transition": "steady_info"},
    )
    assert api._should_request_battery_response(info_event, fallback=False) is True


def test_send_text_message_battery_bypass_sets_metadata() -> None:
    api = _make_api_stub()
    async def _false(*args, **kwargs):
        return False

    api._handle_stop_word = _false
    api._maybe_handle_approval_response = _false
    api.orchestration_state = type("S", (), {"transition": lambda *args, **kwargs: None})()
    api._track_outgoing_event = lambda *args, **kwargs: None
    api._get_injection_priority = lambda trigger: 0
    api._stimuli_coordinator = _StimuliRecorder()
    api.websocket = _FakeWs()

    import asyncio

    asyncio.run(api.send_text_message_to_conversation(
        "Battery voltage: 7.7V",
        request_response=True,
        bypass_response_suppression=True,
    ))

    assert api._stimuli_coordinator.calls
    metadata = api._stimuli_coordinator.calls[0]["metadata"]
    assert metadata["bypass_limits"] is True
