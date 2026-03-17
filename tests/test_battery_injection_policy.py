"""Tests for battery event request-response policy in realtime API."""

from __future__ import annotations

import sys
import time
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.event_bus import Event
from ai.realtime_api import RealtimeAPI
from core.logging import logger


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
    api._pending_research_request = None
    api._suppressed_topics = set()
    api._battery_redline_percent = 10.0
    api._synthetic_input_event_counter = 0
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


def test_battery_topic_suppression_blocks_non_redline_alerts() -> None:
    api = _make_api_stub()
    api._record_user_input("stop talking about battery", source="text_message")

    suppressed_warning = Event(
        source="battery",
        kind="status",
        metadata={"severity": "warning", "event_type": "status", "percent_of_range": 0.30},
    )
    assert api._should_request_battery_response(suppressed_warning, fallback=False) is False

    redline_critical = Event(
        source="battery",
        kind="status",
        metadata={"severity": "critical", "event_type": "status", "percent_of_range": 0.05},
    )
    assert api._should_request_battery_response(redline_critical, fallback=False) is True


def test_send_text_message_battery_bypass_sets_metadata() -> None:
    api = _make_api_stub()

    async def _false(*args, **kwargs):
        return False

    api._handle_stop_word = _false
    api._maybe_handle_approval_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api.orchestration_state = type("S", (), {"transition": lambda *args, **kwargs: None})()
    api._track_outgoing_event = lambda *args, **kwargs: None
    api._get_injection_priority = lambda trigger: 0
    api._stimuli_coordinator = _StimuliRecorder()
    api.websocket = _FakeWs()
    api._get_or_create_transport = lambda: type(
        "T",
        (),
        {"send_json": staticmethod(lambda *_args, **_kwargs: __import__("asyncio").sleep(0))},
    )()

    import asyncio

    asyncio.run(api.send_text_message_to_conversation(
        "Battery voltage: 7.7V",
        request_response=True,
        bypass_response_suppression=True,
    ))

    assert api._stimuli_coordinator.calls
    metadata = api._stimuli_coordinator.calls[0]["metadata"]
    assert metadata["bypass_limits"] is True


def test_battery_response_decision_returns_governance_envelope() -> None:
    api = _make_api_stub()

    enter_warning = Event(
        source="battery",
        kind="status",
        metadata={"severity": "warning", "event_type": "status", "transition": "enter_warning"},
    )

    decision = api._battery_response_decision(enter_warning, fallback=False)

    assert decision.decision == "allow"
    assert decision.reason_code == "warning_transition_allowed"
    assert decision.subsystem == "battery"
    assert decision.metadata["transition"] == "enter_warning"


def test_battery_response_decision_reason_code_is_normalized() -> None:
    api = _make_api_stub()

    api._suppressed_topics.add("battery")
    decision = api._battery_response_decision(
        Event(source="battery", kind="status", metadata={"severity": "warning"}),
        fallback=False,
    )

    assert decision.reason_code == "topic_suppression"


def test_battery_governance_log_includes_standardized_envelope(monkeypatch) -> None:
    api = _make_api_stub()
    info_messages: list[str] = []
    monkeypatch.setattr(logger, "info", lambda msg, *args: info_messages.append(msg % args if args else msg))
    api._current_run_id = lambda: "run-395"
    api._current_turn_id_or_unknown = lambda: "turn_22"

    enter_warning = Event(
        source="battery",
        kind="status",
        metadata={"severity": "warning", "event_type": "status", "transition": "enter_warning"},
    )

    assert api._should_request_battery_response(enter_warning, fallback=False) is True
    assert info_messages
    rendered = info_messages[-1]
    assert "battery_governance" in rendered
    assert "run_id=run-395" in rendered
    assert "turn_id=turn_22" in rendered
    assert "subsystem=battery" in rendered
    assert "decision=allow" in rendered
    assert "reason_code=warning_transition_allowed" in rendered
    assert "priority=" in rendered
    assert "event_type=status" in rendered
    assert "severity=warning" in rendered
    assert "transition=enter_warning" in rendered


def test_context_only_injection_uses_system_role_and_skips_user_intent_path() -> None:
    api = _make_api_stub()

    async def _false(*args, **kwargs):
        return False

    class _Transport:
        def __init__(self) -> None:
            self.events = []

        async def send_json(self, _ws, payload):
            self.events.append(payload)

    transport = _Transport()
    api._handle_stop_word = _false
    api._maybe_handle_approval_response = _false
    api._maybe_handle_research_permission_response = _false
    api._maybe_handle_research_budget_response = _false
    api._maybe_apply_late_confirmation_decision = _false
    api._maybe_handle_preference_recall_intent = _false
    api._maybe_process_research_intent = _false
    api.orchestration_state = type("S", (), {"transition": lambda *args, **kwargs: None})()
    api._track_outgoing_event = lambda *args, **kwargs: None
    api._get_injection_priority = lambda trigger: 0
    api._stimuli_coordinator = _StimuliRecorder()
    api.websocket = _FakeWs()
    api._get_or_create_transport = lambda: transport

    called = {"record_user_input": 0}

    def _record_user_input(*args, **kwargs):
        called["record_user_input"] += 1

    api._record_user_input = _record_user_input

    import asyncio

    asyncio.run(api.send_text_message_to_conversation(
        "Battery voltage: 7.6V",
        request_response=False,
        injection_metadata={"source": "battery", "kind": "status", "severity": "warning"},
    ))

    assert called["record_user_input"] == 0
    assert transport.events
    assert transport.events[0]["item"]["role"] == "system"

