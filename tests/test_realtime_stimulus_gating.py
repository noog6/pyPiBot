"""Tests for external stimulus gating in realtime API."""

from __future__ import annotations

from ai.event_bus import Event
from ai.orchestration import OrchestrationPhase
from ai.realtime_api import RealtimeAPI
from interaction import InteractionState


class _StateManagerStub:
    def __init__(self, state: InteractionState) -> None:
        self.state = state


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._pending_action = object()
    api.response_in_progress = False
    api.state_manager = _StateManagerStub(InteractionState.IDLE)
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.AWAITING_CONFIRMATION})()
    api._awaiting_confirmation_allowed_sources = {("battery", "critical"), ("imu", "critical")}
    return api


def test_can_accept_critical_battery_while_awaiting_confirmation() -> None:
    api = _make_api_stub()

    allowed, reason = api._can_accept_external_stimulus(
        "battery",
        "status",
        priority="critical",
        metadata={"severity": "critical"},
    )

    assert allowed is True
    assert reason.startswith("phase=")


def test_blocks_routine_battery_while_awaiting_confirmation() -> None:
    api = _make_api_stub()

    allowed, reason = api._can_accept_external_stimulus(
        "battery",
        "status",
        priority="high",
        metadata={"severity": "warning"},
    )

    assert allowed is False
    assert reason == "awaiting_confirmation_policy"


def test_inject_event_suppresses_camera_during_awaiting_confirmation() -> None:
    api = _make_api_stub()
    sent: list[str] = []
    api._format_event_for_injection = lambda event: ("camera frame", False)
    api._should_request_battery_response = lambda event, fallback: fallback
    api._is_battery_query_context_active = lambda: False
    api._log_injection_event = lambda event, request_response: None
    api._send_text_message = lambda message, **kwargs: sent.append(message)

    api.inject_event(Event(source="camera", kind="image", priority="normal"))

    assert not sent
