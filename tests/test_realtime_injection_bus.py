"""Tests for InjectionBus stimulus acceptance and rate-limit bookkeeping."""

from __future__ import annotations

import asyncio
from collections import deque

from ai.orchestration import OrchestrationPhase
from ai.realtime.injection_bus import InjectionBus
from ai.realtime.injections import InjectionCoordinator
from interaction import InteractionState


class _Ws:
    async def send(self, payload: str) -> None:
        return None


class _StateManager:
    def __init__(self, state: InteractionState) -> None:
        self.state = state


class _ApiStub:
    def __init__(self) -> None:
        self._pending_confirmation_token = None
        self._pending_action = None
        self.response_in_progress = False
        self.state_manager = _StateManager(InteractionState.IDLE)
        self.orchestration_state = type("S", (), {"phase": OrchestrationPhase.IDLE})()
        self._awaiting_confirmation_allowed_sources = {("battery", "critical")}
        self._injection_response_triggers = {"text_message": {"cooldown_s": 1.0, "max_per_minute": 2, "priority": 0}}
        self._injection_response_cooldown_s = 0.0
        self._max_injection_responses_per_minute = 0
        self._injection_response_trigger_timestamps = {}
        self._injection_response_timestamps = deque()
        self._sensor_event_aggregation_metrics = {"immediate": 0}
        self.rate_limits = None
        self.websocket = _Ws()

    def _expire_confirmation_awaiting_decision_timeout(self) -> None:
        return None

    def _is_awaiting_confirmation_phase(self) -> bool:
        return self.orchestration_state.phase == OrchestrationPhase.AWAITING_CONFIRMATION

    def _is_allowed_awaiting_confirmation_stimulus(self, source: str, kind: str, priority: str) -> bool:
        if priority not in {"critical", "high"}:
            return False
        return (source, kind) in self._awaiting_confirmation_allowed_sources

    async def _should_defer_sensor_response(self, trigger: str, metadata: dict[str, object]) -> bool:
        _ = (trigger, metadata)
        return False

    def _has_active_confirmation_token(self) -> bool:
        return self._pending_confirmation_token is not None

    def _is_user_confirmation_trigger(self, trigger: str, metadata: dict[str, object]) -> bool:
        _ = metadata
        return trigger == "text_message"

    def _queue_pending_image_stimulus(self, trigger: str, metadata: dict[str, object]) -> None:
        _ = (trigger, metadata)

    def _allow_ai_call(self, _key: str, *, bypass: bool = False) -> bool:
        _ = bypass
        return True

    def _consume_pending_memory_brief_note(self):
        return None



def _build_bus(api: _ApiStub) -> InjectionBus:
    coordinator = InjectionCoordinator(
        gate_timeout_s=5.0,
        loop_getter=lambda: None,
        emit_injected_event=lambda payload: None,
        emit_system_context_payload=lambda payload: None,
    )
    return InjectionBus(api, coordinator)


def test_allow_deny_matrix() -> None:
    api = _ApiStub()
    bus = _build_bus(api)

    api._pending_confirmation_token = object()
    api.orchestration_state = type("S", (), {"phase": OrchestrationPhase.AWAITING_CONFIRMATION})()
    allowed, reason = bus.can_accept_external_stimulus(
        "battery", "status", priority="critical", metadata={"severity": "critical"}
    )
    assert allowed is True
    assert reason.startswith("phase=")

    denied, denied_reason = bus.can_accept_external_stimulus(
        "camera", "image", priority="normal", metadata={"severity": "warning"}
    )
    assert denied is False
    assert denied_reason == "awaiting_confirmation_policy"

    api._pending_confirmation_token = None
    api.response_in_progress = True
    denied, denied_reason = bus.can_accept_external_stimulus("camera", "image", priority="normal")
    assert denied is False
    assert denied_reason == "response_in_progress"


def test_trigger_rate_limit_bookkeeping() -> None:
    api = _ApiStub()
    bus = _build_bus(api)
    sent: list[dict[str, object]] = []

    async def _send_response_create(_ws, event, **_kwargs):
        sent.append(event)
        return True

    api._send_response_create = _send_response_create

    asyncio.run(bus.maybe_request_response("text_message", {"source": "camera"}))
    asyncio.run(bus.maybe_request_response("text_message", {"source": "camera"}))
    asyncio.run(bus.maybe_request_response("text_message", {"source": "camera"}))

    assert len(sent) == 1
    assert len(api._injection_response_trigger_timestamps["text_message"]) == 1
