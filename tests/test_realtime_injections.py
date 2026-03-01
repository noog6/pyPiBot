"""Tests for startup injection coordination."""

from __future__ import annotations

import asyncio
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

import pytest

from ai.realtime.injections import InjectionCoordinator
from ai.realtime_api import RealtimeAPI


def _build_coordinator(
    *,
    gate_timeout_s: float = 5.0,
    emit_event=None,
    emit_context=None,
) -> InjectionCoordinator:
    return InjectionCoordinator(
        gate_timeout_s=gate_timeout_s,
        loop_getter=lambda: None,
        emit_injected_event=emit_event or (lambda payload: None),
        emit_system_context_payload=emit_context or (lambda payload: None),
    )


def test_should_defer_while_first_turn_unsettled(monkeypatch) -> None:
    coordinator = _build_coordinator()

    payload = {"type": "event", "message": "camera frame"}
    messages: list[str] = []

    monkeypatch.setattr(
        "ai.realtime.injections.logger.info",
        lambda message, *args: messages.append(message % args),
    )

    assert coordinator.should_defer(payload, "camera") is True
    coordinator.enqueue(payload)
    assert list(coordinator.queue) == [payload]
    assert messages == ["startup_injection_deferred source=camera reason=first_turn_unsettled"]


def test_schedule_timeout_reuses_existing_task_and_flushes_once(monkeypatch) -> None:
    flushed: list[tuple[str, str]] = []
    coordinator = _build_coordinator(
        gate_timeout_s=0.01,
        emit_event=lambda payload: flushed.append(("event", payload["id"])),
        emit_context=lambda payload: flushed.append(("system_context", payload["id"])),
    )
    coordinator.enqueue({"type": "event", "id": "e1"})
    coordinator.enqueue({"type": "system_context", "id": "c1"})
    coordinator.enqueue({"type": "event", "id": "e2"})
    messages: list[str] = []

    monkeypatch.setattr(
        "ai.realtime.injections.logger.info",
        lambda message, *args: messages.append(message % args),
    )

    loop = asyncio.new_event_loop()
    try:
        coordinator.schedule_timeout(loop)
        future = coordinator.timeout_task
        assert future is not None

        coordinator.schedule_timeout(loop)
        assert coordinator.timeout_task is future

        loop.run_until_complete(asyncio.wrap_future(future, loop=loop))
    finally:
        loop.close()

    assert flushed == [("event", "e1"), ("system_context", "c1"), ("event", "e2")]
    assert messages == ["startup_injection_flush count=3 reason=timeout"]


def test_release_is_idempotent() -> None:
    flushed: list[str] = []
    coordinator = _build_coordinator(emit_event=lambda payload: flushed.append(payload["id"]))
    coordinator.enqueue({"type": "event", "id": "e1"})

    coordinator.release("first_turn_complete")
    coordinator.release("first_turn_complete")

    assert flushed == ["e1"]


@pytest.mark.parametrize("should_defer_return", [True, False])
def test_maybe_defer_startup_injection_delegates_to_coordinator(
    monkeypatch,
    should_defer_return: bool,
) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    coordinator = _build_coordinator()
    payload = {"type": "event", "message": "camera frame"}
    calls: list[tuple[str, object, object]] = []

    api._startup_injection_gate_active = lambda: True
    api._startup_gate_is_critical_allowed = lambda source, kind, priority: False
    api._startup_injection_coordinator = lambda: coordinator

    def _fake_should_defer(self, injected_payload, source):
        calls.append(("should_defer", source, injected_payload))
        return should_defer_return

    def _fake_enqueue(self, injected_payload):
        calls.append(("enqueue", None, injected_payload))

    monkeypatch.setattr(InjectionCoordinator, "should_defer", _fake_should_defer)
    monkeypatch.setattr(InjectionCoordinator, "enqueue", _fake_enqueue)

    deferred = api._maybe_defer_startup_injection(
        source="camera",
        kind="frame",
        priority="normal",
        payload=payload,
    )

    assert deferred is should_defer_return
    assert calls[0] == ("should_defer", "camera", payload)
    enqueue_calls = [call for call in calls if call[0] == "enqueue"]
    if should_defer_return:
        assert enqueue_calls == [("enqueue", None, payload)]
    else:
        assert enqueue_calls == []
