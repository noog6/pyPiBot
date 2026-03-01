"""Tests for startup injection coordination."""

from __future__ import annotations

import asyncio

from ai.realtime.injections import InjectionCoordinator


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
