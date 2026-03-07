"""Tests for realtime shutdown coordination helpers."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path


def _load_shutdown_module():
    module_path = Path(__file__).resolve().parents[1] / "ai" / "realtime" / "shutdown.py"
    spec = importlib.util.spec_from_file_location("test_shutdown_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _WebSocketStub:
    def __init__(self) -> None:
        self.close_attempts = 0

    async def close(self) -> None:
        self.close_attempts += 1


async def _idle() -> None:
    await asyncio.Event().wait()


def test_request_shutdown_is_idempotent() -> None:
    shutdown_module = _load_shutdown_module()
    coordinator = shutdown_module.ShutdownCoordinator()

    assert coordinator.request_shutdown() is True
    assert coordinator.request_shutdown() is False


def test_close_websocket_attempted_once() -> None:
    shutdown_module = _load_shutdown_module()
    info_logs: list[str] = []

    async def _run() -> None:
        ws = _WebSocketStub()
        coordinator = shutdown_module.ShutdownCoordinator(close_timeout_s=0.1)

        await coordinator.close_websocket(ws, reason="signal")
        await coordinator.close_websocket(ws, reason="signal")

        assert ws.close_attempts == 1

    original = shutdown_module.logger.info
    shutdown_module.logger.info = lambda message, *args: info_logs.append(message % args)
    try:
        asyncio.run(_run())
    finally:
        shutdown_module.logger.info = original

    assert "WebSocket closed (signal)." in info_logs


def test_cancel_tasks_uses_stable_order_and_excludes_finished_tasks() -> None:
    shutdown_module = _load_shutdown_module()

    async def _run() -> None:
        coordinator = shutdown_module.ShutdownCoordinator()

        active_b = asyncio.create_task(_idle(), name="task-b")
        done = asyncio.create_task(asyncio.sleep(0), name="task-done")
        active_a = asyncio.create_task(_idle(), name="task-a")
        await done

        cancelled = coordinator.cancel_tasks([active_b, done, active_a], exclude_current=True)
        cancelled_names = [task.get_name() for task in cancelled]

        assert cancelled_names == ["task-a", "task-b"]
        assert done.cancelled() is False

        for task in (active_a, active_b):
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_websocket_close_state_transitions() -> None:
    shutdown_module = _load_shutdown_module()

    async def _run() -> None:
        ws = _WebSocketStub()
        coordinator = shutdown_module.ShutdownCoordinator(close_timeout_s=0.1)

        assert await coordinator.websocket_close_state() == "open"
        await coordinator.close_websocket(ws, reason="signal")
        assert await coordinator.websocket_close_state() == "closed"

    asyncio.run(_run())


def test_is_shutdown_requested_reflects_request_state() -> None:
    shutdown_module = _load_shutdown_module()
    coordinator = shutdown_module.ShutdownCoordinator()

    assert coordinator.is_shutdown_requested() is False
    coordinator.request_shutdown()
    assert coordinator.is_shutdown_requested() is True
