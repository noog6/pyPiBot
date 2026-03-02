"""Tests for realtime shutdown task registry integration."""

from __future__ import annotations

import asyncio

from ai.realtime.runtime_tasks import RuntimeTaskRegistry
from ai.realtime.shutdown import ShutdownCoordinator
from ai.realtime_api import RealtimeAPI


async def _wait_forever(cancelled_markers: set[str], name: str) -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        cancelled_markers.add(name)
        raise


def test_request_shutdown_cancels_runtime_registry_tasks_without_orphans() -> None:
    async def _run() -> None:
        api = RealtimeAPI.__new__(RealtimeAPI)
        api.loop = asyncio.get_running_loop()
        api.exit_event = asyncio.Event()
        api.websocket = None
        api._shutdown = ShutdownCoordinator(close_timeout_s=0.01)
        api._runtime_tasks = RuntimeTaskRegistry()

        cancelled: set[str] = set()
        task_a = api._runtime_task_registry().spawn("watchdog.turn_a", _wait_forever(cancelled, "task_a"))
        task_b = api._runtime_task_registry().spawn(
            "response_queue_drain.playback_complete",
            _wait_forever(cancelled, "task_b"),
        )

        await asyncio.sleep(0)
        api._request_shutdown()
        await api._runtime_task_registry().await_all(timeout_s=0.2)

        assert api.exit_event.is_set()
        assert task_a.cancelled()
        assert task_b.cancelled()
        assert cancelled == {"task_a", "task_b"}
        assert not api._runtime_task_registry()._tasks

    asyncio.run(_run())
