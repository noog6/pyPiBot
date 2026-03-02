"""Runtime task registry helpers for realtime background work."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from core.logging import logger


class RuntimeTaskRegistry:
    """Tracks runtime-managed tasks to support coordinated shutdown."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    def spawn(self, name: str, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task_name = f"realtime.{str(name or 'task').strip() or 'task'}"
        task = asyncio.create_task(coro, name=task_name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def cancel_all(self, reason: str) -> None:
        cancelled = 0
        for task in tuple(self._tasks):
            if task.done() or task.cancelled():
                continue
            task.cancel()
            cancelled += 1
        logger.info("runtime_tasks_cancel_all reason=%s cancelled=%s", reason, cancelled)

    async def await_all(self, timeout_s: float) -> None:
        pending = [task for task in tuple(self._tasks) if not task.done() and not task.cancelled()]
        if not pending:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout_s)
        except asyncio.TimeoutError:
            remaining = sum(1 for task in pending if not task.done())
            logger.warning(
                "runtime_tasks_await_timeout timeout_s=%.2f remaining=%s",
                max(0.0, float(timeout_s)),
                remaining,
            )
