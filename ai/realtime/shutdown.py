"""Shutdown ordering/idempotency for realtime API."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Iterable

from core.logging import logger


class ShutdownCoordinator:
    """Coordinates one-time shutdown requests and task/websocket teardown."""

    def __init__(self, *, close_timeout_s: float = 5.0) -> None:
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._close_timeout_s = close_timeout_s
        self._ws_close_lock = asyncio.Lock()
        self._ws_close_started = False
        self._ws_close_done = False

    def request_shutdown(self) -> bool:
        """Return True once when shutdown is first requested."""
        with self._shutdown_lock:
            if self._shutdown_requested:
                logger.debug("Shutdown already in progress; ignoring duplicate shutdown signal.")
                return False
            self._shutdown_requested = True
            return True

    # Backward-compatible alias for call sites/tests that still use the old name.
    def begin_shutdown(self) -> bool:
        return self.request_shutdown()

    def cancel_tasks(
        self,
        tasks: Iterable[asyncio.Task[Any]],
        *,
        exclude_current: bool = True,
        exclude_tasks: Iterable[asyncio.Task[Any]] | None = None,
    ) -> list[asyncio.Task[Any]]:
        """Cancel unfinished tasks in a stable order."""
        current = asyncio.current_task() if exclude_current else None
        excluded = set(exclude_tasks or ())
        ordered = sorted(tasks, key=lambda task: task.get_name())
        cancelled: list[asyncio.Task[Any]] = []
        for task in ordered:
            if task.done() or task.cancelled() or task is current or task in excluded:
                continue
            task.cancel()
            cancelled.append(task)
        return cancelled

    async def close_websocket(
        self,
        websocket: Any,
        *,
        reason: str,
        timeout_s: float | None = None,
    ) -> None:
        """Attempt a single websocket close, preserving current warning/error policy."""
        if not websocket:
            return
        close_timeout_s = self._close_timeout_s if timeout_s is None else timeout_s
        async with self._ws_close_lock:
            if self._ws_close_started or self._ws_close_done:
                return
            self._ws_close_started = True
        try:
            await asyncio.wait_for(websocket.close(), timeout=close_timeout_s)
            logger.info("WebSocket closed (%s).", reason)
        except asyncio.TimeoutError:
            logger.warning("Timed out closing WebSocket (%s).", reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to close WebSocket (%s): %s", reason, exc)
        finally:
            async with self._ws_close_lock:
                self._ws_close_started = False
                self._ws_close_done = True
