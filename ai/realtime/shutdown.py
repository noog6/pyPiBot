"""Shutdown ordering/idempotency for realtime API."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from core.logging import logger


class ShutdownCoordinator:
    def __init__(self) -> None:
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False

    def begin_shutdown(self) -> bool:
        with self._shutdown_lock:
            if self._shutdown_requested:
                logger.debug("Shutdown already in progress; ignoring duplicate shutdown signal.")
                return False
            self._shutdown_requested = True
            return True


class WebsocketCloser:
    def __init__(self, *, close_timeout_s: float) -> None:
        self._close_timeout_s = close_timeout_s
        self._ws_close_lock = asyncio.Lock()
        self._ws_close_started = False
        self._ws_close_done = False

    async def close(self, websocket: Any, *, reason: str, timeout_s: float | None = None) -> None:
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
