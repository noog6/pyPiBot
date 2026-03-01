"""Startup injection gate and timeout helpers."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Callable

from core.logging import logger


class InjectionCoordinator:
    def __init__(
        self,
        *,
        gate_timeout_s: float,
        loop_getter: Callable[[], asyncio.AbstractEventLoop | None],
        emit_injected_event: Callable[[dict[str, Any]], None],
        emit_system_context_payload: Callable[[dict[str, Any]], None],
    ) -> None:
        self._gate_timeout_s = gate_timeout_s
        self._loop_getter = loop_getter
        self._emit_injected_event = emit_injected_event
        self._emit_system_context_payload = emit_system_context_payload
        self.released = False
        self.first_assistant_utterance_observed = False
        self.queue: deque[dict[str, Any]] = deque()
        self.timeout_task: Any = None

    def gate_active(self) -> bool:
        return not self.released and not self.first_assistant_utterance_observed and self._gate_timeout_s > 0.0

    def should_defer(self, payload: dict[str, Any], source: str) -> bool:
        _ = payload
        if not self.gate_active():
            return False
        logger.info("startup_injection_deferred source=%s reason=first_turn_unsettled", source)
        return True

    def enqueue(self, payload: dict[str, Any]) -> None:
        self.queue.append(payload)

    def release(self, reason: str) -> None:
        if self.released:
            return
        self.released = True
        queued_items = list(self.queue)
        self.queue.clear()
        logger.info("startup_injection_flush count=%s reason=%s", len(queued_items), reason)
        for item in queued_items:
            item_type = str(item.get("type") or "")
            if item_type == "event":
                self._emit_injected_event(item)
            elif item_type == "system_context":
                self._emit_system_context_payload(item)

    async def _timeout_release(self) -> None:
        if self._gate_timeout_s <= 0.0:
            return
        try:
            await asyncio.sleep(self._gate_timeout_s)
        except asyncio.CancelledError:
            return
        self.release(reason="timeout")

    def schedule_timeout(self, loop: asyncio.AbstractEventLoop | None) -> None:
        if self.released or self.first_assistant_utterance_observed:
            return
        if loop is None:
            return
        task = self.timeout_task
        if task is not None and not task.done():
            return
        self.timeout_task = asyncio.run_coroutine_threadsafe(self._timeout_release(), loop)


StartupInjectionCoordinator = InjectionCoordinator
