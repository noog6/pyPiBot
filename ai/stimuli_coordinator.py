"""Coordinator for injected stimuli with debounce and prioritization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any, Awaitable, Callable

from core.logging import logger


@dataclass
class StimulusEvent:
    trigger: str
    timestamp: float
    priority: int
    metadata: dict[str, Any]
    count: int = 1


class StimuliCoordinator:
    """Coalesce injected stimuli and emit debounced responses."""

    def __init__(
        self,
        *,
        debounce_window_s: float,
        cooldown_s: float,
        emit_callback: Callable[[str, dict[str, Any]], Awaitable[None]],
    ) -> None:
        self._debounce_window_s = max(0.0, debounce_window_s)
        self._cooldown_s = max(0.0, cooldown_s)
        self._emit_callback = emit_callback
        self._lock = asyncio.Lock()
        self._queue: dict[str, StimulusEvent] = {}
        self._queue_order: list[str] = []
        self._pending_task: asyncio.Task[None] | None = None
        self._last_emit_time = 0.0

    async def enqueue(
        self,
        *,
        trigger: str,
        metadata: dict[str, Any],
        priority: int = 0,
    ) -> None:
        now = time.monotonic()
        async with self._lock:
            if self._cooldown_s > 0.0 and priority <= 0:
                elapsed = now - self._last_emit_time
                if elapsed < self._cooldown_s:
                    logger.info(
                        "Dropping low-priority stimulus %s: cooldown %.2fs remaining.",
                        trigger,
                        self._cooldown_s - elapsed,
                    )
                    return

            if trigger in self._queue:
                event = self._queue[trigger]
                event.timestamp = now
                event.metadata = metadata
                event.count += 1
                event.priority = max(event.priority, priority)
            else:
                self._queue[trigger] = StimulusEvent(
                    trigger=trigger,
                    timestamp=now,
                    priority=priority,
                    metadata=metadata,
                )
                self._queue_order.append(trigger)

            if self._pending_task is None or self._pending_task.done():
                self._pending_task = asyncio.create_task(self._debounce_and_emit())

    async def _debounce_and_emit(self) -> None:
        try:
            if self._debounce_window_s > 0.0:
                await asyncio.sleep(self._debounce_window_s)

            async with self._lock:
                if not self._queue:
                    return
                events = [self._queue[key] for key in self._queue_order if key in self._queue]
                self._queue.clear()
                self._queue_order.clear()

            chosen_event = max(
                events,
                key=lambda event: (event.priority, event.timestamp),
            )
            summary_payload = {
                "event_count": sum(event.count for event in events),
                "triggers": [event.trigger for event in events],
                "counts": {event.trigger: event.count for event in events},
                "latest_metadata": {event.trigger: event.metadata for event in events},
                "debounce_window_s": self._debounce_window_s,
            }
            await self._emit_callback(chosen_event.trigger, summary_payload)
            self._last_emit_time = time.monotonic()
        finally:
            async with self._lock:
                if self._queue:
                    self._pending_task = asyncio.create_task(self._debounce_and_emit())
                else:
                    self._pending_task = None
