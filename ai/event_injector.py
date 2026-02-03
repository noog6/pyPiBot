"""Background injector thread for realtime events."""

from __future__ import annotations

import threading
import time
from typing import Callable

from ai.event_bus import Event, EventBus
from core.logging import logger as LOGGER


class EventInjector:
    """Drains the event bus and injects events into realtime."""

    def __init__(
        self,
        event_bus: EventBus,
        *,
        ready_event: threading.Event,
        is_ready: Callable[[], bool],
        inject_callback: Callable[[Event], None],
    ) -> None:
        self._event_bus = event_bus
        self._ready_event = ready_event
        self._is_ready = is_ready
        self._inject_callback = inject_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sent_by_key: dict[str, float] = {}

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._event_bus.notify()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            if not self._is_ready():
                self._ready_event.wait(timeout=0.5)
                continue
            event = self._event_bus.get_next(timeout=0.5)
            if event is None:
                continue
            if event.is_expired():
                LOGGER.debug("Dropping expired event from %s.", event.source)
                continue
            if self._is_on_cooldown(event):
                LOGGER.debug("Dropping cooldown event %s from %s.", event.dedupe_key, event.source)
                continue
            try:
                self._inject_callback(event)
            except Exception as exc:
                LOGGER.warning("Failed to inject event from %s: %s", event.source, exc)

    def _is_on_cooldown(self, event: Event) -> bool:
        if event.priority == "critical":
            return False
        if not event.dedupe_key or not event.cooldown_s:
            return False
        now = time.time()
        last_sent = self._last_sent_by_key.get(event.dedupe_key)
        if last_sent is not None and now - last_sent < event.cooldown_s:
            return True
        self._last_sent_by_key[event.dedupe_key] = now
        return False
