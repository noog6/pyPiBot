"""Thread-safe event bus for realtime injections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Deque, Iterable


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """Structured event payload for realtime injections."""

    source: str
    kind: str
    priority: str = "normal"
    content: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    dedupe_key: str | None = None
    ttl_s: float | None = None
    cooldown_s: float | None = None
    request_response: bool | None = None
    created_at: float = field(default_factory=time.time)

    def is_expired(self, now: float | None = None) -> bool:
        if self.ttl_s is None:
            return False
        if now is None:
            now = time.time()
        return now - self.created_at > self.ttl_s


class EventBus:
    """Thread-safe queue for pending realtime events."""

    def __init__(self, maxlen: int = 200) -> None:
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._queue: Deque[Event] = deque()

    def publish(self, event: Event, *, coalesce: bool = False) -> None:
        with self._cond:
            if coalesce and event.dedupe_key:
                self._remove_matching(event.dedupe_key)
            if len(self._queue) >= self._maxlen:
                dropped = self._queue.popleft()
                LOGGER.warning("Event bus full; dropping oldest event from %s.", dropped.source)
            self._queue.append(event)
            self._cond.notify()

    def publish_text(
        self,
        message: str,
        *,
        source: str = "system",
        kind: str = "message",
        priority: str = "normal",
        metadata: dict[str, object] | None = None,
        request_response: bool | None = None,
    ) -> None:
        event = Event(
            source=source,
            kind=kind,
            priority=priority,
            content=message,
            metadata=metadata or {},
            request_response=request_response,
        )
        self.publish(event)

    def get_next(self, timeout: float | None = None) -> Event | None:
        with self._cond:
            if not self._queue:
                self._cond.wait(timeout=timeout)
            if not self._queue:
                return None
            event = self._pop_highest_priority()
            return event

    def drain(self) -> Iterable[Event]:
        with self._cond:
            events = list(self._queue)
            self._queue.clear()
            return events

    def notify(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def _remove_matching(self, dedupe_key: str) -> None:
        if not self._queue:
            return
        for index, event in enumerate(self._queue):
            if event.dedupe_key == dedupe_key:
                del self._queue[index]
                return

    def _pop_highest_priority(self) -> Event:
        if len(self._queue) == 1:
            return self._queue.popleft()
        priorities = {"critical": 3, "high": 2, "normal": 1, "low": 0}
        best_index = 0
        best_score = -1
        for index, event in enumerate(self._queue):
            score = priorities.get(event.priority, 1)
            if score > best_score:
                best_score = score
                best_index = index
                if best_score == 3:
                    break
        if best_index == 0:
            return self._queue.popleft()
        event = self._queue[best_index]
        del self._queue[best_index]
        return event
