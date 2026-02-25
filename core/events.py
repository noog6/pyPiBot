"""Shared event contracts for realtime event publishers."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Protocol


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


class EventPublisher(Protocol):
    """Minimal publisher contract consumed by local monitors."""

    def publish(self, event: Event, *, coalesce: bool = False) -> None:
        """Publish an event to downstream consumers."""
        ...
