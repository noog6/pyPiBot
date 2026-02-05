"""Alert policy utilities for routing alerts."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Mapping

from ai.event_bus import Event, EventBus


_SEVERITY_PRIORITY = {
    "critical": "critical",
    "high": "high",
    "warning": "high",
    "info": "normal",
    "low": "low",
}


@dataclass(frozen=True)
class Alert:
    """Alert payload definition."""

    key: str
    message: str
    severity: str = "warning"
    metadata: Mapping[str, object] = field(default_factory=dict)
    ttl_s: float | None = None
    cooldown_s: float | None = None
    request_response: bool | None = None


class AlertPolicy:
    """Policy for emitting alerts with cooldown and TTL."""

    def __init__(self, *, cooldown_s: float = 60.0, ttl_s: float = 120.0) -> None:
        self._cooldown_s = float(cooldown_s)
        self._ttl_s = float(ttl_s)
        self._last_emitted: dict[str, float] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "AlertPolicy":
        alerts_cfg = config.get("alerts") if isinstance(config, Mapping) else None
        if not isinstance(alerts_cfg, Mapping):
            return cls()
        cooldown_s = float(alerts_cfg.get("cooldown_s", 60.0))
        ttl_s = float(alerts_cfg.get("ttl_s", 120.0))
        return cls(cooldown_s=cooldown_s, ttl_s=ttl_s)

    def emit(self, event_bus: EventBus, alert: Alert) -> bool:
        now = time.monotonic()
        cooldown = alert.cooldown_s if alert.cooldown_s is not None else self._cooldown_s
        last_sent = self._last_emitted.get(alert.key)
        if last_sent is not None and (now - last_sent) < cooldown:
            return False
        self._last_emitted[alert.key] = now
        ttl_s = alert.ttl_s if alert.ttl_s is not None else self._ttl_s
        severity = alert.severity.lower()
        priority = _SEVERITY_PRIORITY.get(severity, "normal")
        request_response = alert.request_response
        if request_response is None:
            request_response = severity in {"critical", "high"}
        event = Event(
            source="alert",
            kind="alert",
            priority=priority,
            content=alert.message,
            metadata={"severity": severity, **dict(alert.metadata)},
            dedupe_key=alert.key,
            ttl_s=ttl_s,
            cooldown_s=cooldown,
            request_response=request_response,
        )
        event_bus.publish(event, coalesce=True)
        return True
