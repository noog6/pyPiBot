"""Models for operational orchestration and health tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class HealthStatus(str, Enum):
    """Overall health classification for the runtime."""

    OK = "ok"
    WARMUP = "warmup"
    DEGRADED = "degraded"
    FAILING = "failing"


@dataclass(frozen=True)
class HealthSnapshot:
    """Snapshot of orchestrator health state."""

    timestamp: float
    status: HealthStatus
    summary: str
    details: Mapping[str, str | float | int] = field(default_factory=dict)


@dataclass
class BudgetCounters:
    """Mutable counters for orchestration budgets."""

    ticks: int = 0
    heartbeats: int = 0
    errors: int = 0


class ModeState(str, Enum):
    """Operational mode for the orchestrator."""

    STARTUP = "startup"
    IDLE = "idle"
    ACTIVE = "active"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class OpsEvent:
    """Event emitted by the orchestrator."""

    timestamp: float
    event_type: str
    message: str
    metadata: Mapping[str, str | float | int] = field(default_factory=dict)


@dataclass
class DebouncedState:
    """Hold a debounced state for health probes."""

    status: HealthStatus
    since: float
    last_update: float


@dataclass(frozen=True)
class OpsSnapshot:
    """Canonical snapshot payload for parser-safe ops telemetry."""

    schema_version: str
    emitted_at: float
    reason: str
    mode: str
    loop_phase: str
    active_probe: str
    ticks: int
    heartbeats: int
    errors: int
    health_status: str
    health_summary: str
    loop_period_s: float
    heartbeat_period_s: float

    def to_metadata(self) -> Mapping[str, str | float | int]:
        """Serialize the snapshot into event metadata-friendly primitives."""

        return {
            "schema_version": self.schema_version,
            "emitted_at": self.emitted_at,
            "reason": self.reason,
            "mode": self.mode,
            "loop_phase": self.loop_phase,
            "active_probe": self.active_probe,
            "ticks": self.ticks,
            "heartbeats": self.heartbeats,
            "errors": self.errors,
            "health_status": self.health_status,
            "health_summary": self.health_summary,
            "loop_period_s": self.loop_period_s,
            "heartbeat_period_s": self.heartbeat_period_s,
        }
