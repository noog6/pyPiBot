"""Models for operational orchestration and health tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class HealthStatus(str, Enum):
    """Overall health classification for the runtime."""

    OK = "ok"
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
