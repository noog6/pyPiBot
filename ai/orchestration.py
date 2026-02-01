"""Orchestration phase tracking for realtime behavior."""

from __future__ import annotations

from enum import Enum

from core.logging import log_phase_transition


class OrchestrationPhase(Enum):
    SENSE = "sense"
    PLAN = "plan"
    ACT = "act"
    REFLECT = "reflect"
    IDLE = "idle"


class OrchestrationState:
    def __init__(self, initial: OrchestrationPhase = OrchestrationPhase.IDLE) -> None:
        self._phase = initial

    @property
    def phase(self) -> OrchestrationPhase:
        return self._phase

    def transition(self, phase: OrchestrationPhase, reason: str | None = None) -> None:
        if phase == self._phase:
            return
        previous = self._phase
        self._phase = phase
        log_phase_transition(previous, phase, reason=reason)
