"""Bounded consultative initiative-posture selector.

This seam emits a low-rate, reason-coded recommendation about conversational
stance. It is consultative only and must not directly arbitrate runtime
authority seams (response-create, tool execution, governance, terminal
selection, semantic ownership, or tool-followup release).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from interaction import InteractionState


class InitiativePosture(str, Enum):
    ANSWER_DIRECTLY = "answer_directly"
    CLARIFY_FIRST = "clarify_first"
    CONTINUE_FOLLOWTHROUGH = "continue_followthrough"
    AWAIT_USER = "await_user"
    OBSERVE_ONLY = "observe_only"


class InitiativeConfidenceBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class InitiativePostureInputs:
    interaction_state: InteractionState
    conversation_active: bool
    continuity_stance: str
    recent_utterance_flags: tuple[str, ...]
    followthrough_active: bool
    confirmation_pending: bool
    response_in_flight: bool
    quiet_intent_mode: str


@dataclass(frozen=True)
class InitiativePostureDecision:
    initiative_posture: InitiativePosture
    confidence: float
    reason_codes: tuple[str, ...]
    bounded_inputs: InitiativePostureInputs

    @property
    def confidence_band(self) -> InitiativeConfidenceBand:
        if self.confidence < 0.55:
            return InitiativeConfidenceBand.LOW
        if self.confidence < 0.75:
            return InitiativeConfidenceBand.MEDIUM
        return InitiativeConfidenceBand.HIGH

    def to_consultative_hint(self) -> dict[str, object]:
        return {
            "initiative_posture": self.initiative_posture.value,
            "confidence": round(self.confidence, 2),
            "confidence_band": self.confidence_band.value,
            "reason_codes": list(self.reason_codes),
        }

    def to_diagnostic_snapshot(self) -> dict[str, object]:
        payload = self.to_consultative_hint()
        payload.update(
            {
                "interaction_state": self.bounded_inputs.interaction_state.value,
                "conversation_active": self.bounded_inputs.conversation_active,
                "continuity_stance": self.bounded_inputs.continuity_stance,
                "recent_utterance_flags": list(self.bounded_inputs.recent_utterance_flags),
                "followthrough_active": self.bounded_inputs.followthrough_active,
                "confirmation_pending": self.bounded_inputs.confirmation_pending,
                "response_in_flight": self.bounded_inputs.response_in_flight,
                "quiet_intent_mode": self.bounded_inputs.quiet_intent_mode,
            }
        )
        return payload


class InitiativePostureSelector:
    """Deterministic scored selector for consultative initiative posture."""

    _TIE_BREAK_ORDER: tuple[InitiativePosture, ...] = (
        InitiativePosture.OBSERVE_ONLY,
        InitiativePosture.CONTINUE_FOLLOWTHROUGH,
        InitiativePosture.CLARIFY_FIRST,
        InitiativePosture.ANSWER_DIRECTLY,
        InitiativePosture.AWAIT_USER,
    )

    def select(self, inputs: InitiativePostureInputs) -> InitiativePostureDecision:
        scores: dict[InitiativePosture, float] = {posture: 0.0 for posture in InitiativePosture}
        reasons: dict[InitiativePosture, list[str]] = {posture: [] for posture in InitiativePosture}
        flags = {str(flag).strip().lower() for flag in inputs.recent_utterance_flags if str(flag).strip()}

        if inputs.confirmation_pending:
            scores[InitiativePosture.OBSERVE_ONLY] += 1.0
            reasons[InitiativePosture.OBSERVE_ONLY].append("confirmation_pending")
        if inputs.response_in_flight and inputs.interaction_state in {
            InteractionState.THINKING,
            InteractionState.SPEAKING,
        }:
            scores[InitiativePosture.OBSERVE_ONLY] += 0.65
            reasons[InitiativePosture.OBSERVE_ONLY].append("busy_turn")

        if inputs.followthrough_active:
            scores[InitiativePosture.CONTINUE_FOLLOWTHROUGH] += 0.9
            reasons[InitiativePosture.CONTINUE_FOLLOWTHROUGH].append("followthrough_chain_open")

        if "ambiguous_request" in flags:
            scores[InitiativePosture.CLARIFY_FIRST] += 0.85
            reasons[InitiativePosture.CLARIFY_FIRST].append("ambiguous_request")

        if "direct_question" in flags:
            scores[InitiativePosture.ANSWER_DIRECTLY] += 0.80
            reasons[InitiativePosture.ANSWER_DIRECTLY].append("direct_request_signal")

        if inputs.continuity_stance in {"awaiting_user"}:
            scores[InitiativePosture.AWAIT_USER] += 0.70
            reasons[InitiativePosture.AWAIT_USER].append("continuity_awaiting_user")

        if not inputs.conversation_active:
            scores[InitiativePosture.AWAIT_USER] += 0.20
            reasons[InitiativePosture.AWAIT_USER].append("conversation_inactive")
        if not inputs.conversation_active and "direct_question" not in flags:
            scores[InitiativePosture.OBSERVE_ONLY] += 0.25
            reasons[InitiativePosture.OBSERVE_ONLY].append("idle_without_request")

        if inputs.quiet_intent_mode in {"observer", "resting_ritual"} and "direct_question" not in flags:
            scores[InitiativePosture.OBSERVE_ONLY] += 0.15
            reasons[InitiativePosture.OBSERVE_ONLY].append("quiet_intent_passive_mode")

        winner = self._choose(scores)
        winner_score = min(0.95, max(0.0, scores[winner]))
        confidence = round(max(0.40, winner_score), 2)
        reason_codes = tuple(reasons[winner] or ("default_fallback",))
        return InitiativePostureDecision(
            initiative_posture=winner,
            confidence=confidence,
            reason_codes=reason_codes,
            bounded_inputs=inputs,
        )

    def _choose(self, scores: dict[InitiativePosture, float]) -> InitiativePosture:
        max_score = max(scores.values()) if scores else 0.0
        winners = {posture for posture, score in scores.items() if score == max_score}
        for posture in self._TIE_BREAK_ORDER:
            if posture in winners:
                return posture
        return InitiativePosture.AWAIT_USER
