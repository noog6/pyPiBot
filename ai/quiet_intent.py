"""Deterministic quiet-intent selector for short-horizon posture biasing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from interaction import InteractionState


class QuietIntentMode(str, Enum):
    """Supported quiet intent modes for higher-order posture selection."""

    COMPANION_PRESENCE = "companion_presence"
    OBSERVER = "observer"
    SENTINEL = "sentinel"
    RESTING_RITUAL = "resting_ritual"
    CURIOUS_WITNESS = "curious_witness"


class QuietIntentContinuityStance(str, Enum):
    """Bounded continuity stance tokens consumed by quiet-intent scoring."""

    IDLE = "idle"
    AWAITING_USER = "awaiting_user"
    ASSISTING_QUERY = "assisting_query"
    RECOVERING_CONTEXT = "recovering_context"
    OTHER = "other"


class QuietIntentOpsSeverity(str, Enum):
    """Bounded operations severity tokens consumed by quiet-intent scoring."""

    UNKNOWN = "unknown"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


_KNOWN_CONTINUITY_STANCES = {token.value: token for token in QuietIntentContinuityStance}
_KNOWN_OPS_SEVERITIES = {token.value: token for token in QuietIntentOpsSeverity}


def normalize_continuity_stance(raw_value: object) -> QuietIntentContinuityStance:
    """Normalize raw continuity stance into a bounded token."""

    token = str(raw_value or "").strip().lower()
    if not token:
        return QuietIntentContinuityStance.IDLE
    return _KNOWN_CONTINUITY_STANCES.get(token, QuietIntentContinuityStance.OTHER)


def normalize_ops_severity(raw_value: object) -> QuietIntentOpsSeverity:
    """Normalize raw ops severity into a bounded token."""

    token = str(raw_value or "").strip().lower()
    if not token:
        return QuietIntentOpsSeverity.UNKNOWN
    return _KNOWN_OPS_SEVERITIES.get(token, QuietIntentOpsSeverity.UNKNOWN)


@dataclass(frozen=True)
class QuietIntentPolicyBiases:
    """Bounded policy biases emitted by the quiet intent selector."""

    initiative_level: float
    verbosity_bias: float
    gesture_bias: float
    interruption_tolerance: float
    observation_threshold: float


@dataclass(frozen=True)
class QuietIntentInputs:
    """Small deterministic snapshot consumed by quiet intent selection."""

    interaction_state: InteractionState
    conversation_active: bool
    continuity_stance: QuietIntentContinuityStance
    continuity_stance_raw: str
    ops_severity: QuietIntentOpsSeverity
    ops_severity_raw: str
    recent_utterance_flags: tuple[str, ...]
    attention_active: bool


@dataclass(frozen=True)
class QuietIntentDecision:
    """Read-only quiet intent decision for downstream bias consultation."""

    mode: QuietIntentMode
    confidence: float
    reason_codes: tuple[str, ...]
    policy_biases: QuietIntentPolicyBiases
    bounded_inputs: QuietIntentInputs

    @property
    def confidence_band(self) -> str:
        """Bounded ordinal confidence indicator for consultative consumers."""
        if self.confidence < 0.50:
            return "low"
        if self.confidence < 0.75:
            return "medium"
        return "high"

    def to_consultative_bias_output(self) -> dict[str, object]:
        """Consultative posture output; excludes runtime state snapshot fields."""
        return {
            "mode": self.mode.value,
            "confidence_band": self.confidence_band,
            "reason_codes": list(self.reason_codes),
            "initiative_level": self.policy_biases.initiative_level,
            "verbosity_bias": self.policy_biases.verbosity_bias,
            "gesture_bias": self.policy_biases.gesture_bias,
            "interruption_tolerance": self.policy_biases.interruption_tolerance,
            "observation_threshold": self.policy_biases.observation_threshold,
        }

    def to_diagnostic_snapshot(self) -> dict[str, object]:
        """Diagnostic snapshot for logging and audits, including bounded inputs."""
        payload = self.to_consultative_bias_output()
        payload["confidence"] = round(self.confidence, 2)
        payload["interaction_state"] = self.bounded_inputs.interaction_state.value
        payload["conversation_active"] = self.bounded_inputs.conversation_active
        payload["continuity_stance"] = self.bounded_inputs.continuity_stance.value
        payload["continuity_stance_raw"] = self.bounded_inputs.continuity_stance_raw
        payload["ops_severity"] = self.bounded_inputs.ops_severity.value
        payload["ops_severity_raw"] = self.bounded_inputs.ops_severity_raw
        payload["recent_utterance_flags"] = list(self.bounded_inputs.recent_utterance_flags)
        payload["attention_active"] = self.bounded_inputs.attention_active
        return payload

    def to_log_payload(self) -> dict[str, object]:
        """Transitional logging-only alias for diagnostic payload emission.

        Prefer :meth:`to_diagnostic_snapshot` for explicit diagnostics and
        :meth:`to_consultative_bias_output` for posture-bias consumption.
        """
        return {
            **self.to_diagnostic_snapshot(),
        }


_MODE_BIASES: dict[QuietIntentMode, QuietIntentPolicyBiases] = {
    QuietIntentMode.SENTINEL: QuietIntentPolicyBiases(
        initiative_level=0.85,
        verbosity_bias=0.35,
        gesture_bias=0.10,
        interruption_tolerance=0.75,
        observation_threshold=0.20,
    ),
    QuietIntentMode.COMPANION_PRESENCE: QuietIntentPolicyBiases(
        initiative_level=0.60,
        verbosity_bias=0.65,
        gesture_bias=0.60,
        interruption_tolerance=0.55,
        observation_threshold=0.45,
    ),
    QuietIntentMode.CURIOUS_WITNESS: QuietIntentPolicyBiases(
        initiative_level=0.45,
        verbosity_bias=0.60,
        gesture_bias=0.50,
        interruption_tolerance=0.45,
        observation_threshold=0.30,
    ),
    QuietIntentMode.RESTING_RITUAL: QuietIntentPolicyBiases(
        initiative_level=0.20,
        verbosity_bias=0.25,
        gesture_bias=0.20,
        interruption_tolerance=0.25,
        observation_threshold=0.70,
    ),
    QuietIntentMode.OBSERVER: QuietIntentPolicyBiases(
        initiative_level=0.30,
        verbosity_bias=0.30,
        gesture_bias=0.30,
        interruption_tolerance=0.35,
        observation_threshold=0.60,
    ),
}

_TIE_BREAK_ORDER: tuple[QuietIntentMode, ...] = (
    QuietIntentMode.SENTINEL,
    QuietIntentMode.COMPANION_PRESENCE,
    QuietIntentMode.CURIOUS_WITNESS,
    QuietIntentMode.RESTING_RITUAL,
    QuietIntentMode.OBSERVER,
)


class QuietIntentSelector:
    """Deterministic scored selector for short-horizon quiet intent posture."""

    def select(self, inputs: QuietIntentInputs) -> QuietIntentDecision:
        scores: dict[QuietIntentMode, float] = {mode: 0.0 for mode in QuietIntentMode}
        reasons: dict[QuietIntentMode, list[str]] = {mode: [] for mode in QuietIntentMode}

        flags = {str(flag).strip().lower() for flag in inputs.recent_utterance_flags if str(flag).strip()}
        ops_severity = inputs.ops_severity.value
        stance = inputs.continuity_stance.value

        if ops_severity in {QuietIntentOpsSeverity.CRITICAL.value, QuietIntentOpsSeverity.WARNING.value}:
            scores[QuietIntentMode.SENTINEL] += (
                0.70 if ops_severity == QuietIntentOpsSeverity.CRITICAL.value else 0.50
            )
            reasons[QuietIntentMode.SENTINEL].append(f"ops_severity_{ops_severity}")
        if "anomaly_signal" in flags or "alert_context" in flags:
            scores[QuietIntentMode.SENTINEL] += 0.40
            reasons[QuietIntentMode.SENTINEL].append("anomaly_flag_present")

        if inputs.conversation_active and inputs.interaction_state in {
            InteractionState.LISTENING,
            InteractionState.THINKING,
            InteractionState.SPEAKING,
        }:
            scores[QuietIntentMode.COMPANION_PRESENCE] += 0.55
            reasons[QuietIntentMode.COMPANION_PRESENCE].append("active_conversation")
        if inputs.attention_active:
            scores[QuietIntentMode.COMPANION_PRESENCE] += 0.20
            reasons[QuietIntentMode.COMPANION_PRESENCE].append("attention_continuity_active")

        has_curiosity_context = "curiosity_signal" in flags or "exploration_request" in flags
        if has_curiosity_context:
            scores[QuietIntentMode.CURIOUS_WITNESS] += 0.70
            reasons[QuietIntentMode.CURIOUS_WITNESS].append("curiosity_context")
        if stance == QuietIntentContinuityStance.RECOVERING_CONTEXT.value:
            scores[QuietIntentMode.CURIOUS_WITNESS] += 0.20
            reasons[QuietIntentMode.CURIOUS_WITNESS].append(f"continuity_stance_{stance}")
        elif stance == QuietIntentContinuityStance.ASSISTING_QUERY.value and has_curiosity_context:
            scores[QuietIntentMode.CURIOUS_WITNESS] += 0.20
            reasons[QuietIntentMode.CURIOUS_WITNESS].append(f"continuity_stance_{stance}")

        if "calm_context" in flags or "rest_context" in flags:
            scores[QuietIntentMode.RESTING_RITUAL] += 0.65
            reasons[QuietIntentMode.RESTING_RITUAL].append("calm_or_rest_context")
        if (
            not inputs.conversation_active
            and inputs.interaction_state == InteractionState.IDLE
            and not inputs.attention_active
        ):
            scores[QuietIntentMode.RESTING_RITUAL] += 0.20
            reasons[QuietIntentMode.RESTING_RITUAL].append("idle_without_attention")

        if not inputs.conversation_active:
            scores[QuietIntentMode.OBSERVER] += 0.40
            reasons[QuietIntentMode.OBSERVER].append("conversation_inactive")
        if inputs.interaction_state in {InteractionState.IDLE, InteractionState.THINKING}:
            scores[QuietIntentMode.OBSERVER] += 0.20
            reasons[QuietIntentMode.OBSERVER].append("state_supports_observation")

        winner = self._choose_mode(scores)
        winner_score = min(1.0, max(0.0, scores[winner]))
        winner_reasons = tuple(reasons[winner] or ("default_fallback",))
        confidence = round(max(0.35, winner_score), 2)
        return QuietIntentDecision(
            mode=winner,
            confidence=confidence,
            reason_codes=winner_reasons,
            policy_biases=_MODE_BIASES[winner],
            bounded_inputs=inputs,
        )

    @staticmethod
    def _choose_mode(scores: dict[QuietIntentMode, float]) -> QuietIntentMode:
        max_score = max(scores.values()) if scores else 0.0
        top_modes = [mode for mode, score in scores.items() if score == max_score]
        for mode in _TIE_BREAK_ORDER:
            if mode in top_modes:
                return mode
        return QuietIntentMode.OBSERVER
