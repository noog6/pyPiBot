import importlib.util
from pathlib import Path
import sys

from interaction import InteractionState

_QUIET_INTENT_PATH = Path(__file__).resolve().parents[1] / "ai" / "quiet_intent.py"
_QUIET_INTENT_SPEC = importlib.util.spec_from_file_location("quiet_intent_module", _QUIET_INTENT_PATH)
assert _QUIET_INTENT_SPEC is not None and _QUIET_INTENT_SPEC.loader is not None
_QUIET_INTENT_MODULE = importlib.util.module_from_spec(_QUIET_INTENT_SPEC)
sys.modules[_QUIET_INTENT_SPEC.name] = _QUIET_INTENT_MODULE
_QUIET_INTENT_SPEC.loader.exec_module(_QUIET_INTENT_MODULE)

QuietIntentInputs = _QUIET_INTENT_MODULE.QuietIntentInputs
QuietIntentPolicyBiases = _QUIET_INTENT_MODULE.QuietIntentPolicyBiases
QuietIntentDecision = _QUIET_INTENT_MODULE.QuietIntentDecision
QuietIntentContinuityStance = _QUIET_INTENT_MODULE.QuietIntentContinuityStance
QuietIntentMode = _QUIET_INTENT_MODULE.QuietIntentMode
QuietIntentOpsSeverity = _QUIET_INTENT_MODULE.QuietIntentOpsSeverity
QuietIntentSelector = _QUIET_INTENT_MODULE.QuietIntentSelector
normalize_continuity_stance = _QUIET_INTENT_MODULE.normalize_continuity_stance
normalize_ops_severity = _QUIET_INTENT_MODULE.normalize_ops_severity


def test_quiet_intent_selects_sentinel_on_critical_alert_context() -> None:
    selector = QuietIntentSelector()
    decision = selector.select(
        QuietIntentInputs(
            interaction_state=InteractionState.IDLE,
            conversation_active=False,
            continuity_stance=QuietIntentContinuityStance.IDLE,
            continuity_stance_raw="idle",
            ops_severity=QuietIntentOpsSeverity.CRITICAL,
            ops_severity_raw="critical",
            recent_utterance_flags=("alert_context",),
            attention_active=False,
        )
    )
    assert decision.mode == QuietIntentMode.SENTINEL
    assert "ops_severity_critical" in decision.reason_codes
    assert "anomaly_flag_present" in decision.reason_codes
    assert decision.policy_biases.observation_threshold == 0.20


def test_quiet_intent_selects_resting_ritual_for_calm_idle_context() -> None:
    selector = QuietIntentSelector()
    decision = selector.select(
        QuietIntentInputs(
            interaction_state=InteractionState.IDLE,
            conversation_active=False,
            continuity_stance=QuietIntentContinuityStance.IDLE,
            continuity_stance_raw="idle",
            ops_severity=QuietIntentOpsSeverity.UNKNOWN,
            ops_severity_raw="unknown",
            recent_utterance_flags=("calm_context",),
            attention_active=False,
        )
    )
    assert decision.mode == QuietIntentMode.RESTING_RITUAL
    assert "calm_or_rest_context" in decision.reason_codes
    assert "idle_without_attention" in decision.reason_codes
    assert decision.confidence >= 0.35


def test_quiet_intent_selects_curious_witness_for_query_stance_and_flag() -> None:
    selector = QuietIntentSelector()
    decision = selector.select(
        QuietIntentInputs(
            interaction_state=InteractionState.THINKING,
            conversation_active=True,
            continuity_stance=QuietIntentContinuityStance.ASSISTING_QUERY,
            continuity_stance_raw="assisting_query",
            ops_severity=QuietIntentOpsSeverity.UNKNOWN,
            ops_severity_raw="unknown",
            recent_utterance_flags=("curiosity_signal",),
            attention_active=True,
        )
    )
    assert decision.mode == QuietIntentMode.CURIOUS_WITNESS
    assert "curiosity_context" in decision.reason_codes
    assert "continuity_stance_assisting_query" in decision.reason_codes


def test_quiet_intent_does_not_promote_curious_witness_for_assisting_stance_alone() -> None:
    selector = QuietIntentSelector()
    decision = selector.select(
        QuietIntentInputs(
            interaction_state=InteractionState.IDLE,
            conversation_active=True,
            continuity_stance=QuietIntentContinuityStance.ASSISTING_QUERY,
            continuity_stance_raw="assisting_query",
            ops_severity=QuietIntentOpsSeverity.UNKNOWN,
            ops_severity_raw="unknown",
            recent_utterance_flags=(),
            attention_active=False,
        )
    )
    assert decision.mode == QuietIntentMode.OBSERVER
    assert "continuity_stance_assisting_query" not in decision.reason_codes


def test_quiet_intent_log_payload_contains_bounded_inputs_and_biases() -> None:
    selector = QuietIntentSelector()
    decision = selector.select(
        QuietIntentInputs(
            interaction_state=InteractionState.LISTENING,
            conversation_active=True,
            continuity_stance=QuietIntentContinuityStance.AWAITING_USER,
            continuity_stance_raw="awaiting_user",
            ops_severity=QuietIntentOpsSeverity.UNKNOWN,
            ops_severity_raw="unknown",
            recent_utterance_flags=(),
            attention_active=True,
        )
    )
    payload = decision.to_log_payload()
    assert payload["mode"] == QuietIntentMode.COMPANION_PRESENCE.value
    assert payload["confidence_band"] in {"low", "medium", "high"}
    assert payload["interaction_state"] == InteractionState.LISTENING.value
    assert payload["continuity_stance_raw"] == "awaiting_user"
    assert payload["ops_severity_raw"] == "unknown"
    assert payload["conversation_active"] is True
    assert payload["attention_active"] is True
    assert isinstance(payload["reason_codes"], list)
    assert "initiative_level" in payload


def test_quiet_intent_consultative_bias_output_excludes_runtime_snapshot_fields() -> None:
    selector = QuietIntentSelector()
    decision = selector.select(
        QuietIntentInputs(
            interaction_state=InteractionState.IDLE,
            conversation_active=False,
            continuity_stance=QuietIntentContinuityStance.IDLE,
            continuity_stance_raw="idle",
            ops_severity=QuietIntentOpsSeverity.WARNING,
            ops_severity_raw="warning",
            recent_utterance_flags=("alert_context",),
            attention_active=False,
        )
    )
    payload = decision.to_consultative_bias_output()
    assert payload["mode"] == QuietIntentMode.SENTINEL.value
    assert payload["confidence_band"] == "high"
    assert "interaction_state" not in payload
    assert "conversation_active" not in payload
    assert "ops_severity" not in payload


def test_quiet_intent_confidence_band_is_ordinal_and_bounded() -> None:
    inputs = QuietIntentInputs(
        interaction_state=InteractionState.IDLE,
        conversation_active=False,
        continuity_stance=QuietIntentContinuityStance.IDLE,
        continuity_stance_raw="idle",
        ops_severity=QuietIntentOpsSeverity.UNKNOWN,
        ops_severity_raw="unknown",
        recent_utterance_flags=(),
        attention_active=False,
    )
    biases = QuietIntentPolicyBiases(0.3, 0.3, 0.3, 0.3, 0.3)
    low = QuietIntentDecision(QuietIntentMode.OBSERVER, 0.35, ("default_fallback",), biases, inputs)
    medium = QuietIntentDecision(QuietIntentMode.OBSERVER, 0.70, ("default_fallback",), biases, inputs)
    high = QuietIntentDecision(QuietIntentMode.OBSERVER, 0.90, ("default_fallback",), biases, inputs)
    assert low.confidence_band == "low"
    assert medium.confidence_band == "medium"
    assert high.confidence_band == "high"


def test_quiet_intent_log_payload_alias_matches_diagnostic_snapshot() -> None:
    selector = QuietIntentSelector()
    decision = selector.select(
        QuietIntentInputs(
            interaction_state=InteractionState.LISTENING,
            conversation_active=True,
            continuity_stance=QuietIntentContinuityStance.AWAITING_USER,
            continuity_stance_raw="awaiting_user",
            ops_severity=QuietIntentOpsSeverity.UNKNOWN,
            ops_severity_raw="unknown",
            recent_utterance_flags=("curiosity_signal",),
            attention_active=True,
        )
    )
    assert decision.to_log_payload() == decision.to_diagnostic_snapshot()


def test_quiet_intent_normalizers_bound_unknown_tokens() -> None:
    assert normalize_continuity_stance(" custom_mode ") == QuietIntentContinuityStance.OTHER
    assert normalize_continuity_stance("") == QuietIntentContinuityStance.IDLE
    assert normalize_ops_severity(" noisy ") == QuietIntentOpsSeverity.UNKNOWN
    assert normalize_ops_severity(" warning ") == QuietIntentOpsSeverity.WARNING
