import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.governance_spine import GovernanceDecision, normalize_reason_code
from ai.opportunistic_arbitration import OpportunisticActionCandidate, arbitrate_opportunistic_actions


def test_normalize_reason_code_stable_snake_case() -> None:
    assert normalize_reason_code("Busy Turn!") == "busy_turn"


def test_governance_decision_normalizes_fields() -> None:
    decision = GovernanceDecision(
        decision="defer",
        reason_code="Needs Clarification",
        subsystem="Curiosity Engine",
        priority=50,
    )

    assert decision.reason_code == "needs_clarification"
    assert decision.subsystem == "curiosity_engine"
    assert decision.metadata == {}



def test_opportunistic_arbitration_uses_candidate_priority_not_governance_envelope_priority() -> None:
    # Conflicting envelope priorities across subsystems are seam-local metadata
    # and must not drive opportunistic cross-subsystem ordering by themselves.
    curiosity_governance = GovernanceDecision(
        decision="defer",
        reason_code="confirmation_pending",
        subsystem="curiosity",
        priority=5,
    )
    embodiment_governance = GovernanceDecision(
        decision="allow",
        reason_code="state_cue_emission",
        subsystem="embodiment",
        priority=95,
    )

    result = arbitrate_opportunistic_actions(
        user_turn_priority_active=False,
        response_obligation_priority_active=False,
        confirmation_pending=False,
        busy_turn=False,
        candidate_curiosity=OpportunisticActionCandidate(
            action_kind="curiosity_surface",
            source="curiosity_engine",
            priority=80,
            reason_code=curiosity_governance.reason_code,
            opportunistic=True,
        ),
        candidate_embodiment_flourish=OpportunisticActionCandidate(
            action_kind="embodiment_flourish",
            source="embodiment_policy",
            priority=10,
            reason_code=embodiment_governance.reason_code,
            opportunistic=True,
        ),
    )

    assert curiosity_governance.priority < embodiment_governance.priority
    assert result.selected_action_kind == "curiosity_surface"
    assert result.selected_source == "curiosity_engine"


def test_governance_decision_accepts_expiry_fields_as_optional_adapter_hints() -> None:
    decision = GovernanceDecision(
        decision="defer",
        reason_code="confirmation_pending",
        subsystem="curiosity",
        priority=90,
        ttl_s=1.0,
        expires_at=42.0,
        metadata={"issued_at_monotonic_s": 41.0},
    )

    assert decision.ttl_s == 1.0
    assert decision.expires_at == 42.0
    assert decision.metadata["issued_at_monotonic_s"] == 41.0
