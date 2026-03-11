import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.governance_spine import GovernanceDecision, normalize_reason_code


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

