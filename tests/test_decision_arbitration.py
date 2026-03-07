import sys
import types

import pytest

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.decision_arbitration import (
    ArbitrationAction,
    ArbitrationCandidate,
    decide_arbitration,
)
from ai.interaction_lifecycle_policy import InteractionLifecyclePolicy, ResponseCreateDecisionAction


@pytest.mark.parametrize(
    ("case_name", "candidates", "expected_id", "expected_action", "expected_reason"),
    [
        (
            "tie_falls_back_by_candidate_id",
            [
                ArbitrationCandidate("b-refuse", ArbitrationAction.REFUSE, "refuse_b", 10),
                ArbitrationCandidate("a-refuse", ArbitrationAction.REFUSE, "refuse_a", 10),
            ],
            "a-refuse",
            ArbitrationAction.REFUSE,
            "refuse_a",
        ),
        (
            "higher_priority_beats_action_order",
            [
                ArbitrationCandidate("high-defer", ArbitrationAction.DEFER, "active_response", 100),
                ArbitrationCandidate("low-refuse", ArbitrationAction.REFUSE, "already_delivered", 10),
            ],
            "high-defer",
            ArbitrationAction.DEFER,
            "active_response",
        ),
    ],
)
def test_arbitration_winner_identity_matrix(
    case_name: str,
    candidates: list[ArbitrationCandidate],
    expected_id: str,
    expected_action: ArbitrationAction,
    expected_reason: str,
) -> None:
    _ = case_name
    decision = decide_arbitration(
        policy_name="winner_identity",
        candidates=candidates,
        default_candidate=ArbitrationCandidate(
            candidate_id="default",
            action=ArbitrationAction.DO_NOW,
            reason_code="default",
            priority=0,
        ),
    )

    assert decision.selected_candidate_id == expected_id
    assert decision.action is expected_action
    assert decision.reason_code == expected_reason


def test_arbitration_tie_breaks_stably_by_action_then_candidate_id() -> None:
    decision = decide_arbitration(
        policy_name="tie_break",
        candidates=[
            ArbitrationCandidate(
                candidate_id="b-refuse",
                action=ArbitrationAction.REFUSE,
                reason_code="refuse_b",
                priority=10,
            ),
            ArbitrationCandidate(
                candidate_id="a-refuse",
                action=ArbitrationAction.REFUSE,
                reason_code="refuse_a",
                priority=10,
            ),
            ArbitrationCandidate(
                candidate_id="z-defer",
                action=ArbitrationAction.DEFER,
                reason_code="defer_z",
                priority=10,
            ),
        ],
        default_candidate=ArbitrationCandidate(
            candidate_id="default",
            action=ArbitrationAction.DO_NOW,
            reason_code="default",
            priority=0,
        ),
    )

    assert decision.action is ArbitrationAction.REFUSE
    assert decision.reason_code == "refuse_a"
    assert decision.selected_candidate_id == "a-refuse"


def test_arbitration_repeat_identical_inputs_is_deterministic() -> None:
    candidates = [
        ArbitrationCandidate(
            candidate_id="active_response",
            action=ArbitrationAction.DEFER,
            reason_code="active_response",
            priority=100,
        ),
        ArbitrationCandidate(
            candidate_id="already_delivered",
            action=ArbitrationAction.REFUSE,
            reason_code="already_delivered",
            priority=50,
        ),
    ]
    default_candidate = ArbitrationCandidate(
        candidate_id="direct_send",
        action=ArbitrationAction.DO_NOW,
        reason_code="direct_send",
        priority=0,
    )

    first = decide_arbitration(policy_name="repeatability", candidates=candidates, default_candidate=default_candidate)
    second = decide_arbitration(policy_name="repeatability", candidates=candidates, default_candidate=default_candidate)

    assert first == second


def test_response_create_policy_keeps_single_flight_boundary_reason_code() -> None:
    policy = InteractionLifecyclePolicy()

    decision = policy.decide_response_create(
        response_in_flight=False,
        audio_playback_busy=False,
        consumes_canonical_slot=True,
        canonical_audio_started=False,
        explicit_multipart=False,
        single_flight_block_reason="already_created",
        already_delivered=False,
        preference_recall_lock_blocked=False,
        canonical_key_already_created=False,
        has_safety_override=False,
        suppression_active=False,
        normalized_origin="tool_output",
        awaiting_transcript_final=False,
    )

    assert decision.action is ResponseCreateDecisionAction.BLOCK
    assert decision.reason_code == "already_created"


def test_arbitration_module_is_policy_only_without_governance_surface() -> None:
    decision = decide_arbitration(
        policy_name="governance_boundary",
        candidates=[],
        default_candidate=ArbitrationCandidate(
            candidate_id="default",
            action=ArbitrationAction.DO_NOW,
            reason_code="direct_send",
            priority=0,
        ),
    )

    assert decision.action is ArbitrationAction.DO_NOW
    assert decision.reason_code == "direct_send"
    assert not hasattr(decision, "needs_confirmation")
    assert not hasattr(decision, "risk_tier")
