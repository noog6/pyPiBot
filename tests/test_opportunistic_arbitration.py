import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.opportunistic_arbitration import (
    OpportunisticActionCandidate,
    arbitrate_opportunistic_actions,
)


def _curiosity_candidate(priority: int = 60) -> OpportunisticActionCandidate:
    return OpportunisticActionCandidate(
        action_kind="curiosity_surface",
        source="curiosity_engine",
        priority=priority,
        reason_code="curiosity_repeat_topic",
        opportunistic=True,
        ttl_s=120.0,
    )


def _embodiment_candidate(priority: int = 30) -> OpportunisticActionCandidate:
    return OpportunisticActionCandidate(
        action_kind="embodiment_flourish",
        source="embodiment_policy",
        priority=priority,
        reason_code="embodiment_optional",
        opportunistic=True,
    )


def test_explicit_user_intent_beats_curiosity_candidate() -> None:
    result = arbitrate_opportunistic_actions(
        user_turn_priority_active=True,
        response_obligation_priority_active=False,
        confirmation_pending=False,
        busy_turn=False,
        candidate_curiosity=_curiosity_candidate(),
    )

    assert result.selected_action_kind == "explicit_user_turn"
    assert result.reason_code == "explicit_intent_priority"


def test_confirmation_pending_defers_opportunistic_candidates() -> None:
    result = arbitrate_opportunistic_actions(
        user_turn_priority_active=False,
        response_obligation_priority_active=False,
        confirmation_pending=True,
        busy_turn=False,
        candidate_curiosity=_curiosity_candidate(),
    )

    assert result.selected_action_kind == "wait"
    assert result.reason_code == "confirmation_pending"
    assert result.suppressed_or_deferred[0].reason_code == "opportunistic_deferred"


def test_open_obligation_defers_opportunistic_candidates() -> None:
    result = arbitrate_opportunistic_actions(
        user_turn_priority_active=False,
        response_obligation_priority_active=True,
        confirmation_pending=False,
        busy_turn=False,
        candidate_curiosity=_curiosity_candidate(),
    )

    assert result.selected_action_kind == "wait"
    assert result.reason_code == "obligation_open"
    assert result.suppressed_or_deferred[0].result == "defer"


def test_busy_turn_blocks_low_priority_opportunistic_surface() -> None:
    result = arbitrate_opportunistic_actions(
        user_turn_priority_active=False,
        response_obligation_priority_active=False,
        confirmation_pending=False,
        busy_turn=True,
        candidate_low_priority_injection=OpportunisticActionCandidate(
            action_kind="low_priority_injection",
            source="injection_bus",
            priority=20,
            reason_code="injection_low_priority",
            opportunistic=True,
        ),
    )

    assert result.selected_action_kind == "wait"
    assert result.reason_code == "busy_turn"


def test_deterministic_winner_selection_for_same_candidate_set() -> None:
    kwargs = dict(
        user_turn_priority_active=False,
        response_obligation_priority_active=False,
        confirmation_pending=False,
        busy_turn=False,
        candidate_curiosity=_curiosity_candidate(priority=60),
        candidate_low_priority_injection=OpportunisticActionCandidate(
            action_kind="low_priority_injection",
            source="injection_bus",
            priority=60,
            reason_code="injection_low_priority",
            opportunistic=True,
        ),
    )

    result_1 = arbitrate_opportunistic_actions(**kwargs)
    result_2 = arbitrate_opportunistic_actions(**kwargs)

    assert result_1.selected_action_kind == result_2.selected_action_kind
    assert result_1.selected_source == result_2.selected_source
    assert result_1.reason_code == "arbitration_selected"


def test_selected_candidate_native_reason_is_preserved_for_metadata() -> None:
    result = arbitrate_opportunistic_actions(
        user_turn_priority_active=False,
        response_obligation_priority_active=False,
        confirmation_pending=False,
        busy_turn=False,
        candidate_curiosity=_curiosity_candidate(priority=80),
        candidate_low_priority_injection=OpportunisticActionCandidate(
            action_kind="low_priority_injection",
            source="injection_bus",
            priority=10,
            reason_code="injection_low_priority",
            opportunistic=True,
        ),
    )

    assert result.reason_code == "arbitration_selected"
    assert result.selected_native_reason_code == "curiosity_repeat_topic"


def test_embodiment_flourish_loses_to_explicit_task_response() -> None:
    result = arbitrate_opportunistic_actions(
        user_turn_priority_active=True,
        response_obligation_priority_active=False,
        confirmation_pending=False,
        busy_turn=False,
        candidate_embodiment_flourish=_embodiment_candidate(),
    )

    assert result.selected_action_kind == "explicit_user_turn"
    assert result.reason_code == "explicit_intent_priority"


def test_seam_distinguishes_no_obligation_vs_obligation_vs_user_turn_priority() -> None:
    no_obligation = arbitrate_opportunistic_actions(
        user_turn_priority_active=False,
        response_obligation_priority_active=False,
        confirmation_pending=False,
        busy_turn=False,
        candidate_curiosity=_curiosity_candidate(),
    )
    obligation_open = arbitrate_opportunistic_actions(
        user_turn_priority_active=False,
        response_obligation_priority_active=True,
        confirmation_pending=False,
        busy_turn=False,
        candidate_curiosity=_curiosity_candidate(),
    )
    explicit_signal = arbitrate_opportunistic_actions(
        user_turn_priority_active=True,
        response_obligation_priority_active=False,
        confirmation_pending=False,
        busy_turn=False,
        candidate_curiosity=_curiosity_candidate(),
    )

    assert no_obligation.selected_action_kind == "curiosity_surface"
    assert no_obligation.reason_code == "arbitration_selected"
    assert obligation_open.selected_action_kind == "wait"
    assert obligation_open.reason_code == "obligation_open"
    assert explicit_signal.selected_action_kind == "explicit_user_turn"
    assert explicit_signal.reason_code == "explicit_intent_priority"
