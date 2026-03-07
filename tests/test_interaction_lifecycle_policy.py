import sys
import types

import pytest

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.interaction_lifecycle_policy import (
    InteractionLifecyclePolicy,
    ResponseCreateDecisionAction,
    ServerAutoCreatedDecisionAction,
)


@pytest.mark.parametrize(
    (
        "case_name",
        "kwargs",
        "expected_action",
        "expected_reason",
    ),
    [
        (
            "architecture_law_active_response_beats_send",
            dict(
                response_in_flight=True,
                audio_playback_busy=False,
                consumes_canonical_slot=True,
                canonical_audio_started=False,
                explicit_multipart=False,
                single_flight_block_reason="",
                already_delivered=False,
                preference_recall_lock_blocked=False,
                canonical_key_already_created=False,
                has_safety_override=False,
                suppression_active=False,
                normalized_origin="assistant_message",
                awaiting_transcript_final=False,
            ),
            ResponseCreateDecisionAction.SCHEDULE,
            "active_response",
        ),
        (
            "architecture_law_suppression_beats_send",
            dict(
                response_in_flight=False,
                audio_playback_busy=False,
                consumes_canonical_slot=True,
                canonical_audio_started=False,
                explicit_multipart=False,
                single_flight_block_reason="",
                already_delivered=False,
                preference_recall_lock_blocked=False,
                canonical_key_already_created=False,
                has_safety_override=False,
                suppression_active=True,
                normalized_origin="server_auto",
                awaiting_transcript_final=False,
            ),
            ResponseCreateDecisionAction.BLOCK,
            "preference_recall_suppressed",
        ),
        (
            "architecture_law_awaiting_transcript_final_beats_send",
            dict(
                response_in_flight=False,
                audio_playback_busy=False,
                consumes_canonical_slot=True,
                canonical_audio_started=False,
                explicit_multipart=False,
                single_flight_block_reason="",
                already_delivered=False,
                preference_recall_lock_blocked=False,
                canonical_key_already_created=False,
                has_safety_override=False,
                suppression_active=False,
                normalized_origin="server_auto",
                awaiting_transcript_final=True,
            ),
            ResponseCreateDecisionAction.SCHEDULE,
            "awaiting_transcript_final",
        ),
        (
            "architecture_law_active_response_defers_before_single_flight_block",
            dict(
                response_in_flight=True,
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
                normalized_origin="assistant_message",
                awaiting_transcript_final=False,
            ),
            ResponseCreateDecisionAction.SCHEDULE,
            "active_response",
        ),
    ],
)
def test_decide_response_create_precedence_matrix(
    case_name: str,
    kwargs: dict[str, object],
    expected_action: ResponseCreateDecisionAction,
    expected_reason: str,
) -> None:
    _ = case_name
    policy = InteractionLifecyclePolicy()

    decision = policy.decide_response_create(**kwargs)

    assert decision.action is expected_action
    assert decision.reason_code == expected_reason


def test_decide_response_create_prefers_schedule_for_active_response() -> None:
    policy = InteractionLifecyclePolicy()

    decision = policy.decide_response_create(
        response_in_flight=True,
        audio_playback_busy=False,
        consumes_canonical_slot=True,
        canonical_audio_started=False,
        explicit_multipart=False,
        single_flight_block_reason="",
        already_delivered=False,
        preference_recall_lock_blocked=False,
        canonical_key_already_created=False,
        has_safety_override=False,
        suppression_active=False,
        normalized_origin="assistant_message",
        awaiting_transcript_final=False,
    )

    assert decision.action is ResponseCreateDecisionAction.SCHEDULE
    assert decision.reason_code == "active_response"
    assert decision.queue_reason == "active_response"


def test_decide_server_auto_created_obligation_replacement_wins() -> None:
    policy = InteractionLifecyclePolicy()

    decision = policy.decide_server_auto_created(
        normalized_origin="server_auto",
        has_turn_id=True,
        has_canonical_key=True,
        suppression_by_turn=True,
        suppression_window_active=False,
        suppression_by_input_event=False,
        obligation_replacement=True,
    )

    assert decision.action is ServerAutoCreatedDecisionAction.CANCEL_PRE_AUDIO
    assert decision.reason_code == "response_obligation_replacement"


def test_decide_watchdog_timeout_default_timeout_schedules_micro_ack() -> None:
    policy = InteractionLifecyclePolicy()

    decision = policy.decide_watchdog_timeout(
        suppressed_by_turn=False,
        suppressed_by_input_event=False,
        suppression_window_active=False,
        response_in_flight=False,
        active_response_origin="unknown",
        active_response_id="unknown",
        delivery_state_terminal=False,
        audio_playback_busy=False,
        has_pending_response_create=False,
        pending_origin="unknown",
        pending_reason="unknown",
        listening_state_gate=False,
    )

    assert decision.reason_code == "timeout"
    assert decision.details == "response.created missing before timeout"
    assert decision.should_schedule_micro_ack is True


def test_decide_response_create_schedules_server_auto_while_awaiting_transcript_final() -> None:
    policy = InteractionLifecyclePolicy()

    decision = policy.decide_response_create(
        response_in_flight=False,
        audio_playback_busy=False,
        consumes_canonical_slot=True,
        canonical_audio_started=False,
        explicit_multipart=False,
        single_flight_block_reason="",
        already_delivered=False,
        preference_recall_lock_blocked=False,
        canonical_key_already_created=False,
        has_safety_override=False,
        suppression_active=False,
        normalized_origin="server_auto",
        awaiting_transcript_final=True,
    )

    assert decision.action is ResponseCreateDecisionAction.SCHEDULE
    assert decision.reason_code == "awaiting_transcript_final"
    assert decision.queue_reason == "awaiting_transcript_final"


def test_decide_response_create_keeps_non_server_auto_direct_during_transcript_wait() -> None:
    policy = InteractionLifecyclePolicy()

    decision = policy.decide_response_create(
        response_in_flight=False,
        audio_playback_busy=False,
        consumes_canonical_slot=True,
        canonical_audio_started=False,
        explicit_multipart=False,
        single_flight_block_reason="",
        already_delivered=False,
        preference_recall_lock_blocked=False,
        canonical_key_already_created=False,
        has_safety_override=False,
        suppression_active=False,
        normalized_origin="assistant_message",
        awaiting_transcript_final=True,
    )

    assert decision.action is ResponseCreateDecisionAction.SEND
    assert decision.reason_code == "direct_send"
