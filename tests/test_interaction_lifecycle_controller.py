from ai.interaction_lifecycle_controller import (
    InteractionLifecycleController,
    InteractionLifecycleState,
    LifecycleDecisionAction,
)


def test_server_auto_create_deferred_until_transcript_final() -> None:
    controller = InteractionLifecycleController()
    key = "run-1:turn-1:item-1"

    deferred = controller.decide_response_create_allow(key, origin="server_auto")
    assert deferred.action is LifecycleDecisionAction.DEFER

    controller.on_transcript_final(key)
    allowed = controller.decide_response_create_allow(key, origin="server_auto")
    assert allowed.action is LifecycleDecisionAction.ALLOW


def test_audio_and_done_transitions_are_deterministic() -> None:
    controller = InteractionLifecycleController()
    key = "run-1:turn-1:item-1"
    controller.on_transcript_final(key)
    controller.on_response_created(key, origin="assistant_message")

    audio = controller.on_audio_delta(key)
    assert audio.action is LifecycleDecisionAction.ALLOW
    assert controller.state_for(key) == InteractionLifecycleState.AUDIO_STARTED

    controller.on_response_done(key)
    assert controller.state_for(key) == InteractionLifecycleState.DONE

    cancelled = controller.on_audio_delta(key)
    assert cancelled.action is LifecycleDecisionAction.CANCEL


def test_audio_delta_transitions_once_then_stays_steady_state() -> None:
    controller = InteractionLifecycleController()
    key = "run-1:turn-1:item-1"
    controller.on_response_created(key, origin="assistant_message")

    first = controller.on_audio_delta(key)
    second = controller.on_audio_delta(key)
    third = controller.on_audio_delta(key)

    assert first.action is LifecycleDecisionAction.ALLOW
    assert first.reason == "transitioned=audio_started"
    assert second.action is LifecycleDecisionAction.ALLOW
    assert second.reason == "state=audio_started"
    assert third.action is LifecycleDecisionAction.ALLOW
    assert third.reason == "state=audio_started"


def test_replace_marks_old_key_replaced() -> None:
    controller = InteractionLifecycleController()
    old_key = "run-1:turn-1:synthetic"
    new_key = "run-1:turn-1:item-1"
    controller.on_transcript_final(old_key)
    controller.on_replaced(old_key, new_key)

    assert controller.state_for(old_key) == InteractionLifecycleState.REPLACED
    assert controller.decide_response_create_allow(old_key, origin="assistant_message").action is LifecycleDecisionAction.CANCEL
    assert controller.decide_response_create_allow(new_key, origin="assistant_message").action is LifecycleDecisionAction.ALLOW


def test_key_rebound_moves_record_without_terminal_replace() -> None:
    controller = InteractionLifecycleController()
    old_key = "run-1:turn-1:synthetic"
    new_key = "run-1:turn-1:item-1"

    controller.on_transcript_final(old_key)
    controller.on_response_created(old_key, origin="server_auto")
    controller.on_key_rebound(old_key, new_key)

    assert controller.state_for(old_key) == InteractionLifecycleState.NEW
    assert controller.state_for(new_key) == InteractionLifecycleState.SERVER_AUTO_CREATED
    assert controller.decide_response_create_allow(new_key, origin="assistant_message").action is LifecycleDecisionAction.CANCEL


def test_response_created_transition_reason_is_origin_neutral() -> None:
    controller = InteractionLifecycleController()
    key = "run-1:turn-1:item-1"

    decision = controller.on_response_created(key, origin="assistant_message")

    assert decision.action is LifecycleDecisionAction.ALLOW
    assert decision.reason == "transitioned=response_created"
