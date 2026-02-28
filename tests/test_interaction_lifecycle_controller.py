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


def test_replace_marks_old_key_replaced() -> None:
    controller = InteractionLifecycleController()
    old_key = "run-1:turn-1:synthetic"
    new_key = "run-1:turn-1:item-1"
    controller.on_transcript_final(old_key)
    controller.on_replaced(old_key, new_key)

    assert controller.state_for(old_key) == InteractionLifecycleState.REPLACED
    assert controller.decide_response_create_allow(old_key, origin="assistant_message").action is LifecycleDecisionAction.CANCEL
    assert controller.decide_response_create_allow(new_key, origin="assistant_message").action is LifecycleDecisionAction.ALLOW

