from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime.lifecycle_state import LifecycleStateCoordinator
from ai.realtime.types import CanonicalResponseState


def test_canonical_utterance_key_normalization_and_bindings() -> None:
    coordinator = LifecycleStateCoordinator()

    canonical = coordinator.canonical_utterance_key(
        run_id=" run-1 ",
        turn_id=" turn_1 ",
        input_event_key=" input_evt_1 ",
    )
    assert canonical == "run-1:turn_1:input_evt_1"

    fallback = coordinator.canonical_utterance_key(run_id="", turn_id="", input_event_key="")
    assert fallback == "run-unknown:turn-unknown:synthetic:turn-unknown"

    bindings: dict[str, str] = {}
    coordinator.bind_active_input_event_key_for_turn(bindings, turn_id=" turn_1 ", input_event_key=" input_evt_1 ")
    coordinator.bind_active_input_event_key_for_turn(bindings, turn_id="", input_event_key="ignored")
    assert coordinator.active_input_event_key_for_turn(bindings, turn_id="turn_1") == "input_evt_1"


def test_response_obligation_transitions_set_and_clear() -> None:
    coordinator = LifecycleStateCoordinator()
    state_store: dict[str, CanonicalResponseState] = {}

    obligation_key = coordinator.response_obligation_key(run_id="run-1", turn_id="turn_1", input_event_key="evt_1")

    set_state, present_before = coordinator.transition_response_obligation(
        state_store=state_store,
        obligation_key=obligation_key,
        turn_id="turn_1",
        input_event_key="evt_1",
        source="assistant_message",
        present=True,
    )
    assert present_before is False
    assert set_state.obligation_present is True
    assert set_state.obligation is not None
    assert set_state.obligation["source"] == "assistant_message"

    cleared_state, present_before_clear = coordinator.transition_response_obligation(
        state_store=state_store,
        obligation_key=obligation_key,
        turn_id="turn_1",
        input_event_key="evt_1",
        source="assistant_message",
        present=False,
    )
    assert present_before_clear is True
    assert cleared_state.obligation_present is False
    assert cleared_state.obligation is None
