from __future__ import annotations

import sys
import types
from types import SimpleNamespace

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.initiative_posture import InitiativePostureSelector
from ai.quiet_intent import QuietIntentMode
from ai.realtime_api import RealtimeAPI
from interaction import InteractionState


class _ConfirmationRuntimeStub:
    def __init__(self, pending: bool) -> None:
        self._pending = pending

    def is_confirmation_pending(self) -> bool:
        return self._pending


def _build_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._initiative_posture_selector = InitiativePostureSelector()
    api._latest_initiative_posture_decision = None
    api._latest_initiative_posture_log_fingerprint = None
    api._latest_quiet_intent_decision = SimpleNamespace(mode=QuietIntentMode.COMPANION_PRESENCE)
    api._last_user_input_time = 0.0
    api._last_user_input_text = "Can you summarize this?"
    api.response_in_progress = False
    api._response_in_flight = False
    api._confirmation_runtime = _ConfirmationRuntimeStub(pending=False)
    api._current_run_id = lambda: "run-initiative"
    api._current_turn_id_or_unknown = lambda: "turn-initiative"
    api.get_continuity_brief = lambda **_kwargs: SimpleNamespace(stance="assisting_query")
    api.get_continuity_turn_settlement = lambda **_kwargs: SimpleNamespace(settlement_state="settled")
    return api


def test_refresh_initiative_posture_produces_decision() -> None:
    api = _build_api()

    RealtimeAPI._refresh_initiative_posture(api, state=InteractionState.IDLE)

    assert api._latest_initiative_posture_decision is not None
    assert api._latest_initiative_posture_decision.initiative_posture.value == "answer_directly"


def test_attach_initiative_posture_metadata_is_consultative_only() -> None:
    api = _build_api()
    RealtimeAPI._refresh_initiative_posture(api, state=InteractionState.IDLE)

    event = {"type": "response.create", "response": {"metadata": {"tool_followup": "true"}}}
    RealtimeAPI._attach_initiative_posture_metadata(api, response_create_event=event)

    metadata = event["response"]["metadata"]
    assert metadata["initiative_posture"] == "answer_directly"
    assert metadata["initiative_confidence_band"] in {"low", "medium", "high"}
    assert "initiative_reason_codes" in metadata
    assert metadata["tool_followup"] == "true"
