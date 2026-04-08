from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_INITIATIVE_POSTURE_PATH = Path(__file__).resolve().parents[1] / "ai" / "initiative_posture.py"
_INITIATIVE_POSTURE_SPEC = importlib.util.spec_from_file_location("initiative_posture_module", _INITIATIVE_POSTURE_PATH)
assert _INITIATIVE_POSTURE_SPEC is not None and _INITIATIVE_POSTURE_SPEC.loader is not None
_INITIATIVE_POSTURE_MODULE = importlib.util.module_from_spec(_INITIATIVE_POSTURE_SPEC)
sys.modules[_INITIATIVE_POSTURE_SPEC.name] = _INITIATIVE_POSTURE_MODULE
_INITIATIVE_POSTURE_SPEC.loader.exec_module(_INITIATIVE_POSTURE_MODULE)

InitiativePosture = _INITIATIVE_POSTURE_MODULE.InitiativePosture
InitiativePostureInputs = _INITIATIVE_POSTURE_MODULE.InitiativePostureInputs
InitiativePostureSelector = _INITIATIVE_POSTURE_MODULE.InitiativePostureSelector
from interaction import InteractionState


def _inputs(**overrides: object) -> InitiativePostureInputs:
    base: dict[str, object] = {
        "interaction_state": InteractionState.IDLE,
        "conversation_active": True,
        "continuity_stance": "assisting_query",
        "recent_utterance_flags": ("direct_question",),
        "followthrough_active": False,
        "confirmation_pending": False,
        "response_in_flight": False,
        "quiet_intent_mode": "companion_presence",
    }
    base.update(overrides)
    return InitiativePostureInputs(**base)


def test_direct_question_without_blockers_prefers_answer_directly() -> None:
    decision = InitiativePostureSelector().select(_inputs())

    assert decision.initiative_posture == InitiativePosture.ANSWER_DIRECTLY
    assert "direct_request_signal" in decision.reason_codes


def test_ambiguous_request_prefers_clarify_first() -> None:
    decision = InitiativePostureSelector().select(
        _inputs(recent_utterance_flags=("direct_question", "ambiguous_request"))
    )

    assert decision.initiative_posture == InitiativePosture.CLARIFY_FIRST
    assert "ambiguous_request" in decision.reason_codes


def test_followthrough_open_prefers_continue_followthrough() -> None:
    decision = InitiativePostureSelector().select(
        _inputs(
            followthrough_active=True,
            recent_utterance_flags=(),
            continuity_stance="awaiting_tool",
        )
    )

    assert decision.initiative_posture == InitiativePosture.CONTINUE_FOLLOWTHROUGH
    assert "followthrough_chain_open" in decision.reason_codes


def test_idle_no_request_prefers_await_or_observe() -> None:
    decision = InitiativePostureSelector().select(
        _inputs(
            conversation_active=False,
            recent_utterance_flags=(),
            quiet_intent_mode="observer",
            continuity_stance="idle",
        )
    )

    assert decision.initiative_posture in {InitiativePosture.AWAIT_USER, InitiativePosture.OBSERVE_ONLY}


def test_confirmation_pending_suppresses_to_observe_only() -> None:
    decision = InitiativePostureSelector().select(
        _inputs(
            confirmation_pending=True,
            followthrough_active=True,
            recent_utterance_flags=("direct_question",),
        )
    )

    assert decision.initiative_posture == InitiativePosture.OBSERVE_ONLY
    assert "confirmation_pending" in decision.reason_codes


def test_output_is_consultative_only_shape() -> None:
    decision = InitiativePostureSelector().select(_inputs())
    payload = decision.to_consultative_hint()

    assert set(payload) == {"initiative_posture", "confidence", "confidence_band", "reason_codes"}
    assert "tool" not in " ".join(payload["reason_codes"])
