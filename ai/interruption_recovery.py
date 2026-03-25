"""Deterministic policy for interrupted pre-evidence tool-output followups."""

from __future__ import annotations

from enum import Enum


class InterruptedToolOutputResolution(str, Enum):
    RESUME_AFTER_NOISE = "interruption_resume_after_noise"
    MERGED_INTO_FOLLOWUP = "interruption_merged_into_followup"
    SUPERSEDED_BY_NEW_TURN = "interruption_superseded_by_new_turn"


_SUPERSEDE_PHRASES = (
    "never mind",
    "nevermind",
    "ignore that",
    "ignore it",
    "scratch that",
    "stop that",
    "different question",
    "new question",
    "change topic",
    "instead",
)


def decide_interrupted_tool_output_resolution(*, transcript: str, transcript_word_count: int) -> InterruptedToolOutputResolution:
    """Resolve interrupted deferred tool-output candidates after transcript-final.

    Rules are intentionally bounded and deterministic:
    - empty/zero-word transcript -> resume the interrupted answer
    - explicit supersede phrases -> drop the interrupted answer
    - otherwise -> merge interrupted answer into the new turn
    """

    if int(transcript_word_count or 0) <= 0:
        return InterruptedToolOutputResolution.RESUME_AFTER_NOISE

    normalized = str(transcript or "").strip().lower()
    if not normalized:
        return InterruptedToolOutputResolution.RESUME_AFTER_NOISE

    if any(marker in normalized for marker in _SUPERSEDE_PHRASES):
        return InterruptedToolOutputResolution.SUPERSEDED_BY_NEW_TURN

    return InterruptedToolOutputResolution.MERGED_INTO_FOLLOWUP
