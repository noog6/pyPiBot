import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.interruption_recovery import (
    InterruptedToolOutputResolution,
    decide_interrupted_tool_output_resolution,
)


def test_interruption_policy_resumes_after_noise_empty_transcript() -> None:
    decision = decide_interrupted_tool_output_resolution(transcript="", transcript_word_count=0)

    assert decision is InterruptedToolOutputResolution.RESUME_AFTER_NOISE


def test_interruption_policy_merges_related_followup_by_default() -> None:
    decision = decide_interrupted_tool_output_resolution(
        transcript="yes and also include the battery trend",
        transcript_word_count=8,
    )

    assert decision is InterruptedToolOutputResolution.MERGED_INTO_FOLLOWUP


def test_interruption_policy_supersedes_on_explicit_redirect_phrase() -> None:
    decision = decide_interrupted_tool_output_resolution(
        transcript="never mind, different question",
        transcript_word_count=4,
    )

    assert decision is InterruptedToolOutputResolution.SUPERSEDED_BY_NEW_TURN
