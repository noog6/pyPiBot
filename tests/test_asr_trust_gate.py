from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime.asr_trust import (
    build_utterance_trust_snapshot,
    should_clarify,
    topic_mismatch_detected,
)


def test_should_clarify_low_confidence() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-473",
        turn_id="turn-2",
        input_event_key="evt-1",
        transcript_text="Hey Theo what is my editor",
        utterance_duration_ms=1200,
        asr_meta={"confidence": 0.4},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=snapshot.transcript_text,
        snapshot=snapshot,
        min_confidence=0.65,
    )

    assert clarify is True
    assert reason == "low_conf"


def test_should_clarify_very_short_utterance() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-473",
        turn_id="turn-3",
        input_event_key="evt-2",
        transcript_text="Yes",
        utterance_duration_ms=250,
        asr_meta={},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text="Yes",
        snapshot=snapshot,
        min_confidence=0.65,
    )

    assert clarify is True
    assert reason == "short_utterance"


def test_topic_mismatch_triggers_clarify() -> None:
    assert topic_mismatch_detected(
        "what color my pants are",
        "I don't have any stored information about a favorite color.",
    )


def test_no_clarify_on_normal_high_confidence() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-473",
        turn_id="turn-2",
        input_event_key="evt-1",
        transcript_text="do you know what my favorite editor is",
        utterance_duration_ms=1100,
        asr_meta={"confidence": 0.9},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=snapshot.transcript_text,
        snapshot=snapshot,
        min_confidence=0.65,
    )

    assert clarify is False
    assert reason == "none"
