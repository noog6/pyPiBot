from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime.asr_trust import (
    build_utterance_trust_snapshot,
    extract_topic_anchors,
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


def test_extract_topic_anchors_uses_configured_assistant_name_as_noise_token() -> None:
    anchors = extract_topic_anchors("hey nova what is my favorite editor", assistant_name="Nova")
    assert "nova" not in anchors
    assert "editor" in anchors


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


def test_compound_motion_visual_request_does_not_force_visual_unavailable_when_camera_is_active() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-609",
        turn_id="turn-2",
        input_event_key="evt-compound",
        transcript_text="move back to center and tell me what you see in front of you",
        utterance_duration_ms=4200,
        asr_meta={"confidence": 0.92},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=snapshot.transcript_text,
        snapshot=snapshot,
        min_confidence=0.65,
        camera_available=True,
        camera_recent=False,
    )

    assert clarify is False
    assert reason == "compound_motion_visual_defer"


def test_phatic_thanks_does_not_route_to_clarify() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-700",
        turn_id="turn-ack",
        input_event_key="evt-thanks",
        transcript_text="Thanks.",
        utterance_duration_ms=180,
        asr_meta={},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=snapshot.transcript_text,
        snapshot=snapshot,
        min_confidence=0.65,
    )

    assert clarify is False
    assert reason == "phatic_acknowledgement"


def test_extended_phatic_acknowledgements_do_not_route_to_clarify() -> None:
    for transcript_text in ("Gotcha.", "sounds good", "all good", "perfect", "alright"):
        snapshot = build_utterance_trust_snapshot(
            run_id="run-701",
            turn_id="turn-ack-extended",
            input_event_key=f"evt-{transcript_text}",
            transcript_text=transcript_text,
            utterance_duration_ms=1800,
            asr_meta={},
            short_utterance_ms=450,
        )

        clarify, reason = should_clarify(
            transcript_text=snapshot.transcript_text,
            snapshot=snapshot,
            min_confidence=0.65,
        )

        assert clarify is False
        assert reason == "phatic_acknowledgement"


def test_web_search_phrasing_does_not_set_visual_question_flag() -> None:
    for transcript_text in (
        "search online for Little Caesars menu",
        "see if you can find the Little Caesars menu online",
    ):
        snapshot = build_utterance_trust_snapshot(
            run_id="run-910",
            turn_id="turn-search",
            input_event_key=f"evt-{transcript_text}",
            transcript_text=transcript_text,
            utterance_duration_ms=2200,
            asr_meta={"confidence": 0.95},
            short_utterance_ms=450,
        )

        clarify, reason = should_clarify(
            transcript_text=snapshot.transcript_text,
            snapshot=snapshot,
            min_confidence=0.65,
            camera_available=False,
            camera_recent=False,
        )

        assert snapshot.visual_question is False
        assert clarify is False
        assert reason == "none"


def test_true_visual_requests_still_set_visual_question_flag() -> None:
    for transcript_text in (
        "What do you see?",
        "Can you look at this?",
        "Is anything in front of you?",
    ):
        snapshot = build_utterance_trust_snapshot(
            run_id="run-911",
            turn_id="turn-vision",
            input_event_key=f"evt-{transcript_text}",
            transcript_text=transcript_text,
            utterance_duration_ms=1800,
            asr_meta={"confidence": 0.95},
            short_utterance_ms=450,
        )

        clarify, reason = should_clarify(
            transcript_text=snapshot.transcript_text,
            snapshot=snapshot,
            min_confidence=0.65,
            camera_available=False,
            camera_recent=False,
        )

        assert snapshot.visual_question is True
        assert clarify is True
        assert reason == "visual_unavailable"
