from __future__ import annotations

import asyncio
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai import realtime_api as realtime_api_module
from ai.realtime.asr_trust import build_utterance_trust_snapshot, should_clarify
from ai.realtime_api import RealtimeAPI


class _Transport:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    async def send_json(self, _ws, event: dict[str, object]) -> None:
        self.sent.append(event)


def test_one_word_ambiguous_utterance_prefers_clarify() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-526",
        turn_id="turn-4",
        input_event_key="item-1",
        transcript_text="Yoren",
        utterance_duration_ms=1900,
        asr_meta={},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=snapshot.transcript_text,
        snapshot=snapshot,
        min_confidence=0.65,
    )

    assert clarify is True
    assert reason == "low_semantic_confidence"


def test_short_greeting_stays_natural_without_clarify() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-526",
        turn_id="turn-5",
        input_event_key="item-2",
        transcript_text="hey",
        utterance_duration_ms=1500,
        asr_meta={},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=snapshot.transcript_text,
        snapshot=snapshot,
        min_confidence=0.65,
    )

    assert clarify is False
    assert reason == "none"


def test_short_clear_command_not_blocked() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-526",
        turn_id="turn-6",
        input_event_key="item-3",
        transcript_text="look center",
        utterance_duration_ms=1700,
        asr_meta={},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=snapshot.transcript_text,
        snapshot=snapshot,
        min_confidence=0.65,
    )

    assert clarify is False
    assert reason == "none"


def test_visual_question_snapshot_lexical_detection_matches_current_api() -> None:
    positive_cases = [
        "what do you see right now",
        "can you see it now",
        "what color shirt",
    ]
    negative_cases = [
        "what did you hear earlier",
        "what percentage is your battery level",
        "favorite editor",
    ]

    for transcript in positive_cases:
        snapshot = build_utterance_trust_snapshot(
            run_id="run-526",
            turn_id="turn-visual",
            input_event_key="item-positive",
            transcript_text=transcript,
            utterance_duration_ms=1500,
            asr_meta={},
            short_utterance_ms=450,
        )
        assert snapshot.visual_question is True

    for transcript in negative_cases:
        snapshot = build_utterance_trust_snapshot(
            run_id="run-526",
            turn_id="turn-non-visual",
            input_event_key="item-negative",
            transcript_text=transcript,
            utterance_duration_ms=1500,
            asr_meta={},
            short_utterance_ms=450,
        )
        assert snapshot.visual_question is False


def test_should_clarify_visual_question_paths_cover_visual_unavailable_and_defer() -> None:
    snapshot = build_utterance_trust_snapshot(
        run_id="run-526",
        turn_id="turn-visual-1",
        input_event_key="item-visual-1",
        transcript_text="what do you see right now",
        utterance_duration_ms=1700,
        asr_meta={},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=snapshot.transcript_text,
        snapshot=snapshot,
        min_confidence=0.65,
        camera_available=False,
        camera_recent=False,
    )

    assert clarify is True
    assert reason == "visual_unavailable"

    compound_snapshot = build_utterance_trust_snapshot(
        run_id="run-526",
        turn_id="turn-visual-2",
        input_event_key="item-visual-2",
        transcript_text="go back to center and tell me what you see",
        utterance_duration_ms=1700,
        asr_meta={},
        short_utterance_ms=450,
    )

    clarify, reason = should_clarify(
        transcript_text=compound_snapshot.transcript_text,
        snapshot=compound_snapshot,
        min_confidence=0.65,
        camera_available=True,
        camera_recent=False,
    )

    assert clarify is False
    assert reason == "compound_motion_visual_defer"


def test_preference_recall_query_not_treated_as_ambiguous() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 450
    api._asr_verify_min_confidence = 0.65
    api._is_memory_intent = lambda _text: False
    api.get_vision_state = lambda: {"available": False, "can_capture": False}
    api._set_response_gating_verdict = lambda **_kwargs: None
    api._pending_server_auto_response_for_turn = lambda **_kwargs: None

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="favorite editor",
            websocket=object(),
            turn_id="turn-7",
            input_event_key="item-4",
            snapshot={"run_id": "run-526", "utterance_duration_ms": 1800},
        )
    )

    assert clarified is False


def test_context_enrichment_suppressed_for_ambiguous_input(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 2
    api._asr_verify_short_utterance_ms = 450
    api._asr_verify_min_confidence = 0.65
    api._is_memory_intent = lambda _text: False
    api._set_response_gating_verdict = lambda **_kwargs: None
    api._pending_server_auto_response_for_turn = lambda **_kwargs: None
    api._current_run_id = lambda: "run-526"
    api._stale_response_ids = lambda: set()
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: _Transport()
    api.get_vision_state = lambda: {"available": True, "can_capture": True}
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._assistant_reply_response_id = None

    sent_messages: list[tuple[str, dict[str, str]]] = []

    async def _send_assistant_message(msg: str, _ws, *, response_metadata=None, **_kwargs):
        sent_messages.append((msg, response_metadata or {}))

    api.send_assistant_message = _send_assistant_message

    info_logs: list[str] = []

    def _capture_info(message: str, *args, **_kwargs) -> None:
        info_logs.append(message % args if args else message)

    monkeypatch.setattr(realtime_api_module.logger, "info", _capture_info)

    clarified = asyncio.run(api._maybe_verify_on_risk_clarify(
        transcript="Yoren",
        websocket=object(),
        turn_id="turn-8",
        input_event_key="item-5",
        snapshot={"run_id": "run-526", "utterance_duration_ms": 1800},
    ))

    assert clarified is True
    assert sent_messages
    text, _metadata = sent_messages[0]
    assert "board games" not in text.lower()
    assert "not sure what you mean" in text.lower()
    assert any("context_enrichment_suppressed" in entry for entry in info_logs)


def test_bounded_clarify_adds_response_instruction_guardrail() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    event = api._bounded_clarify_response_create_event(
        message="I heard you, but I’m not sure what you mean yet. Could you be a bit more specific?",
        metadata={
            "origin": "assistant_message",
            "trigger": "asr_verify_on_risk",
            "reason": "low_semantic_confidence",
            "input_event_key": "item-6:clarify",
        },
    )

    instructions = event["response"].get("instructions", "")
    assert "bounded clarify mode" in instructions.lower()
    assert "nothing else" in instructions.lower()
    assert "imu" in instructions.lower()


def test_is_bounded_clarify_mode_includes_visual_unavailable() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    assert (
        api._is_bounded_clarify_mode(
            {
                "trigger": "asr_verify_on_risk",
                "reason": "visual_unavailable",
            }
        )
        is True
    )


def test_normalize_verify_clarify_message_visual_unavailable_uses_live_vision_state() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.get_vision_state = lambda: {
        "available": False,
        "last_frame_age_ms": None,
        "queued_frame_count": 0,
        "can_capture": False,
        "camera_active": False,
    }

    normalized = api._normalize_verify_clarify_message(
        message="stale text",
        metadata={
            "trigger": "asr_verify_on_risk",
            "reason": "visual_unavailable",
            "turn_id": "turn-legacy",
        },
    )

    assert normalized == "I can’t see right now. Want me to take a quick look with the camera?"


def test_normalize_verify_clarify_message_non_visual_unavailable_is_unchanged() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.get_vision_state = lambda: {
        "available": False,
        "last_frame_age_ms": None,
        "queued_frame_count": 0,
        "can_capture": False,
        "camera_active": False,
    }

    message = "I heard you, but I’m not sure what you mean yet. Could you be a bit more specific?"
    normalized = api._normalize_verify_clarify_message(
        message=message,
        metadata={
            "trigger": "asr_verify_on_risk",
            "reason": "low_semantic_confidence",
        },
    )

    assert normalized == message


def test_visual_unavailable_clarify_uses_dedicated_bounded_instruction_guardrail() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    event = api._bounded_clarify_response_create_event(
        message="I can’t see right now. Want me to take a quick look with the camera?",
        metadata={
            "origin": "assistant_message",
            "trigger": "asr_verify_on_risk",
            "reason": "visual_unavailable",
            "input_event_key": "item-7:clarify",
        },
    )

    instructions = event["response"].get("instructions", "")
    assert "bounded clarify mode" in instructions.lower()
    assert "exactly this clarify sentence" in instructions.lower()
    assert "do not assert or describe any scene" in instructions.lower()
    assert "do not add a second sentence" in instructions.lower()
    assert "short and non-assertive" in instructions.lower()


def test_low_semantic_bounded_clarify_instruction_is_unchanged() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    event = api._bounded_clarify_response_create_event(
        message="I heard you, but I’m not sure what you mean yet. Could you be a bit more specific?",
        metadata={
            "origin": "assistant_message",
            "trigger": "asr_verify_on_risk",
            "reason": "low_semantic_confidence",
            "input_event_key": "item-8:clarify",
        },
    )

    assert event["response"].get("instructions", "") == (
        "You are in bounded clarify mode. Speak exactly this one sentence and nothing else: "
        "'I heard you, but I’m not sure what you mean yet. Could you be a bit more specific?'. "
        "Do not add scene, IMU, memory, or environment commentary."
    )


def test_short_utterance_bounded_clarify_instruction_is_unchanged() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    event = api._bounded_clarify_response_create_event(
        message="Sorry, I only caught a tiny bit. Could you repeat that?",
        metadata={
            "origin": "assistant_message",
            "trigger": "asr_verify_on_risk",
            "reason": "short_utterance",
            "input_event_key": "item-9:clarify",
        },
    )

    assert event["response"].get("instructions", "") == (
        "You are in bounded clarify mode. Speak exactly this one sentence and nothing else: "
        "'Sorry, I only caught a tiny bit. Could you repeat that?'. "
        "Do not add scene, IMU, memory, or environment commentary."
    )
